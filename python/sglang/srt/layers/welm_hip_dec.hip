// WeLM paged decode attention, split-KV partial kernel on gfx938 matrix cores.
// KV layout (HCU legacy): K [pages, KVH, 64, D], V [pages, KVH, D, 64]; D = 256.
// One wave = one (batch, kv head, split of CP pages); G <= 16 query heads per
// kv head are the M rows of a 16x16x16 mmac. Writes normalized partial O and
// natural-log LSE in the same format as welmv4_fast_op._welm_dec_attn_reduce_kernel.
//
// mmac_f32_16x16x16 layouts (probed): lane l, c = l % 16, g = l / 16, elem i:
//   A[m = c][k = 4g + i], B[k = 4g + i][n = c], C[m = c][n = g + 4i].
// Token / d permutations are chosen so every operand is loaded straight from
// global memory with 16-byte loads and S (C layout) is reused as P (A layout):
//   QK subtile s: B column n is token 16*(n%4) + 4s + n/4, so the lane's S
//   holds tokens 16g + 4s + i == the PV A operand for k-step s.
//   QK k index for k-step kk = 4j + q: d = 64j + 16g + 4q + i.
//   PV n-tile nn: B column n = d c + 16nn; C holds d = g + 4i + 16nn.
// fp8 e4m3 -> fp16 is exact via bit placement (value * 2^-8); the 2^8 is folded
// into the q scale (K) and the reduce kernel's v_scale (V).
#include <hip/hip_runtime.h>
#include <stdint.h>

typedef _Float16 h4 __attribute__((ext_vector_type(4)));
typedef short s4 __attribute__((ext_vector_type(4)));
typedef float f4 __attribute__((ext_vector_type(4)));

#define D 256
#define PAGE 64

__device__ __forceinline__ h4 e4m3x4_to_h4(uint32_t x) {
  uint32_t t01 = ((x & 0xffu) << 8) | ((x & 0xff00u) << 16);
  uint32_t t23 = ((x >> 8) & 0xff00u) | (x & 0xff000000u);
  t01 = ((t01 >> 1) & 0x3f803f80u) | (t01 & 0x80008000u);
  t23 = ((t23 >> 1) & 0x3f803f80u) | (t23 & 0x80008000u);
  uint2 u = make_uint2(t01, t23);
  return *reinterpret_cast<h4*>(&u);
}

__device__ __forceinline__ float bf16_to_f(uint16_t v) {
  return __uint_as_float(((uint32_t)v) << 16);
}

__device__ __forceinline__ float shfl_xor(float v, int m) { return __shfl_xor(v, m, 64); }

template <int CP>
__global__ __launch_bounds__(64) void welm_dec_fp8_partial(
    const uint16_t* __restrict__ Q, const uint8_t* __restrict__ K, const uint8_t* __restrict__ V,
    const int* __restrict__ BT, const int* __restrict__ SEQK, float* __restrict__ MID_O,
    float* __restrict__ MID_LSE, float qscale,  // softmax_scale * k_scale * 256 * log2(e)
    long sqb, long sqh, long skb, long skh, long skp, long svb, long svh, long svd, long sbt,
    long smob, long smoh, long smos, long smlb, long smlh, int KVH, int G, int window_left) {
  const int l = threadIdx.x, c = l & 15, g = l >> 4;
  const int b = blockIdx.x / KVH, kvh = blockIdx.x % KVH, slot = blockIdx.y;
  const int kv_len = SEQK[b];
  const int w_lo = window_left >= 0 ? max(kv_len - 1 - window_left, 0) : 0;
  const int split = slot + (window_left >= 0 ? w_lo / (CP * PAGE) : 0);
  const int lo = split * CP * PAGE;
  if (lo >= kv_len) return;
  const int hi = min(lo + CP * PAGE, kv_len);
  const int vlo = max(lo, w_lo);

  // Q A operand: 16 k-steps x 4 halves, head c, d = 64j + 16g + 4q + i.
  h4 qa[16];
  {
    const uint16_t* qp = Q + (long)b * sqb + (long)(kvh * G + c) * sqh;
    for (int j = 0; j < 4; j++) {
      uint4 raw[2];
      if (c < G) {
        const uint4* p = reinterpret_cast<const uint4*>(qp + 64 * j + 16 * g);
        raw[0] = p[0]; raw[1] = p[1];
      } else {
        raw[0] = raw[1] = make_uint4(0, 0, 0, 0);
      }
      const uint16_t* e = reinterpret_cast<const uint16_t*>(raw);
      for (int q = 0; q < 4; q++)
        for (int i = 0; i < 4; i++) qa[4 * j + q][i] = (_Float16)(bf16_to_f(e[4 * q + i]) * qscale);
    }
  }

  f4 acc[16];
  for (int nn = 0; nn < 16; nn++) acc[nn] = f4{0.f, 0.f, 0.f, 0.f};
  float m_i = -INFINITY, l_lane = 0.f;

  const int* bt = BT + (long)b * sbt;
  for (int pg = lo / PAGE; pg * PAGE < hi; pg++) {
    const long page = bt[pg];
    const uint8_t* kp = K + page * skb + (long)kvh * skh;
    const uint8_t* vp = V + page * svb + (long)kvh * svh;
    uint4 kr[4][4], vr[16];
    for (int s = 0; s < 4; s++) {
      const int tok = 16 * (c & 3) + 4 * s + (c >> 2);
      const uint4* p = reinterpret_cast<const uint4*>(kp + (long)tok * skp + 16 * g);
      for (int j = 0; j < 4; j++) kr[s][j] = p[4 * j];
    }

    f4 S[4];
    for (int s = 0; s < 4; s++) {
      f4 a = f4{0.f, 0.f, 0.f, 0.f};
      for (int j = 0; j < 4; j++) {
        const uint32_t* w = reinterpret_cast<const uint32_t*>(&kr[s][j]);
        for (int q = 0; q < 4; q++) a = __builtin_hcu_mmac_f32_16x16x16_f16(qa[4 * j + q], e4m3x4_to_h4(w[q]), a);
      }
      S[s] = a;
    }
    for (int nn = 0; nn < 16; nn++)
      vr[nn] = *reinterpret_cast<const uint4*>(vp + (long)(c + 16 * nn) * svd + 16 * g);
    // mask + online softmax (rows = head c, lane holds tokens 16g + 4s + i)
    const int base = pg * PAGE + 16 * g;
    float mx = -INFINITY;
    for (int s = 0; s < 4; s++)
      for (int i = 0; i < 4; i++) {
        const int t = base + 4 * s + i;
        if (t < vlo || t >= hi) S[s][i] = -INFINITY;
        mx = fmaxf(mx, S[s][i]);
      }
    mx = fmaxf(mx, shfl_xor(mx, 16));
    mx = fmaxf(mx, shfl_xor(mx, 32));
    const float m_new = fmaxf(m_i, mx);
    const float m_safe = m_new == -INFINITY ? 0.f : m_new;
    const float alpha = exp2f(m_i - m_safe);
    m_i = m_new;
    h4 pa[4];
    float ls = 0.f;
    for (int s = 0; s < 4; s++)
      for (int i = 0; i < 4; i++) {
        const float p = exp2f(S[s][i] - m_safe);
        ls += p;
        pa[s][i] = (_Float16)p;
      }
    l_lane = l_lane * alpha + ls;
    for (int nn = 0; nn < 16; nn++) {
      f4 a = acc[nn] * alpha;
      const uint32_t* w = reinterpret_cast<const uint32_t*>(&vr[nn]);
      for (int s = 0; s < 4; s++) a = __builtin_hcu_mmac_f32_16x16x16_f16(pa[s], e4m3x4_to_h4(w[s]), a);
      acc[nn] = a;
    }
  }
  float l_tot = l_lane + shfl_xor(l_lane, 16);
  l_tot += shfl_xor(l_tot, 32);
  if (c >= G) return;
  const int head = kvh * G + c;
  const float inv = l_tot > 0.f ? 1.f / l_tot : 0.f;
  float* mo = MID_O + (long)b * smob + (long)head * smoh + (long)slot * smos;
  for (int nn = 0; nn < 16; nn++)
    for (int i = 0; i < 4; i++) mo[g + 4 * i + 16 * nn] = acc[nn][i] * inv;
  if (g == 0)
    MID_LSE[(long)b * smlb + (long)head * smlh + slot] = m_i * 0.6931471805599453f + logf(l_tot);
}


typedef unsigned short u4 __attribute__((__vector_size__(4 * sizeof(unsigned short))));

__device__ __forceinline__ uint32_t f2_to_bf2(float a, float b) {
  uint32_t ua = __float_as_uint(a), ub = __float_as_uint(b);
  ua = (ua + 0x7fffu + ((ua >> 16) & 1u)) >> 16;
  ub = (ub + 0x7fffu + ((ub >> 16) & 1u)) & 0xffff0000u;
  return ua | ub;
}

// bf16 KV variant: same permutations; Q, K, V, P in bf16 on the bf16 mmac.
template <int CP>
__global__ __launch_bounds__(64) void welm_dec_bf16_partial(
    const uint16_t* __restrict__ Q, const uint16_t* __restrict__ K, const uint16_t* __restrict__ V,
    const int* __restrict__ BT, const int* __restrict__ SEQK, float* __restrict__ MID_O,
    float* __restrict__ MID_LSE, float qscale,  // softmax_scale * k_scale * log2(e)
    long sqb, long sqh, long skb, long skh, long skp, long svb, long svh, long svd, long sbt,
    long smob, long smoh, long smos, long smlb, long smlh, int KVH, int G, int window_left) {
  const int l = threadIdx.x, c = l & 15, g = l >> 4;
  const int b = blockIdx.x / KVH, kvh = blockIdx.x % KVH, slot = blockIdx.y;
  const int kv_len = SEQK[b];
  const int w_lo = window_left >= 0 ? max(kv_len - 1 - window_left, 0) : 0;
  const int split = slot + (window_left >= 0 ? w_lo / (CP * PAGE) : 0);
  const int lo = split * CP * PAGE;
  if (lo >= kv_len) return;
  const int hi = min(lo + CP * PAGE, kv_len);
  const int vlo = max(lo, w_lo);

  u4 qa[16];
  {
    const uint16_t* qp = Q + (long)b * sqb + (long)(kvh * G + c) * sqh;
    for (int j = 0; j < 4; j++) {
      uint4 raw[2];
      if (c < G) {
        const uint4* p = reinterpret_cast<const uint4*>(qp + 64 * j + 16 * g);
        raw[0] = p[0]; raw[1] = p[1];
      } else {
        raw[0] = raw[1] = make_uint4(0, 0, 0, 0);
      }
      const u4* e = reinterpret_cast<const u4*>(raw);
      for (int q = 0; q < 4; q++) qa[4 * j + q] = e[q];
    }
  }

  f4 acc[16];
  for (int nn = 0; nn < 16; nn++) acc[nn] = f4{0.f, 0.f, 0.f, 0.f};
  float m_i = -INFINITY, l_lane = 0.f;

  const int* bt = BT + (long)b * sbt;
  for (int pg = lo / PAGE; pg * PAGE < hi; pg++) {
    const long page = bt[pg];
    const uint16_t* kp = K + page * skb + (long)kvh * skh;
    const uint16_t* vp = V + page * svb + (long)kvh * svh;
    f4 S[4];
    for (int s = 0; s < 4; s++) {
      const int tok = 16 * (c & 3) + 4 * s + (c >> 2);
      const uint4* p = reinterpret_cast<const uint4*>(kp + (long)tok * skp + 16 * g);
      uint4 kr[4][2];
      for (int j = 0; j < 4; j++) { kr[j][0] = p[8 * j]; kr[j][1] = p[8 * j + 1]; }
      f4 a = f4{0.f, 0.f, 0.f, 0.f};
      for (int j = 0; j < 4; j++) {
        const u4* w = reinterpret_cast<const u4*>(kr[j]);
        for (int q = 0; q < 4; q++) a = __builtin_hcu_mmac_f32_16x16x16_bf16(qa[4 * j + q], w[q], a);
      }
      S[s] = a * qscale;
    }
    const int base = pg * PAGE + 16 * g;
    float mx = -INFINITY;
    for (int s = 0; s < 4; s++)
      for (int i = 0; i < 4; i++) {
        const int t = base + 4 * s + i;
        if (t < vlo || t >= hi) S[s][i] = -INFINITY;
        mx = fmaxf(mx, S[s][i]);
      }
    mx = fmaxf(mx, shfl_xor(mx, 16));
    mx = fmaxf(mx, shfl_xor(mx, 32));
    const float m_new = fmaxf(m_i, mx);
    const float m_safe = m_new == -INFINITY ? 0.f : m_new;
    const float alpha = exp2f(m_i - m_safe);
    m_i = m_new;
    u4 pa[4];
    float ls = 0.f;
    for (int s = 0; s < 4; s++) {
      float p[4];
      for (int i = 0; i < 4; i++) { p[i] = exp2f(S[s][i] - m_safe); ls += p[i]; }
      uint2 w = make_uint2(f2_to_bf2(p[0], p[1]), f2_to_bf2(p[2], p[3]));
      pa[s] = *reinterpret_cast<u4*>(&w);
    }
    l_lane = l_lane * alpha + ls;
    for (int h = 0; h < 2; h++) {
      uint4 vr[8][2];
      for (int n = 0; n < 8; n++) {
        const uint4* p = reinterpret_cast<const uint4*>(vp + (long)(c + 16 * (8 * h + n)) * svd + 16 * g);
        vr[n][0] = p[0]; vr[n][1] = p[1];
      }
      for (int n = 0; n < 8; n++) {
        const int nn = 8 * h + n;
        f4 a = acc[nn] * alpha;
        const u4* w = reinterpret_cast<const u4*>(vr[n]);
        for (int s = 0; s < 4; s++) a = __builtin_hcu_mmac_f32_16x16x16_bf16(pa[s], w[s], a);
        acc[nn] = a;
      }
    }
  }
  float l_tot = l_lane + shfl_xor(l_lane, 16);
  l_tot += shfl_xor(l_tot, 32);
  if (c >= G) return;
  const int head = kvh * G + c;
  const float inv = l_tot > 0.f ? 1.f / l_tot : 0.f;
  float* mo = MID_O + (long)b * smob + (long)head * smoh + (long)slot * smos;
  for (int nn = 0; nn < 16; nn++)
    for (int i = 0; i < 4; i++) mo[g + 4 * i + 16 * nn] = acc[nn][i] * inv;
  if (g == 0)
    MID_LSE[(long)b * smlb + (long)head * smlh + slot] = m_i * 0.6931471805599453f + logf(l_tot);
}

extern "C" int welm_dec_partial_launch(
    int kv8, int cp, int nbkv, int nsplit, const void* Q, const void* K, const void* V, const void* BT,
    const void* SEQK, void* MID_O, void* MID_LSE, float qscale, long sqb, long sqh, long skb,
    long skh, long skp, long svb, long svh, long svd, long sbt, long smob, long smoh, long smos,
    long smlb, long smlh, int KVH, int G, int window_left, hipStream_t st) {
  dim3 grid(nbkv, nsplit), block(64);
#define L(CPV)                                                                                     \
  if (kv8) hipLaunchKernelGGL(welm_dec_fp8_partial<CPV>, grid, block, 0, st, (const uint16_t*)Q,   \
                     (const uint8_t*)K, (const uint8_t*)V, (const int*)BT, (const int*)SEQK,       \
                     (float*)MID_O, (float*)MID_LSE, qscale, sqb, sqh, skb, skh, skp, svb, svh,    \
                     svd, sbt, smob, smoh, smos, smlb, smlh, KVH, G, window_left);                 \
  else hipLaunchKernelGGL(welm_dec_bf16_partial<CPV>, grid, block, 0, st, (const uint16_t*)Q,      \
                     (const uint16_t*)K, (const uint16_t*)V, (const int*)BT, (const int*)SEQK,       \
                     (float*)MID_O, (float*)MID_LSE, qscale, sqb, sqh, skb, skh, skp, svb, svh,    \
                     svd, sbt, smob, smoh, smos, smlb, smlh, KVH, G, window_left)
  switch (cp) {
    case 1: L(1); break;
    case 2: L(2); break;
    case 4: L(4); break;
    case 8: L(8); break;
    default: return -1;
  }
#undef L
  return (int)hipGetLastError();
}
