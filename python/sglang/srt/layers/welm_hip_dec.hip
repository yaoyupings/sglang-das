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
#ifdef NO_CVT
  uint2 r = make_uint2(x, x ^ 0x1234u); return *reinterpret_cast<h4*>(&r);
#endif
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
#ifdef NO_PV
    for (int nn = 0; nn < 1; nn++) {
#else
    for (int nn = 0; nn < 16; nn++) {
#endif
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


typedef char c8 __attribute__((ext_vector_type(8)));

__device__ __forceinline__ f4 mm8(c8 a, c8 b, f4 c) {
#ifdef NO_MM
  long x = *reinterpret_cast<long*>(&a) ^ *reinterpret_cast<long*>(&b);
  c[0] += (float)(int)x; return c;
#endif
  return __builtin_hcu_mmac_f32_16x16x32_fp8_fp8_lit_lts(a, b, c, 1, 0);
}

// e4m3 hi + residual lo of two floats packed into one 16-bit pair each
__device__ __forceinline__ void split2(float a, float b, uint32_t& hi, uint32_t& lo, int word) {
  const uint32_t h = __builtin_hcu_cvt_pk_fp8_f32(a, b, 0, false) & 0xffffu;
  const float ha = __builtin_hcu_cvt_f32_fp8((int)h, false, 0, 0), hb = __builtin_hcu_cvt_f32_fp8((int)h, false, 0, 1);
  const uint32_t r = __builtin_hcu_cvt_pk_fp8_f32(a - ha, b - hb, 0, false) & 0xffffu;
  hi |= h << (16 * word);
  lo |= r << (16 * word);
}

// fp8 KV on the fp8 mmac (16x16x32, probed with lit=1):
//   A[m = c][k = 8g + i], B[k = 8g + i][n = c], C[m = c][n = 4g + i] (i < 8 / < 4).
// q and P are e4m3 hi + lo (residual) pairs, both accumulated into one sum,
// ~7 mantissa bits like bf16. K/V bytes feed the mmac as stored.
//   QK k-step kk = 2j + h: d = 64j + 16g + 8h + i; subtile s: B column c is
//   token 16*(c/4) + 4s + c%4, so S holds tokens 16g + 4s + i.
//   PV k-step h (tokens 32h..): A k = 8g + i' is token 16g + 8h + i' (= S of
//   subtiles 2h, 2h+1); B column c of n-tile nn is d = c + 16nn; C holds
//   d = 4g + i + 16nn.
template <int CP>
__global__ __launch_bounds__(64) void welm_dec_fp8mm_partial(
    const uint16_t* __restrict__ Q, const uint8_t* __restrict__ K, const uint8_t* __restrict__ V,
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

  // q row c: 64 values (d = 64j + 16g + t), per-row scale to the e4m3 range
  c8 qh[8], ql[8];
  float qsc;
  {
    float qf[64];
    const uint16_t* qp = Q + (long)b * sqb + (long)(kvh * G + c) * sqh;
    float amax = 0.f;
    for (int j = 0; j < 4; j++) {
      uint4 raw[2];
      if (c < G) {
        const uint4* p = reinterpret_cast<const uint4*>(qp + 64 * j + 16 * g);
        raw[0] = p[0]; raw[1] = p[1];
      } else {
        raw[0] = raw[1] = make_uint4(0, 0, 0, 0);
      }
      const uint16_t* e = reinterpret_cast<const uint16_t*>(raw);
      for (int t = 0; t < 16; t++) { qf[16 * j + t] = bf16_to_f(e[t]); amax = fmaxf(amax, fabsf(qf[16 * j + t])); }
    }
    amax = fmaxf(amax, shfl_xor(amax, 16));
    amax = fmaxf(amax, shfl_xor(amax, 32));
    const float inv = amax > 0.f ? 448.f / amax : 0.f;
    qsc = (amax > 0.f ? amax / 448.f : 0.f) * qscale;
    for (int kk = 0; kk < 8; kk++) {
      uint32_t h0 = 0, h1 = 0, l0 = 0, l1 = 0;
      const float* x = qf + 8 * kk;  // kk = 2j + h -> offset 16j + 8h = 8kk
      split2(x[0] * inv, x[1] * inv, h0, l0, 0); split2(x[2] * inv, x[3] * inv, h0, l0, 1);
      split2(x[4] * inv, x[5] * inv, h1, l1, 0); split2(x[6] * inv, x[7] * inv, h1, l1, 1);
      uint2 hh = make_uint2(h0, h1), ll = make_uint2(l0, l1);
      qh[kk] = *reinterpret_cast<c8*>(&hh); ql[kk] = *reinterpret_cast<c8*>(&ll);
    }
  }

  f4 acc[16];
  for (int nn = 0; nn < 16; nn++) acc[nn] = f4{0.f, 0.f, 0.f, 0.f};
  float m_i = -INFINITY, l_lane = 0.f;

  const int* bt = BT + (long)b * sbt;
  for (int pg = lo / PAGE; pg * PAGE < hi; pg++) {
#ifdef SAME_PAGE
    const long page = 0;
#else
    const long page = bt[pg];
#endif
    const uint8_t* kp = K + page * skb + (long)kvh * skh;
    const uint8_t* vp = V + page * svb + (long)kvh * svh;
    uint4 kr[4][4];
    for (int s = 0; s < 4; s++) {
      const int tok = 16 * (c >> 2) + 4 * s + (c & 3);
      const uint4* p = reinterpret_cast<const uint4*>(kp + (long)tok * skp + 16 * g);
      for (int j = 0; j < 4; j++) kr[s][j] = p[4 * j];
    }
    f4 S[4];
    for (int s = 0; s < 4; s++) S[s] = f4{0.f, 0.f, 0.f, 0.f};
    for (int j = 0; j < 4; j++)
      for (int h = 0; h < 2; h++)
        for (int s = 0; s < 4; s++) {
          const c8 kb = reinterpret_cast<const c8*>(&kr[s][j])[h];
          S[s] = mm8(qh[2 * j + h], kb, S[s]);
          S[s] = mm8(ql[2 * j + h], kb, S[s]);
        }
    uint4 vr[16];
    for (int nn = 0; nn < 16; nn++)
      vr[nn] = *reinterpret_cast<const uint4*>(vp + (long)(c + 16 * nn) * svd + 16 * g);
    const int base = pg * PAGE + 16 * g;
    float mx = -INFINITY;
    for (int s = 0; s < 4; s++)
      for (int i = 0; i < 4; i++) {
        const int t = base + 4 * s + i;
        S[s][i] = (t < vlo || t >= hi) ? -INFINITY : S[s][i] * qsc;
        mx = fmaxf(mx, S[s][i]);
      }
    mx = fmaxf(mx, shfl_xor(mx, 16));
    mx = fmaxf(mx, shfl_xor(mx, 32));
    const float m_new = fmaxf(m_i, mx);
    const float m_safe = m_new == -INFINITY ? 0.f : m_new;
    const float alpha = exp2f(m_i - m_safe);
    m_i = m_new;
    c8 ph[2], pl[2];
    float ls = 0.f;
    for (int h = 0; h < 2; h++) {
      uint32_t h0 = 0, h1 = 0, l0 = 0, l1 = 0;
      float p[8];
      for (int u = 0; u < 2; u++)
        for (int i = 0; i < 4; i++) { p[4 * u + i] = exp2f(S[2 * h + u][i] - m_safe); ls += p[4 * u + i]; }
      // P * 256 keeps the e4m3 hi/lo pair in the normal range
      split2(p[0] * 256.f, p[1] * 256.f, h0, l0, 0); split2(p[2] * 256.f, p[3] * 256.f, h0, l0, 1);
      split2(p[4] * 256.f, p[5] * 256.f, h1, l1, 0); split2(p[6] * 256.f, p[7] * 256.f, h1, l1, 1);
      uint2 hh = make_uint2(h0, h1), ll = make_uint2(l0, l1);
      ph[h] = *reinterpret_cast<c8*>(&hh); pl[h] = *reinterpret_cast<c8*>(&ll);
    }
    l_lane = l_lane * alpha + ls;
    for (int nn = 0; nn < 16; nn++) {
      f4 a = acc[nn] * alpha;
      for (int h = 0; h < 2; h++) {
        const c8 vb = reinterpret_cast<const c8*>(&vr[nn])[h];
        a = mm8(ph[h], vb, a);
        a = mm8(pl[h], vb, a);
      }
      acc[nn] = a;
    }
  }
  float l_tot = l_lane + shfl_xor(l_lane, 16);
  l_tot += shfl_xor(l_tot, 32);
  if (c >= G) return;
  const int head = kvh * G + c;
  const float inv = l_tot > 0.f ? 1.f / (256.f * l_tot) : 0.f;
  float* mo = MID_O + (long)b * smob + (long)head * smoh + (long)slot * smos;
  for (int nn = 0; nn < 16; nn++)
    *reinterpret_cast<float4*>(mo + 4 * g + 16 * nn) =
        make_float4(acc[nn][0] * inv, acc[nn][1] * inv, acc[nn][2] * inv, acc[nn][3] * inv);
  if (g == 0)
    MID_LSE[(long)b * smlb + (long)head * smlh + slot] = m_i * 0.6931471805599453f + logf(l_tot);
}


// Cooperative variant: one 256-thread block (4 waves) per (batch, kv head, split).
// Wave w: QK for subtile s = w of every page (16 tokens, full D), PV for
// d in [64w, 64w + 64) (n-tiles nn = 4w..4w+3). Page max and P go through LDS,
// q hi/lo lives in LDS. Small register footprint -> several blocks per CU hide
// the HBM latency. Same layouts / permutations as welm_dec_fp8mm_partial.
template <int CP>
__global__ __launch_bounds__(256) void welm_dec_fp8co_partial(
    const uint16_t* __restrict__ Q, const uint8_t* __restrict__ K, const uint8_t* __restrict__ V,
    const int* __restrict__ BT, const int* __restrict__ SEQK, float* __restrict__ MID_O,
    float* __restrict__ MID_LSE, float qscale,
    long sqb, long sqh, long skb, long skh, long skp, long svb, long svh, long svd, long sbt,
    long smob, long smoh, long smos, long smlb, long smlh, int KVH, int G, int window_left) {
  __shared__ uint2 s_qh[8][64], s_ql[8][64];
  __shared__ uint32_t s_ph[2][4][64], s_pl[2][4][64];  // [parity][subtile][lane]
  __shared__ float s_mx[2][4][16];
  __shared__ float s_l[4][16];
  const int w = threadIdx.x >> 6, l = threadIdx.x & 63, c = l & 15, g = l >> 4;
  const int b = blockIdx.x / KVH, kvh = blockIdx.x % KVH, slot = blockIdx.y;
  const int kv_len = SEQK[b];
  const int w_lo = window_left >= 0 ? max(kv_len - 1 - window_left, 0) : 0;
  const int split = slot + (window_left >= 0 ? w_lo / (CP * PAGE) : 0);
  const int lo = split * CP * PAGE;
  if (lo >= kv_len) return;
  const int hi = min(lo + CP * PAGE, kv_len);
  const int vlo = max(lo, w_lo);
  const int* bt = BT + (long)b * sbt;
  const int pg0 = lo / PAGE, npg = (hi - lo + PAGE - 1) / PAGE;

  // q prologue: wave w handles k-steps kk = 2w, 2w + 1 (d = 64w + 16g + t), row amax via LDS
  float qsc;
  {
    const uint16_t* qp = Q + (long)b * sqb + (long)(kvh * G + c) * sqh;
    float qf[16];
    uint4 raw[2];
    if (c < G) {
      const uint4* p = reinterpret_cast<const uint4*>(qp + 64 * w + 16 * g);
      raw[0] = p[0]; raw[1] = p[1];
    } else {
      raw[0] = raw[1] = make_uint4(0, 0, 0, 0);
    }
    const uint16_t* e = reinterpret_cast<const uint16_t*>(raw);
    float amax = 0.f;
    for (int t = 0; t < 16; t++) { qf[t] = bf16_to_f(e[t]); amax = fmaxf(amax, fabsf(qf[t])); }
    amax = fmaxf(amax, shfl_xor(amax, 16));
    amax = fmaxf(amax, shfl_xor(amax, 32));
    if (g == 0) s_l[w][c] = amax;
    __syncthreads();
    amax = fmaxf(fmaxf(s_l[0][c], s_l[1][c]), fmaxf(s_l[2][c], s_l[3][c]));
    const float inv = amax > 0.f ? 448.f / amax : 0.f;
    qsc = (amax > 0.f ? amax / 448.f : 0.f) * qscale;
    for (int h = 0; h < 2; h++) {
      uint32_t h0 = 0, h1 = 0, l0 = 0, l1 = 0;
      const float* x = qf + 8 * h;
      split2(x[0] * inv, x[1] * inv, h0, l0, 0); split2(x[2] * inv, x[3] * inv, h0, l0, 1);
      split2(x[4] * inv, x[5] * inv, h1, l1, 0); split2(x[6] * inv, x[7] * inv, h1, l1, 1);
      s_qh[2 * w + h][l] = make_uint2(h0, h1); s_ql[2 * w + h][l] = make_uint2(l0, l1);
    }
  }
  __syncthreads();

  f4 acc[4];
  for (int n = 0; n < 4; n++) acc[n] = f4{0.f, 0.f, 0.f, 0.f};
  float m_i = -INFINITY, l_lane = 0.f;
  const int tok = 16 * (c >> 2) + 4 * w + (c & 3);

  long page = bt[pg0];
  for (int ip = 0; ip < npg; ip++) {
    const int pg = pg0 + ip, par = ip & 1;
    const uint8_t* kp = K + page * skb + (long)kvh * skh;
    const uint8_t* vp = V + page * svb + (long)kvh * svh;
    uint4 kr[4], vr[4];
    {
      const uint4* p = reinterpret_cast<const uint4*>(kp + (long)tok * skp + 16 * g);
      for (int j = 0; j < 4; j++) kr[j] = p[4 * j];
    }
    for (int n = 0; n < 4; n++)
      vr[n] = *reinterpret_cast<const uint4*>(vp + (long)(c + 16 * (4 * w + n)) * svd + 16 * g);
    if (ip + 1 < npg) page = bt[pg + 1];
    f4 S0 = f4{0.f, 0.f, 0.f, 0.f}, S1 = S0;
    for (int j = 0; j < 4; j++)
      for (int h = 0; h < 2; h++) {
        const c8 kb = reinterpret_cast<const c8*>(&kr[j])[h];
        const uint2 qh = s_qh[2 * j + h][l], ql = s_ql[2 * j + h][l];
        S0 = mm8(*reinterpret_cast<const c8*>(&qh), kb, S0);
        S1 = mm8(*reinterpret_cast<const c8*>(&ql), kb, S1);
      }
    f4 S = S0 + S1;
    const int base = pg * PAGE + 16 * g + 4 * w;
    float mx = -INFINITY;
    for (int i = 0; i < 4; i++) {
      const int t = base + i;
      S[i] = (t < vlo || t >= hi) ? -INFINITY : S[i] * qsc;
      mx = fmaxf(mx, S[i]);
    }
    mx = fmaxf(mx, shfl_xor(mx, 16));
    mx = fmaxf(mx, shfl_xor(mx, 32));
    if (g == 0) s_mx[par][w][c] = mx;
    __syncthreads();
    mx = fmaxf(fmaxf(s_mx[par][0][c], s_mx[par][1][c]), fmaxf(s_mx[par][2][c], s_mx[par][3][c]));
    const float m_new = fmaxf(m_i, mx);
    const float m_safe = m_new == -INFINITY ? 0.f : m_new;
    const float alpha = exp2f(m_i - m_safe);
    m_i = m_new;
    float p[4], ls = 0.f;
    for (int i = 0; i < 4; i++) { p[i] = exp2f(S[i] - m_safe); ls += p[i]; }
    l_lane = l_lane * alpha + ls;
    {
      uint32_t h0 = 0, l0 = 0;
      split2(p[0] * 256.f, p[1] * 256.f, h0, l0, 0); split2(p[2] * 256.f, p[3] * 256.f, h0, l0, 1);
      s_ph[par][w][l] = h0; s_pl[par][w][l] = l0;
    }
    __syncthreads();
    c8 ph[2], pl[2];
    for (int h = 0; h < 2; h++) {
      uint2 a = make_uint2(s_ph[par][2 * h][l], s_ph[par][2 * h + 1][l]);
      uint2 bq = make_uint2(s_pl[par][2 * h][l], s_pl[par][2 * h + 1][l]);
      ph[h] = *reinterpret_cast<c8*>(&a); pl[h] = *reinterpret_cast<c8*>(&bq);
    }
    for (int n = 0; n < 4; n++) {
      f4 a = acc[n] * alpha;
      for (int h = 0; h < 2; h++) {
        const c8 vb = reinterpret_cast<const c8*>(&vr[n])[h];
        a = mm8(ph[h], vb, a);
        a = mm8(pl[h], vb, a);
      }
      acc[n] = a;
    }
  }
  float lw = l_lane + shfl_xor(l_lane, 16);
  lw += shfl_xor(lw, 32);
  if (g == 0) s_l[w][c] = lw;  // safe: s_l last read before the first page barrier
  __syncthreads();
  const float l_tot = s_l[0][c] + s_l[1][c] + s_l[2][c] + s_l[3][c];
  if (c >= G) return;
  const int head = kvh * G + c;
  const float inv = l_tot > 0.f ? 1.f / (256.f * l_tot) : 0.f;
  float* mo = MID_O + (long)b * smob + (long)head * smoh + (long)slot * smos;
  for (int n = 0; n < 4; n++)
    *reinterpret_cast<float4*>(mo + 4 * g + 16 * (4 * w + n)) =
        make_float4(acc[n][0] * inv, acc[n][1] * inv, acc[n][2] * inv, acc[n][3] * inv);
  if (w == 0 && g == 0)
    MID_LSE[(long)b * smlb + (long)head * smlh + slot] = m_i * 0.6931471805599453f + logf(l_tot);
}


// Split reduce: one wave per (batch, head), lane owns d = 4l..4l+3. Same math
// as welmv4_fast_op._welm_dec_attn_reduce_kernel (sink as an extra logit).
__global__ __launch_bounds__(64) void welm_dec_reduce(
    const float* __restrict__ MID_O, const float* __restrict__ MID_LSE, const int* __restrict__ SEQK,
    const void* __restrict__ SINKS, int sink_kind,  // 0 none, 1 bf16, 2 f32
    uint16_t* __restrict__ OUT, long smob, long smoh, long smos, long smlb, long smlh, long sob, long soh,
    int chunk, int window_left, float v_scale, const uint16_t* __restrict__ GATE, long sg) {
  const int b = blockIdx.x, h = blockIdx.y, l = threadIdx.x;
  const int kv_len = SEQK[b];
  int n = (kv_len + chunk - 1) / chunk;
  if (window_left >= 0) n -= max(kv_len - 1 - window_left, 0) / chunk;
  const float* lse = MID_LSE + (long)b * smlb + (long)h * smlh;
  const float* mo = MID_O + (long)b * smob + (long)h * smoh + 4 * l;
  float sink = -INFINITY;
  if (sink_kind == 1) sink = bf16_to_f(reinterpret_cast<const uint16_t*>(SINKS)[h]);
  else if (sink_kind == 2) sink = reinterpret_cast<const float*>(SINKS)[h];
  float M = sink;
  for (int s = l; s < n; s += 64) M = fmaxf(M, lse[s]);
  for (int o = 32; o > 0; o >>= 1) M = fmaxf(M, __shfl_xor(M, o, 64));
  float4 acc = make_float4(0.f, 0.f, 0.f, 0.f);
  float den = sink_kind ? __expf(sink - M) : 0.f;
  int s = 0;
  for (; s + 4 <= n; s += 4) {
    float w[4]; float4 o[4];
#pragma unroll
    for (int u = 0; u < 4; u++) { w[u] = lse[s + u]; o[u] = *reinterpret_cast<const float4*>(mo + (long)(s + u) * smos); }
#pragma unroll
    for (int u = 0; u < 4; u++) {
      const float e = __expf(w[u] - M);
      acc.x += e * o[u].x; acc.y += e * o[u].y; acc.z += e * o[u].z; acc.w += e * o[u].w; den += e;
    }
  }
  for (; s < n; s++) {
    const float w0 = __expf(lse[s] - M);
    const float4 o0 = *reinterpret_cast<const float4*>(mo + (long)s * smos);
    acc.x += w0 * o0.x; acc.y += w0 * o0.y; acc.z += w0 * o0.z; acc.w += w0 * o0.w;
    den += w0;
  }
  const float r = (den > 0.f ? 1.f / den : 1.f) * v_scale;
  uint32_t lo = f2_to_bf2(acc.x * r, acc.y * r), hi = f2_to_bf2(acc.z * r, acc.w * r);
  if (GATE) {
    // y = bf16(bf16(sigmoid(g)) * bf16(attn)), as sigmoid_mul_strided_
    const float sg_ = bf16_to_f(f2_to_bf2(1.f / (1.f + expf(-bf16_to_f(GATE[(long)b * sg + h]))), 0.f) & 0xffffu);
    lo = f2_to_bf2(sg_ * bf16_to_f(lo & 0xffffu), sg_ * bf16_to_f(lo >> 16));
    hi = f2_to_bf2(sg_ * bf16_to_f(hi & 0xffffu), sg_ * bf16_to_f(hi >> 16));
  }
  *reinterpret_cast<uint2*>(OUT + (long)b * sob + (long)h * soh + 4 * l) = make_uint2(lo, hi);
}

extern "C" int welm_dec_reduce_launch(int B, int H, const void* MID_O, const void* MID_LSE, const void* SEQK,
                                      const void* SINKS, int sink_kind, void* OUT, long smob, long smoh,
                                      long smos, long smlb, long smlh, long sob, long soh, int chunk,
                                      int window_left, float v_scale, const void* GATE, long sg,
                                      hipStream_t st) {
  hipLaunchKernelGGL(welm_dec_reduce, dim3(B, H), dim3(64), 0, st, (const float*)MID_O, (const float*)MID_LSE,
                     (const int*)SEQK, SINKS, sink_kind, (uint16_t*)OUT, smob, smoh, smos, smlb, smlh, sob,
                     soh, chunk, window_left, v_scale, (const uint16_t*)GATE, sg);
  return (int)hipGetLastError();
}

extern "C" int welm_dec_partial_launch(
    int kv8, int cp, int nbkv, int nsplit, const void* Q, const void* K, const void* V, const void* BT,
    const void* SEQK, void* MID_O, void* MID_LSE, float qscale, long sqb, long sqh, long skb,
    long skh, long skp, long svb, long svh, long svd, long sbt, long smob, long smoh, long smos,
    long smlb, long smlh, int KVH, int G, int window_left, hipStream_t st) {
  dim3 grid(nbkv, nsplit), block(kv8 == 3 ? 256 : 64);
#define L(CPV)                                                                                     \
  if (kv8 == 3) hipLaunchKernelGGL(welm_dec_fp8co_partial<CPV>, grid, block, 0, st, (const uint16_t*)Q,   \
                     (const uint8_t*)K, (const uint8_t*)V, (const int*)BT, (const int*)SEQK,       \
                     (float*)MID_O, (float*)MID_LSE, qscale, sqb, sqh, skb, skh, skp, svb, svh,    \
                     svd, sbt, smob, smoh, smos, smlb, smlh, KVH, G, window_left);                 \
  else if (kv8 == 2) hipLaunchKernelGGL(welm_dec_fp8mm_partial<CPV>, grid, block, 0, st, (const uint16_t*)Q,   \
                     (const uint8_t*)K, (const uint8_t*)V, (const int*)BT, (const int*)SEQK,       \
                     (float*)MID_O, (float*)MID_LSE, qscale, sqb, sqh, skb, skh, skp, svb, svh,    \
                     svd, sbt, smob, smoh, smos, smlb, smlh, KVH, G, window_left);                 \
  else if (kv8) hipLaunchKernelGGL(welm_dec_fp8_partial<CPV>, grid, block, 0, st, (const uint16_t*)Q,   \
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

// ---------------------------------------------------------------------------
// Prefix attention (short new queries over a long paged prefix), fp8 KV.
// Rows r = tok * G + head (as _welm_pfx_attn_partial_kernel); each wave owns
// 16 rows and runs the welm_dec_fp8mm_partial page loop with a per-row causal
// limit. Split s covers [s * split_len, ...), split_len = ceil(ceil(L/NS)/64)*64,
// so the Triton _welm_pfx_attn_reduce_kernel merges the partials.
// Grid (B * KVH, ceil(max_q * G / 64), NS), block 256 (4 independent waves).
// ---------------------------------------------------------------------------
__global__ __launch_bounds__(256) void welm_pfx_fp8mm_partial(
    const uint16_t* __restrict__ Q, const uint8_t* __restrict__ K, const uint8_t* __restrict__ V,
    const int* __restrict__ BT, const int* __restrict__ SEQK, const int* __restrict__ CUQ,
    float* __restrict__ MID_O, float* __restrict__ MID_LSE, float qscale,
    long sqt, long sqh, long skb, long skh, long skp, long svb, long svh, long svd, long sbt,
    long smot, long smoh, long smos, long smlt, long smlh, int KVH, int G, int NS) {
  const int w = threadIdx.x >> 6, l = threadIdx.x & 63, c = l & 15, g = l >> 4;
  const int b = blockIdx.x / KVH, kvh = blockIdx.x % KVH, split = blockIdx.z;
  const int q_start = CUQ[b], n_q = CUQ[b + 1] - q_start;
  const int rows = n_q * G;
  const int r0 = blockIdx.y * 64 + w * 16;
  if (r0 >= rows) return;
  const int kv_len = SEQK[b], prefix = kv_len - n_q;
  const int split_len = ((kv_len + NS - 1) / NS + PAGE - 1) / PAGE * PAGE;
  const int lo = split * split_len;
  const int hi = min(lo + split_len, kv_len);
  const int r = r0 + c;
  const bool rv = r < rows;
  const int tok = rv ? r / G : 0, head = kvh * G + (rv ? r % G : 0);
  const int q_pos = prefix + tok;
  // rows r0..r0+15 see keys < min(hi, max q_pos + 1)
  const int r_last = min(r0 + 15, rows - 1);
  const int hi_w = min(hi, prefix + r_last / G + 1);

  c8 qh[8], ql[8];
  float qsc;
  {
    float qf[64];
    const uint16_t* qp = Q + (long)(q_start + tok) * sqt + (long)head * sqh;
    float amax = 0.f;
    for (int j = 0; j < 4; j++) {
      uint4 raw[2];
      if (rv) {
        const uint4* p = reinterpret_cast<const uint4*>(qp + 64 * j + 16 * g);
        raw[0] = p[0]; raw[1] = p[1];
      } else {
        raw[0] = raw[1] = make_uint4(0, 0, 0, 0);
      }
      const uint16_t* e = reinterpret_cast<const uint16_t*>(raw);
      for (int t = 0; t < 16; t++) { qf[16 * j + t] = bf16_to_f(e[t]); amax = fmaxf(amax, fabsf(qf[16 * j + t])); }
    }
    amax = fmaxf(amax, shfl_xor(amax, 16));
    amax = fmaxf(amax, shfl_xor(amax, 32));
    const float inv = amax > 0.f ? 448.f / amax : 0.f;
    qsc = (amax > 0.f ? amax / 448.f : 0.f) * qscale;
    for (int kk = 0; kk < 8; kk++) {
      uint32_t h0 = 0, h1 = 0, l0 = 0, l1 = 0;
      const float* x = qf + 8 * kk;
      split2(x[0] * inv, x[1] * inv, h0, l0, 0); split2(x[2] * inv, x[3] * inv, h0, l0, 1);
      split2(x[4] * inv, x[5] * inv, h1, l1, 0); split2(x[6] * inv, x[7] * inv, h1, l1, 1);
      uint2 hh = make_uint2(h0, h1), ll = make_uint2(l0, l1);
      qh[kk] = *reinterpret_cast<c8*>(&hh); ql[kk] = *reinterpret_cast<c8*>(&ll);
    }
  }

  f4 acc[16];
  for (int nn = 0; nn < 16; nn++) acc[nn] = f4{0.f, 0.f, 0.f, 0.f};
  float m_i = -INFINITY, l_lane = 0.f;
  const int lim = min(hi, q_pos + 1);  // this row's keys: [lo, lim)

  const int* bt = BT + (long)b * sbt;
  for (int pg = lo / PAGE; pg * PAGE < hi_w; pg++) {
    const long page = bt[pg];
    const uint8_t* kp = K + page * skb + (long)kvh * skh;
    const uint8_t* vp = V + page * svb + (long)kvh * svh;
    uint4 kr[4][4];
    for (int s = 0; s < 4; s++) {
      const int t = 16 * (c >> 2) + 4 * s + (c & 3);
      const uint4* p = reinterpret_cast<const uint4*>(kp + (long)t * skp + 16 * g);
      for (int j = 0; j < 4; j++) kr[s][j] = p[4 * j];
    }
    f4 S[4];
    for (int s = 0; s < 4; s++) S[s] = f4{0.f, 0.f, 0.f, 0.f};
    for (int j = 0; j < 4; j++)
      for (int h = 0; h < 2; h++)
        for (int s = 0; s < 4; s++) {
          const c8 kb = reinterpret_cast<const c8*>(&kr[s][j])[h];
          S[s] = mm8(qh[2 * j + h], kb, S[s]);
          S[s] = mm8(ql[2 * j + h], kb, S[s]);
        }
    uint4 vr[16];
    for (int nn = 0; nn < 16; nn++)
      vr[nn] = *reinterpret_cast<const uint4*>(vp + (long)(c + 16 * nn) * svd + 16 * g);
    const int base = pg * PAGE + 16 * g;
    float mx = -INFINITY;
    for (int s = 0; s < 4; s++)
      for (int i = 0; i < 4; i++) {
        const int t = base + 4 * s + i;
        S[s][i] = (t >= lim) ? -INFINITY : S[s][i] * qsc;
        mx = fmaxf(mx, S[s][i]);
      }
    mx = fmaxf(mx, shfl_xor(mx, 16));
    mx = fmaxf(mx, shfl_xor(mx, 32));
    const float m_new = fmaxf(m_i, mx);
    const float m_safe = m_new == -INFINITY ? 0.f : m_new;
    const float alpha = exp2f(m_i - m_safe);
    m_i = m_new;
    c8 ph[2], pl[2];
    float ls = 0.f;
    for (int h = 0; h < 2; h++) {
      uint32_t h0 = 0, h1 = 0, l0 = 0, l1 = 0;
      float p[8];
      for (int u = 0; u < 2; u++)
        for (int i = 0; i < 4; i++) { p[4 * u + i] = exp2f(S[2 * h + u][i] - m_safe); ls += p[4 * u + i]; }
      split2(p[0] * 256.f, p[1] * 256.f, h0, l0, 0); split2(p[2] * 256.f, p[3] * 256.f, h0, l0, 1);
      split2(p[4] * 256.f, p[5] * 256.f, h1, l1, 0); split2(p[6] * 256.f, p[7] * 256.f, h1, l1, 1);
      uint2 hh = make_uint2(h0, h1), ll = make_uint2(l0, l1);
      ph[h] = *reinterpret_cast<c8*>(&hh); pl[h] = *reinterpret_cast<c8*>(&ll);
    }
    l_lane = l_lane * alpha + ls;
    for (int nn = 0; nn < 16; nn++) {
      f4 a = acc[nn] * alpha;
      for (int h = 0; h < 2; h++) {
        const c8 vb = reinterpret_cast<const c8*>(&vr[nn])[h];
        a = mm8(ph[h], vb, a);
        a = mm8(pl[h], vb, a);
      }
      acc[nn] = a;
    }
  }
  float l_tot = l_lane + shfl_xor(l_lane, 16);
  l_tot += shfl_xor(l_tot, 32);
  if (!rv) return;
  const float inv = l_tot > 0.f ? 1.f / (256.f * l_tot) : 0.f;
  float* mo = MID_O + (long)(q_start + tok) * smot + (long)head * smoh + (long)split * smos;
  for (int nn = 0; nn < 16; nn++)
    *reinterpret_cast<float4*>(mo + 4 * g + 16 * nn) =
        make_float4(acc[nn][0] * inv, acc[nn][1] * inv, acc[nn][2] * inv, acc[nn][3] * inv);
  if (g == 0)
    MID_LSE[(long)(q_start + tok) * smlt + (long)head * smlh + split] =
        l_tot > 0.f ? m_i * 0.6931471805599453f + logf(l_tot) : -INFINITY;
}

extern "C" int welm_pfx_partial_launch(
    int bkv, int row_tiles, int ns, const void* Q, const void* K, const void* V, const void* BT,
    const void* SEQK, const void* CUQ, void* MID_O, void* MID_LSE, float qscale, long sqt, long sqh,
    long skb, long skh, long skp, long svb, long svh, long svd, long sbt, long smot, long smoh,
    long smos, long smlt, long smlh, int KVH, int G, hipStream_t st) {
  hipLaunchKernelGGL(welm_pfx_fp8mm_partial, dim3(bkv, row_tiles, ns), dim3(256), 0, st, (const uint16_t*)Q,
                     (const uint8_t*)K, (const uint8_t*)V, (const int*)BT, (const int*)SEQK,
                     (const int*)CUQ, (float*)MID_O, (float*)MID_LSE, qscale, sqt, sqh, skb, skh, skp,
                     svb, svh, svd, sbt, smot, smoh, smos, smlt, smlh, KVH, G, ns);
  return (int)hipGetLastError();
}

// ---------------------------------------------------------------------------
// Router biased top-k (drop-in for welmv4_fast_op._welm_router_topk_kernel,
// WRITE_SCORES=False): one wave per row, N = 64 * NPL logits, same key
// (ordered routing bits << 32 | (N-1-tie_rank)*N + idx) and selection order.
// ---------------------------------------------------------------------------
__device__ __forceinline__ long long shfl_xor_ll(long long v, int m) {
  int lo = __shfl_xor((int)(v & 0xffffffffll), m, 64), hi = __shfl_xor((int)(v >> 32), m, 64);
  return ((long long)hi << 32) | (unsigned int)lo;
}

template <int NPL>
__global__ __launch_bounds__(256) void welm_router_topk(const float* __restrict__ part, const float* __restrict__ bias,
                                                       float* __restrict__ scores, float* __restrict__ tw,
                                                       long long* __restrict__ tid, int M, int split_k, int topk,
                                                       int shared_id) {
  constexpr int N = 64 * NPL;
  const int l = threadIdx.x & 63;
  const int row = blockIdx.x * 4 + (threadIdx.x >> 6);
  if (row >= M) return;
  const int out_k = topk + (shared_id >= 0 ? 1 : 0);
  long long key[NPL];
  float lg[NPL];
#pragma unroll
  for (int j = 0; j < NPL; j++) lg[j] = 0.f;
  for (int s = 0; s < split_k; s++) {
    const float* p = part + ((long)s * M + row) * N;
#pragma unroll
    for (int j = 0; j < NPL; j++) lg[j] += p[l + 64 * j];
  }
#pragma unroll
  for (int j = 0; j < NPL; j++) {
    const int offs = l + 64 * j;
    float sc = 1.f / (1.f + __expf(-lg[j]));
    if (scores) scores[(long)row * N + offs] = sc;
    if (sc != sc) sc = -INFINITY;
    float routing = sc + bias[offs];
    routing = routing + 0.f;
    const int lane_idx = (offs % 128) / 4, local_idx = (offs / 128) * 4 + (offs % 4);
    const int tie_rank = lane_idx * (N / 32) + local_idx;
    const int bits = __float_as_int(routing);
    const long long ordered = (long long)(bits >= 0 ? bits : bits ^ 0x7fffffff);
    key[j] = (ordered << 32) | (long long)((N - 1 - tie_rank) * N + offs);
  }
  const long long NEG = (long long)0x8000000000000000ull;
  float* wrow = tw + (long)row * out_k;
  long long* irow = tid + (long)row * out_k;
  for (int k = 0; k < topk; k++) {
    long long b = key[0];
#pragma unroll
    for (int j = 1; j < NPL; j++) b = key[j] > b ? key[j] : b;
#pragma unroll
    for (int o = 32; o > 0; o >>= 1) { const long long t = shfl_xor_ll(b, o); b = t > b ? t : b; }
    const int idx = (int)(b & (N - 1));
    if ((idx & 63) == l) {
      const int j = idx >> 6;
#pragma unroll
      for (int jj = 0; jj < NPL; jj++)
        if (jj == j) {
          key[jj] = NEG;
          float w = 1.f / (1.f + __expf(-lg[jj]));
          if (w != w) w = -INFINITY;
          wrow[k] = w;
          irow[k] = idx;
        }
    }
  }
  if (shared_id >= 0 && l == 0) { wrow[topk] = 1.f; irow[topk] = shared_id; }
}

extern "C" int welm_router_topk_launch(const void* part, const void* bias, void* scores, void* tw, void* tid, int M, int N,
                                       int split_k, int topk, int shared_id, hipStream_t st) {
  const dim3 grid((M + 3) / 4), block(256);
  if (N == 512)
    hipLaunchKernelGGL(welm_router_topk<8>, grid, block, 0, st, (const float*)part, (const float*)bias, (float*)scores, (float*)tw,
                       (long long*)tid, M, split_k, topk, shared_id);
  else if (N == 256)
    hipLaunchKernelGGL(welm_router_topk<4>, grid, block, 0, st, (const float*)part, (const float*)bias, (float*)scores, (float*)tw,
                       (long long*)tid, M, split_k, topk, shared_id);
  else
    return -1;
  return (int)hipGetLastError();
}
