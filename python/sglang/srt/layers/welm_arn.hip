// WeLM fused 2-rank all-reduce + RMSNorm(s) for TP2 decode (gfx938).
// Inputs: the two ranks' bf16 partial sums in IPC-mapped uncached buffers (the aiter AR input pool),
// summed in rank order (bitwise identical on both ranks), then per row:
//   mode 0 (post-attn, mmq_style_norm_after_attn [+ fp8 per-token quant]):
//     s = bf16(in0 + in1); o = bf16(rms(s) * g0); h = f32(o) + res;  res_out = h
//     r = rms(bf16(h)) * g1;  fp32_out = r; out = bf16(r); [xq, xs = per-token fp8 quant of out]
//   mode 1 (ppln input_layernorm, residual_after_layernorm + clone_fp32_out):
//     s = bf16(in0 + in1); h = f32(s) + res; r = rms(bf16(h)) * g0; fp32_out = r; out = bf16(r)
// Barrier: aiter custom_all_reduce start_sync / end_sync<final> protocol on our own signal buffer.
#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>
#include <stdint.h>

#define MAXB 128
#ifdef TS
__device__ unsigned long long g_ts[8];
#define T(i) if (blockIdx.x == 0 && threadIdx.x == 0) g_ts[i] = wall_clock64();
#else
#define T(i)
#endif
#define NT 256
#ifndef PAD
#define PAD 32
#endif
struct Signal {
    alignas(128) uint32_t start[MAXB][PAD];
    alignas(128) uint32_t end[MAXB][PAD];
    alignas(128) uint32_t flag[MAXB][PAD];
};

__device__ __forceinline__ void start_sync(Signal* s0, Signal* s1, Signal* self, int rank) {
    uint32_t flag = self->flag[blockIdx.x][0] + 1;
    if (threadIdx.x < 2) {
        Signal* dst = threadIdx.x == 0 ? s0 : s1;
        __scoped_atomic_store_n(&dst->start[blockIdx.x][rank], flag, __ATOMIC_RELAXED, __MEMORY_SCOPE_SYSTEM);
        while (__scoped_atomic_load_n(&self->start[blockIdx.x][threadIdx.x], __ATOMIC_RELAXED, __MEMORY_SCOPE_DEVICE) < flag)
            ;
    }
    __syncthreads();
    if (threadIdx.x == 0) self->flag[blockIdx.x][0] = flag;
}

__device__ __forceinline__ void end_sync(Signal* s0, Signal* s1, Signal* self, int rank) {
    __syncthreads();
    uint32_t flag = self->flag[blockIdx.x][0] + 1;
    if (threadIdx.x < 2) {
        Signal* dst = threadIdx.x == 0 ? s0 : s1;
        __scoped_atomic_store_n(&dst->end[blockIdx.x][rank], flag, __ATOMIC_RELAXED, __MEMORY_SCOPE_SYSTEM);
        while (__scoped_atomic_load_n(&self->end[blockIdx.x][threadIdx.x], __ATOMIC_RELAXED, __MEMORY_SCOPE_DEVICE) < flag)
            ;
    }
    __syncthreads();
    if (threadIdx.x == 0) self->flag[blockIdx.x][0] = flag;
}

__device__ __forceinline__ float bfw(uint32_t w, int hi) { return __uint_as_float(hi ? (w & 0xffff0000u) : (w << 16)); }
__device__ __forceinline__ float bfu(const uint4& v, int j) {
    uint32_t w = j < 2 ? v.x : j < 4 ? v.y : j < 6 ? v.z : v.w;
    return bfw(w, j & 1);
}
__device__ __forceinline__ uint32_t bfbits(float x) { return (uint32_t)__bfloat16_as_ushort(__float2bfloat16(x)); }
__device__ __forceinline__ float bf(float x) { return __bfloat162float(__float2bfloat16(x)); }

__device__ __forceinline__ float block_sum(float v, float* sh) {
#pragma unroll
    for (int o = 32; o > 0; o >>= 1) v += __shfl_xor(v, o, 64);
    int w = threadIdx.x >> 6, l = threadIdx.x & 63;
    __syncthreads();
    if (l == 0) sh[w] = v;
    __syncthreads();
    float t = 0.f;
#pragma unroll
    for (int i = 0; i < NT / 64; i++) t += sh[i];
    return t;
}

__device__ __forceinline__ float2 block_sum_max(float v, float m, float2* sh) {
#pragma unroll
    for (int o = 32; o > 0; o >>= 1) { v += __shfl_xor(v, o, 64); m = fmaxf(m, __shfl_xor(m, o, 64)); }
    int w = threadIdx.x >> 6, l = threadIdx.x & 63;
    __syncthreads();
    if (l == 0) sh[w] = make_float2(v, m);
    __syncthreads();
    float t = 0.f, u = 0.f;
#pragma unroll
    for (int i = 0; i < NT / 64; i++) { float2 a = sh[i]; t += a.x; u = fmaxf(u, a.y); }
    return make_float2(t, u);
}

__device__ __forceinline__ float block_max(float v, float* sh) {
#pragma unroll
    for (int o = 32; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor(v, o, 64));
    int w = threadIdx.x >> 6, l = threadIdx.x & 63;
    __syncthreads();
    if (l == 0) sh[w] = v;
    __syncthreads();
    float t = sh[0];
#pragma unroll
    for (int i = 1; i < NT / 64; i++) t = fmaxf(t, sh[i]);
    return t;
}

// float (|x| <= 448) -> OCP e4m3fn, round-to-nearest-even.
__device__ __forceinline__ uint8_t to_e4m3(float x) {
    uint8_t sgn = x < 0.f ? 0x80 : 0;
    float a = fabsf(x);
    uint32_t code;
    if (a < 0.015625f) {                       // subnormal (2^-6)
        code = (uint32_t)rintf(a * 512.f);    // units of 2^-9; 8 -> 0x08 = min normal
    } else {
        int e;
        float m = frexpf(a, &e);               // a = m * 2^e, m in [0.5, 1)
        e -= 1;                                // a = (2m) * 2^e, 2m in [1, 2)
        uint32_t q = (uint32_t)rintf((2.f * m - 1.f) * 8.f);
        if (q == 8) { q = 0; e += 1; }
        code = ((uint32_t)(e + 7) << 3) | q;
    }
    return sgn | (uint8_t)code;
}

template <int MODE, bool QUANT, int VPT, bool END>
__global__ void __launch_bounds__(NT, 1) arn_kernel(
    const __hip_bfloat16* __restrict__ in0, const __hip_bfloat16* __restrict__ in1,
    Signal* s0, Signal* s1, Signal* self, int rank, int rows, int cols,
    const __hip_bfloat16* __restrict__ g0, const __hip_bfloat16* __restrict__ g1,
    const float* __restrict__ res, __hip_bfloat16* __restrict__ out, float* __restrict__ res_out,
    float* __restrict__ f32_out, uint8_t* __restrict__ xq, float* __restrict__ xs, float eps) {
    static_assert(VPT == 8, "");
    __shared__ float sh[NT / 64];
    __shared__ float2 sh2[NT / 64];
    const int c0 = threadIdx.x * VPT;
    const bool act = c0 < cols;
    // Everything that does not depend on the peer is loaded before the barrier: gammas, the residual
    // and this rank's own partial sum of the first row (written by the previous kernel in-stream).
    const __hip_bfloat16* own = rank == 0 ? in0 : in1;
    const __hip_bfloat16* peer = rank == 0 ? in1 : in0;
    uint4 gw0 = make_uint4(0, 0, 0, 0), gw1 = make_uint4(0, 0, 0, 0);
    uint4 va = make_uint4(0, 0, 0, 0);
    float4 r0 = make_float4(0, 0, 0, 0), r1 = r0;
    int row = blockIdx.x;
#ifndef PF
#define PF 1
#endif
    if (PF && act) {
        gw0 = *(const uint4*)(g0 + c0);
        if (MODE == 0) gw1 = *(const uint4*)(g1 + c0);
        if (row < rows) {
            const long off = (long)row * cols + c0;
            va = *(const uint4*)(own + off);
            r0 = *(const float4*)(res + off); r1 = *(const float4*)(res + off + 4);
        }
    }
    T(0)
    start_sync(s0, s1, self, rank);
    T(1)
    if (!PF && act) {
        gw0 = *(const uint4*)(g0 + c0);
        if (MODE == 0) gw1 = *(const uint4*)(g1 + c0);
    }
    float w0[VPT], w1[VPT];
#pragma unroll
    for (int j = 0; j < VPT; j++) {
        w0[j] = bfu(gw0, j);
        w1[j] = bfu(gw1, j);
    }
    const float inv_cols = 1.f / (float)cols;
    for (; row < rows; row += gridDim.x) {
        const long off = (long)row * cols + c0;
        if ((!PF || row != blockIdx.x) && act) {
            va = *(const uint4*)(own + off);
            r0 = *(const float4*)(res + off); r1 = *(const float4*)(res + off + 4);
        }
        float s[VPT];
        if (act) {
            uint4 vb = *(const uint4*)(peer + off);
#ifdef TS
            asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
#endif
            T(2)
            // rank-order sum (in0 + in1) so both ranks are bitwise identical
#pragma unroll
            for (int j = 0; j < VPT; j++) {
                float x = bfu(va, j), y = bfu(vb, j);
                s[j] = rank == 0 ? bf(x + y) : bf(y + x);
            }
        } else {
#pragma unroll
            for (int j = 0; j < VPT; j++) s[j] = 0.f;
        }
        float h[VPT];
        if (MODE == 0) {
            float ss = 0.f;
#pragma unroll
            for (int j = 0; j < VPT; j++) ss += s[j] * s[j];
            float inv = rsqrtf(block_sum(ss, sh) * inv_cols + eps);
#pragma unroll
            for (int j = 0; j < VPT; j++) h[j] = bf(s[j] * inv * w0[j]);
        } else {
#pragma unroll
            for (int j = 0; j < VPT; j++) h[j] = s[j];
        }
        if (act) {
            h[0] += r0.x; h[1] += r0.y; h[2] += r0.z; h[3] += r0.w;
            h[4] += r1.x; h[5] += r1.y; h[6] += r1.z; h[7] += r1.w;
            if (MODE == 0) {
                *(float4*)(res_out + off) = make_float4(h[0], h[1], h[2], h[3]);
                *(float4*)(res_out + off + 4) = make_float4(h[4], h[5], h[6], h[7]);
            }
        }
        float ss2 = 0.f;
        const float* wr = MODE == 0 ? w1 : w0;
#pragma unroll
        for (int j = 0; j < VPT; j++) { h[j] = bf(h[j]); ss2 += h[j] * h[j]; }
#ifndef NOFUSEMAX
        float hwm = 0.f;
        if (QUANT) {
#pragma unroll
            for (int j = 0; j < VPT; j++) hwm = fmaxf(hwm, fabsf(h[j] * wr[j]));
        }
        float inv2;
        float sxf = 0.f;
        if (QUANT) {
            float2 sm = block_sum_max(ss2, hwm, sh2);
            inv2 = rsqrtf(sm.x * inv_cols + eps);
            sxf = fmaxf(bf(sm.y * inv2), 1e-10f) / 448.f;
        } else {
            inv2 = rsqrtf(block_sum(ss2, sh) * inv_cols + eps);
        }
#else
        float inv2 = rsqrtf(block_sum(ss2, sh) * inv_cols + eps);
#endif
        float r[VPT];
        uint32_t ow[VPT / 2];
        float ob[VPT];
        float amax = 0.f;
#pragma unroll
        for (int j = 0; j < VPT; j++) {
            r[j] = h[j] * inv2 * wr[j];
            ob[j] = bf(r[j]);
            amax = fmaxf(amax, fabsf(ob[j]));
        }
#pragma unroll
        for (int j = 0; j < VPT / 2; j++) ow[j] = bfbits(r[2 * j]) | (bfbits(r[2 * j + 1]) << 16);
        if (act) {
            *(float4*)(f32_out + off) = make_float4(r[0], r[1], r[2], r[3]);
            *(float4*)(f32_out + off + 4) = make_float4(r[4], r[5], r[6], r[7]);
            *(uint4*)(out + off) = make_uint4(ow[0], ow[1], ow[2], ow[3]);
        }
        if (QUANT) {
#ifndef NOFUSEMAX
            float sx = sxf;
#else
            float sx = fmaxf(block_max(amax, sh), 1e-10f) / 448.f;
#endif
            if (act) {
#ifdef SWCVT
                uint32_t q[2] = {0, 0};
#pragma unroll
                for (int j = 0; j < VPT; j++) {
                    float v = ob[j] / sx;
                    v = fminf(fmaxf(v, -448.f), 448.f);
                    q[j >> 2] |= (uint32_t)to_e4m3(v) << (8 * (j & 3));
                }
#else
                float v[VPT];
#pragma unroll
                for (int j = 0; j < VPT; j++) v[j] = fminf(fmaxf(ob[j] / sx, -448.f), 448.f);
                int q[2];
                q[0] = __builtin_hcu_cvt_pk_fp8_f32(v[0], v[1], 0, false);
                q[0] = __builtin_hcu_cvt_pk_fp8_f32(v[2], v[3], q[0], true);
                q[1] = __builtin_hcu_cvt_pk_fp8_f32(v[4], v[5], 0, false);
                q[1] = __builtin_hcu_cvt_pk_fp8_f32(v[6], v[7], q[1], true);
#endif
                *(uint2*)(xq + off) = make_uint2((uint32_t)q[0], (uint32_t)q[1]);
            }
            if (threadIdx.x == 0) xs[row] = sx;
        }
    }
    T(3)
    if (END) end_sync(s0, s1, self, rank);
    T(4)
}

template <int K>
__global__ void __launch_bounds__(NT, 1) dbg_kernel(Signal* s0, Signal* s1, Signal* self, int rank) {
    if (K >= 1) start_sync(s0, s1, self, rank);
    if (K >= 2) end_sync(s0, s1, self, rank);
}

extern "C" int welm_arn_dbg(int k, int grid, void* s0, void* s1, void* self, int rank, hipStream_t stream) {
    if (k == 0) hipLaunchKernelGGL(dbg_kernel<0>, dim3(grid), dim3(NT), 0, stream, (Signal*)s0, (Signal*)s1, (Signal*)self, rank);
    if (k == 1) hipLaunchKernelGGL(dbg_kernel<1>, dim3(grid), dim3(NT), 0, stream, (Signal*)s0, (Signal*)s1, (Signal*)self, rank);
    if (k == 2) hipLaunchKernelGGL(dbg_kernel<2>, dim3(grid), dim3(NT), 0, stream, (Signal*)s0, (Signal*)s1, (Signal*)self, rank);
    return (int)hipGetLastError();
}

extern "C" int welm_arn_launch(int mode, int quant, int grid, const void* in0, const void* in1, void* s0, void* s1,
                               void* self, int rank, int rows, int cols, const void* g0, const void* g1,
                               const void* res, void* out, void* res_out, void* f32_out, void* xq, void* xs,
                               float eps, int end, hipStream_t stream) {
    if (cols > NT * 8 || cols % 8 || grid < 1 || grid > MAXB) return -1;
#define L(M, Q, E) hipLaunchKernelGGL((arn_kernel<M, Q, 8, E>), dim3(grid), dim3(NT), 0, stream,                       \
                                   (const __hip_bfloat16*)in0, (const __hip_bfloat16*)in1, (Signal*)s0,          \
                                   (Signal*)s1, (Signal*)self, rank, rows, cols, (const __hip_bfloat16*)g0,      \
                                   (const __hip_bfloat16*)g1, (const float*)res, (__hip_bfloat16*)out,           \
                                   (float*)res_out, (float*)f32_out, (uint8_t*)xq, (float*)xs, eps)
    // end == 0: caller guarantees the inputs live in a region no producer rewrites before the next
    // fused kernel's start barrier (double-buffered pool regions), so the trailing barrier is skipped.
    if (end) {
        if (mode == 0 && quant) L(0, true, true);
        else if (mode == 0) L(0, false, true);
        else L(1, false, true);
    } else {
        if (mode == 0 && quant) L(0, true, false);
        else if (mode == 0) L(0, false, false);
        else L(1, false, false);
    }
#undef L
    return (int)hipGetLastError();
}

extern "C" int welm_arn_signal_size() { return (int)sizeof(Signal); }

extern "C" void welm_arn_ts(unsigned long long* o) {
#ifdef TS
    hipMemcpyFromSymbol(o, HIP_SYMBOL(g_ts), sizeof(unsigned long long) * 8);
#endif
}
