# Fused gather + per-tensor e4m3 quantization of a batch's KV pages for the fa_src
# fp8 prefix-prefill kernel.
#   key_cache   [NP, HK, P, D] bf16 (legacy_bhsd)      -> k8 [M, P, HK, D] e4m3 (bshd)
#   value_cache [NP, HK, D, P] bf16 (legacy, V^T pages) -> v8 [M, P, HK, D] e4m3 (bshd)
# M = B * maxp gathered pages (page_table[:, :maxp]); tokens past seqused are
# excluded from the amax (their fp8 values are don't-care).
import torch
import triton
import triton.language as tl


@triton.jit
def _amax_kernel(kc, vc, pt, su, amax, maxp, pt_stride, HK: tl.constexpr, P: tl.constexpr, D: tl.constexpr):
    m = tl.program_id(0); h = tl.program_id(1)
    b = m // maxp; j = m % maxp
    page = tl.load(pt + b * pt_stride + j).to(tl.int64)
    n = tl.load(su + b) - j * P
    t = tl.arange(0, P); d = tl.arange(0, D)
    ok = (t < n)
    kt = tl.load(kc + ((page * HK + h) * P + t[:, None]) * D + d[None, :], mask=ok[:, None], other=0.0)
    vt = tl.load(vc + ((page * HK + h) * D + d[:, None]) * P + t[None, :], mask=ok[None, :], other=0.0)
    tl.atomic_max(amax, tl.max(tl.max(tl.abs(kt.to(tl.float32)), 1), 0))
    tl.atomic_max(amax + 1, tl.max(tl.max(tl.abs(vt.to(tl.float32)), 1), 0))


@triton.jit
def _quant_kernel(kc, vc, pt, amax, k8, v8, maxp, pt_stride, HK: tl.constexpr, P: tl.constexpr, D: tl.constexpr):
    m = tl.program_id(0); h = tl.program_id(1)
    b = m // maxp; j = m % maxp
    page = tl.load(pt + b * pt_stride + j).to(tl.int64)
    t = tl.arange(0, P); d = tl.arange(0, D)
    ik = 448.0 / tl.maximum(tl.load(amax), 1e-6)
    iv = 448.0 / tl.maximum(tl.load(amax + 1), 1e-6)
    kt = tl.load(kc + ((page * HK + h) * P + t[:, None]) * D + d[None, :]).to(tl.float32) * ik
    vt = tl.load(vc + ((page * HK + h) * D + d[:, None]) * P + t[None, :]).to(tl.float32) * iv  # [D, P]
    kt = tl.minimum(tl.maximum(kt, -448.0), 448.0)
    vt = tl.minimum(tl.maximum(vt, -448.0), 448.0)
    m64 = m.to(tl.int64)
    tl.store(k8 + ((m64 * P + t[:, None]) * HK + h) * D + d[None, :], kt.to(tl.float8e4nv))
    tl.store(v8 + ((m64 * P + t[None, :]) * HK + h) * D + d[:, None], vt.to(tl.float8e4nv))


@triton.jit
def _q_kernel(q, q8, ks_amax, numel, cinv, BLOCK: tl.constexpr):
    # q8 = clamp(q * ks / C); ks = amax_k / 448
    i = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    s = tl.load(ks_amax) / 448.0 * cinv
    x = tl.load(q + i, mask=i < numel, other=0.0).to(tl.float32) * s
    x = tl.minimum(tl.maximum(x, -448.0), 448.0)
    tl.store(q8 + i, x.to(tl.float8e4nv), mask=i < numel)


def gather_quant_kv(key_cache, value_cache, page_table, seqused, maxp):
    """Returns k8, v8 [B*maxp, P, HK, D] e4m3 and amax fp32 [2] (k, v) on device."""
    NP, HK, P, D = key_cache.shape
    B = seqused.shape[0]
    M = B * maxp
    amax = torch.zeros(2, device=key_cache.device, dtype=torch.float32)
    k8 = torch.empty((M, P, HK, D), device=key_cache.device, dtype=torch.float8_e4m3fn)
    v8 = torch.empty_like(k8)
    pt = page_table if page_table.stride(1) == 1 else page_table.contiguous()
    su = seqused.to(torch.int32)
    _amax_kernel[(M, HK)](key_cache, value_cache, pt, su, amax, maxp, pt.stride(0), HK=HK, P=P, D=D, num_warps=8)
    _quant_kernel[(M, HK)](key_cache, value_cache, pt, amax, k8, v8, maxp, pt.stride(0), HK=HK, P=P, D=D, num_warps=8)
    return k8, v8, amax


def quant_q(q, amax, C):
    """q8 = clamp(q * (amax_k/448) / C) in e4m3 (q descale = C / ks)."""
    q = q.contiguous()
    q8 = torch.empty(q.shape, device=q.device, dtype=torch.float8_e4m3fn)
    n = q.numel(); BLOCK = 4096
    _q_kernel[(triton.cdiv(n, BLOCK),)](q, q8, amax, n, 1.0 / C, BLOCK=BLOCK, num_warps=8)
    return q8


@triton.jit
def _gather_fp8_kernel(kc, vc, pt, k8, v8, maxp, pt_stride, HK: tl.constexpr, P: tl.constexpr, D: tl.constexpr):
    # fp8 legacy pages -> bshd pages, no requantization
    m = tl.program_id(0); h = tl.program_id(1)
    b = m // maxp; j = m % maxp
    page = tl.load(pt + b * pt_stride + j).to(tl.int64)
    t = tl.arange(0, P); d = tl.arange(0, D)
    kt = tl.load(kc + ((page * HK + h) * P + t[:, None]) * D + d[None, :])
    vt = tl.load(vc + ((page * HK + h) * D + d[:, None]) * P + t[None, :])  # [D, P]
    m64 = m.to(tl.int64)
    tl.store(k8 + ((m64 * P + t[:, None]) * HK + h) * D + d[None, :], kt)
    tl.store(v8 + ((m64 * P + t[None, :]) * HK + h) * D + d[:, None], vt)


def gather_kv_fp8(key_cache, value_cache, page_table, maxp):
    """fp8 key_cache [NP, HK, P, D] / value_cache [NP, HK, D, P] -> k8, v8 [B*maxp, P, HK, D]."""
    NP, HK, P, D = key_cache.shape
    B = page_table.shape[0]
    M = B * maxp
    k8 = torch.empty((M, P, HK, D), device=key_cache.device, dtype=torch.float8_e4m3fn)
    v8 = torch.empty_like(k8)
    pt = page_table if page_table.stride(1) == 1 else page_table.contiguous()
    _gather_fp8_kernel[(M, HK)](key_cache, value_cache, pt, k8, v8, maxp, pt.stride(0), HK=HK, P=P, D=D, num_warps=8)
    return k8, v8
