"""Triton fp8 MoE down GEMM for HCU (WeLM).

Per-token fp8 activations x per-channel fp8 weights. Reads either a plain [E, N, K] weight or,
in serving, the lightop marlin-packed w2 itself ([E, K/64, N*64]), so no extra weight copy is kept.
Under CUDA graphs on gfx938 (TP2 shapes) it ties marlin for M < 1024 and beats it above
(M=1024 1.21x, 4096 1.77x, 20480 1.82x: 2.0 vs 3.7 ms).

Enabled with WELM_TRITON_MOE_DOWN_MIN_M=<tokens> (0 = off). Reuses the sorted ids of the
up GEMM, which are aligned to its BLOCK_SIZE_M (128 at prefill M).
"""
import os

import torch
import triton
import triton.language as tl

WELM_TRITON_MOE_DOWN_MIN_M = int(os.environ.get("WELM_TRITON_MOE_DOWN_MIN_M", "0"))
# Rows (tokens x topk) from which SwiGLU + per-token fp8 quant runs in Triton instead of lightop
# (2.7x faster at 200k rows; below this the eager launch overhead dominates). 0 = off.
WELM_TRITON_SILU_QUANT_MIN_ROWS = int(os.environ.get("WELM_TRITON_SILU_QUANT_MIN_ROWS", "8192"))

# (min_m, BM, BN, BK, num_warps, num_stages, waves_per_eu, matrix_instr_nonkdim), largest min_m first.
# Tuned under CUDA graphs on the marlin-packed TP2 w2 (moedown/mk_graph.py); M < 1024 ties marlin.
_CFGS = [
    (16384, 128, 128, 128, 2, 1, 1, 16),
    (2048, 128, 128, 128, 4, 1, 0, 16),
    (1024, 64, 128, 64, 2, 1, 0, 16),
    (512, 16, 256, 64, 1, 1, 0, 16),
    (64, 16, 128, 64, 4, 1, 0, 16),
    (0, 16, 64, 64, 1, 1, 1, 16),
]


@triton.jit
def _moe_down_fp8_kernel(
    a, b, c, a_s, b_s, tw, sorted_ids, expert_ids, npad_ptr, num_valid, N, K,
    sam, sak, sbe, sbn, sbk, scm, sbse,
    ALIGN_DIV: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, G: tl.constexpr,
    MK: tl.constexpr = 0,
):
    pid = tl.program_id(0)
    npad = tl.load(npad_ptr)
    npm = tl.cdiv(npad, BM)
    npn = tl.cdiv(N, BN)
    # The grid is sized for the max padded length; npad is only known on device.
    if pid >= npm * npn:
        return
    gid = pid // (G * npn)
    fm = gid * G
    gs = min(npm - fm, G)
    pm = fm + ((pid % (G * npn)) % gs)
    pn = (pid % (G * npn)) // gs
    e = tl.load(expert_ids + pm // ALIGN_DIV).to(tl.int64)
    offs = pm * BM + tl.arange(0, BM)
    tok = tl.load(sorted_ids + offs)
    valid = tok < num_valid
    on = pn * BN + tl.arange(0, BN)
    ok = tl.arange(0, BK)
    ap = a + tok[:, None].to(tl.int64) * sam + ok[None, :] * sak
    acc = tl.zeros((BM, BN), tl.float32)
    if MK == 0:
        bp = b + e * sbe + on[None, :] * sbn + ok[:, None] * sbk
        for _ in range(0, K, BK):
            acc += tl.dot(tl.load(ap, mask=valid[:, None], other=0.0), tl.load(bp))
            ap += BK * sak
            bp += BK * sbk
    else:
        # lightop marlin fp8 layout: per expert [K/MK, N, MK], element (n, k) at (k//MK)*N*MK + n*MK + k%MK.
        bb = b + e * sbe + on[None, :] * MK
        for k0 in range(0, K, BK):
            kk = k0 + ok
            acc += tl.dot(tl.load(ap, mask=valid[:, None], other=0.0),
                          tl.load(bb + ((kk // MK) * (N * MK) + kk % MK)[:, None]))
            ap += BK * sak
    sa = tl.load(a_s + tok, mask=valid, other=0.0).to(tl.float32)
    sb = tl.load(b_s + e * sbse + on).to(tl.float32)
    w = tl.load(tw + tok, mask=valid, other=0.0).to(tl.float32)
    acc = acc * (sa * w)[:, None] * sb[None, :]
    tl.store(c + tok[:, None].to(tl.int64) * scm + on[None, :], acc.to(c.dtype.element_ty), mask=valid[:, None])


def use_triton_down(m: int, align_block: int) -> bool:
    return WELM_TRITON_MOE_DOWN_MIN_M > 0 and m >= WELM_TRITON_MOE_DOWN_MIN_M and align_block % 32 == 0


def moe_down_fp8(a, a_scale, w, w_scale, out, topk_weights, sorted_ids, expert_ids, num_tokens_post_padded, align_block, m,
                 marlin_shape=None, cfg=None):
    """a [M*topk, K] fp8, a_scale [M*topk], w [E, N, K] fp8, w_scale [E, N(,1)], out [M*topk, N].

    marlin_shape=(E, N, K): w is the lightop marlin-packed copy ([E, K/64, N*64]) instead of plain [E, N, K].
    """
    if marlin_shape is not None:
        E, N, K = marlin_shape
        mk = 64
        assert w[0].numel() == N * K
    else:
        E, N, K = w.shape
        mk = 0
    cfg = cfg or next(c for c in _CFGS if m >= c[0])
    _, BM, BN, BK, nw, ns, wpe, nk = cfg
    BM = min(BM, align_block)
    assert align_block % BM == 0 and K % BK == 0 and N % BN == 0
    # Padded length is at most rows + (active experts) * (align_block - 1); don't launch the empty tail.
    rows = a.shape[0]
    max_pad = min(sorted_ids.numel(), rows + min(E, rows) * (align_block - 1))
    grid = (triton.cdiv(max_pad, BM) * triton.cdiv(N, BN),)
    _moe_down_fp8_kernel[grid](
        a, w, out, a_scale, w_scale, topk_weights, sorted_ids, expert_ids, num_tokens_post_padded,
        out.shape[0], N, K, a.stride(0), a.stride(1), w.stride(0), *((w.stride(1), w.stride(2)) if mk == 0 else (0, 0)),
        out.stride(0), w_scale.stride(0),
        ALIGN_DIV=align_block // BM, BM=BM, BN=BN, BK=BK, G=8, MK=mk,
        num_warps=nw, num_stages=ns, waves_per_eu=wpe, matrix_instr_nonkdim=nk,
    )
    return out


@triton.jit
def _silu_mul_fp8_quant_kernel(x, o, s, R, D: tl.constexpr, BR: tl.constexpr):
    r = tl.program_id(0) * BR + tl.arange(0, BR)
    m = r < R
    c = tl.arange(0, D)
    base = x + r[:, None].to(tl.int64) * (2 * D) + c[None, :]
    g = tl.load(base, mask=m[:, None], other=0.0).to(tl.float32)
    u = tl.load(base + D, mask=m[:, None], other=0.0).to(tl.float32)
    # Same bf16 rounding of silu(g) and of the product as lightop fuse_silu_mul_fp8_quant.
    y = ((g / (1.0 + tl.exp2(-g * 1.4426950408889634))).to(tl.bfloat16).to(tl.float32) * u)
    y = y.to(tl.bfloat16).to(tl.float32)
    amax = tl.maximum(tl.max(tl.abs(y), axis=1), 1e-10)
    q = tl.clamp(y * (448.0 / amax)[:, None], -448.0, 448.0).to(tl.float8e4nv)
    tl.store(o + r[:, None].to(tl.int64) * D + c[None, :], q, mask=m[:, None])
    tl.store(s + r, amax / 448.0, mask=m)


def use_triton_silu_quant(x) -> bool:
    return (0 < WELM_TRITON_SILU_QUANT_MIN_ROWS <= x.shape[0] and x.dim() == 2 and x.is_contiguous()
            and x.dtype == torch.bfloat16 and x.shape[1] in (256, 512, 1024))


def silu_mul_fp8_quant(x):
    """x [R, 2D] bf16 (gate | up) -> (fp8 e4m3 [R, D], fp32 scale [R, 1])."""
    R, D = x.shape[0], x.shape[1] // 2
    o = torch.empty(R, D, dtype=torch.float8_e4m3fn, device=x.device)
    s = torch.empty(R, 1, dtype=torch.float32, device=x.device)
    _silu_mul_fp8_quant_kernel[(triton.cdiv(R, 16),)](x, o, s, R, D=D, BR=16, num_warps=4)
    return o, s
