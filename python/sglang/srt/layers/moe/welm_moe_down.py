"""Triton fp8 MoE down GEMM for HCU (WeLM).

Per-token fp8 activations x per-channel fp8 weights, plain [E, N, K] weight layout.
On gfx938 this is ~1.5x faster than the lightop marlin down kernel at prefill-sized M
(M=20480: 2.4 vs 4.0 ms), while marlin stays faster for decode-sized M.

Enabled with WELM_TRITON_MOE_DOWN_MIN_M=<tokens> (0 = off). Reuses the sorted ids of the
up GEMM, which are aligned to its BLOCK_SIZE_M (128 at prefill M).
"""
import os

import torch
import triton
import triton.language as tl

WELM_TRITON_MOE_DOWN_MIN_M = int(os.environ.get("WELM_TRITON_MOE_DOWN_MIN_M", "0"))

# (min_m, BM, BN, BK, num_warps, num_stages, waves_per_eu, matrix_instr_nonkdim), largest min_m first.
_CFGS = [
    (16384, 128, 128, 128, 4, 1, 1, 16),
    (0, 128, 128, 128, 4, 1, 0, 16),
]


@triton.jit
def _moe_down_fp8_kernel(
    a, b, c, a_s, b_s, tw, sorted_ids, expert_ids, npad_ptr, num_valid, N, K,
    sam, sak, sbe, sbn, sbk, scm, sbse,
    ALIGN_DIV: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, G: tl.constexpr,
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
    bp = b + e * sbe + on[None, :] * sbn + ok[:, None] * sbk
    acc = tl.zeros((BM, BN), tl.float32)
    for _ in range(0, K, BK):
        acc += tl.dot(tl.load(ap, mask=valid[:, None], other=0.0), tl.load(bp))
        ap += BK * sak
        bp += BK * sbk
    sa = tl.load(a_s + tok, mask=valid, other=0.0).to(tl.float32)
    sb = tl.load(b_s + e * sbse + on).to(tl.float32)
    w = tl.load(tw + tok, mask=valid, other=0.0).to(tl.float32)
    acc = acc * (sa * w)[:, None] * sb[None, :]
    tl.store(c + tok[:, None].to(tl.int64) * scm + on[None, :], acc.to(c.dtype.element_ty), mask=valid[:, None])


def use_triton_down(m: int, align_block: int) -> bool:
    return WELM_TRITON_MOE_DOWN_MIN_M > 0 and m >= WELM_TRITON_MOE_DOWN_MIN_M and align_block % 32 == 0


def moe_down_fp8(a, a_scale, w, w_scale, out, topk_weights, sorted_ids, expert_ids, num_tokens_post_padded, align_block, m):
    """a [M*topk, K] fp8, a_scale [M*topk], w [E, N, K] fp8, w_scale [E, N(,1)], out [M*topk, N]."""
    E, N, K = w.shape
    cfg = next(c for c in _CFGS if m >= c[0])
    _, BM, BN, BK, nw, ns, wpe, nk = cfg
    BM = min(BM, align_block)
    assert align_block % BM == 0 and K % BK == 0 and N % BN == 0
    grid = (triton.cdiv(sorted_ids.numel(), BM) * triton.cdiv(N, BN),)
    _moe_down_fp8_kernel[grid](
        a, w, out, a_scale, w_scale, topk_weights, sorted_ids, expert_ids, num_tokens_post_padded,
        out.shape[0], N, K, a.stride(0), a.stride(1), w.stride(0), w.stride(1), w.stride(2),
        out.stride(0), w_scale.stride(0),
        ALIGN_DIV=align_block // BM, BM=BM, BN=BN, BK=BK, G=8,
        num_warps=nw, num_stages=ns, waves_per_eu=wpe, matrix_instr_nonkdim=nk,
    )
    return out
