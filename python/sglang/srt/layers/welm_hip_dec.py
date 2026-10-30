"""HIP split-KV decode attention for WeLM (gfx938 mmac), see welm_hip_dec.hip.

Same contract as welmv4_fast_op.welm_decode_attention for D=256, PAGE=64,
G<=16, bf16 or fp8 e4m3 KV. fp8 KV uses the cooperative fp8-mmac partial kernel
(mode 3, q and P split into fp8 hi/lo); partials are merged by a HIP reduce kernel.

libwelm_hip_dec.so is not tracked (git-ignored); build it on the target machine with
`bash welm_hip_dec_build.sh` (needs /opt/dtk hipcc, gfx938)."""
import ctypes, os
import torch, triton

_LIB = None
_LOG2E = 1.4426950408889634
_P, _I, _L = ctypes.c_void_p, ctypes.c_int, ctypes.c_long


def _lib():
    global _LIB
    if _LIB is None:
        so = os.path.join(os.path.dirname(os.path.abspath(__file__)), "libwelm_hip_dec.so")
        if not os.path.exists(so):
            raise RuntimeError(f"{so} not found: WELM_HIP_DEC is set, run `bash welm_hip_dec_build.sh` in {os.path.dirname(so)}")
        _LIB = ctypes.CDLL(so)
        _LIB.welm_dec_partial_launch.restype = ctypes.c_int
        _LIB.welm_dec_reduce_launch.restype = ctypes.c_int
        _LIB.welm_pfx_partial_launch.restype = ctypes.c_int
    return _LIB


def supported(q, k_cache, v_cache):
    B, H, D = q.shape
    KVH, PAGE = k_cache.shape[1], k_cache.shape[2]
    return (D == 256 and PAGE == 64 and H % KVH == 0 and H // KVH <= 16
            and q.stride(2) == 1 and k_cache.stride(3) == 1 and v_cache.stride(3) == 1
            and k_cache.dtype in (torch.bfloat16, torch.float8_e4m3fn, torch.uint8))


def hip_decode(q, k_cache, v_cache, block_table, seq_lens, softmax_scale, max_context_len,
               sinks=None, out=None, window_left=-1, kv_scales=None, cp=None, block_s=32, gate=None):
    """gate (B, H), stride(1) == 1: also apply out *= sigmoid(gate) (sigmoid_mul_strided_)."""
    from sglang.srt.layers.welmv4_fast_op import _welm_dec_attn_reduce_kernel
    B, H, D = q.shape
    KVH, PAGE = k_cache.shape[1], k_cache.shape[2]
    G = H // KVH
    ks, vs = kv_scales if kv_scales is not None else (1.0, 1.0)
    kv8 = k_cache.dtype in (torch.float8_e4m3fn, torch.uint8)
    f8 = 1.0
    if kv8:
        k_cache, v_cache = k_cache.view(torch.uint8), v_cache.view(torch.uint8)
    if cp is None:
        cp = 4 if window_left >= 0 else 8
    max_splits = triton.cdiv(max_context_len, cp * PAGE)
    gs = min(max_splits, triton.cdiv(window_left + 1, cp * PAGE) + 1) if window_left >= 0 else max_splits
    mid_o = torch.empty((B, H, gs, D), dtype=torch.float32, device=q.device)
    mid_lse = torch.empty((B, H, gs), dtype=torch.float32, device=q.device)
    if out is None:
        out = torch.empty_like(q)
    r = _lib().welm_dec_partial_launch(
        _I(3 if kv8 else 0), _I(cp), _I(B * KVH), _I(gs), _P(q.data_ptr()), _P(k_cache.data_ptr()), _P(v_cache.data_ptr()),
        _P(block_table.data_ptr()), _P(seq_lens.data_ptr()), _P(mid_o.data_ptr()), _P(mid_lse.data_ptr()),
        ctypes.c_float(softmax_scale * ks * f8 * _LOG2E),
        _L(q.stride(0)), _L(q.stride(1)), _L(k_cache.stride(0)), _L(k_cache.stride(1)), _L(k_cache.stride(2)),
        _L(v_cache.stride(0)), _L(v_cache.stride(1)), _L(v_cache.stride(2)), _L(block_table.stride(0)),
        _L(mid_o.stride(0)), _L(mid_o.stride(1)), _L(mid_o.stride(2)), _L(mid_lse.stride(0)), _L(mid_lse.stride(1)),
        _I(KVH), _I(G), _I(window_left), _P(torch.cuda.current_stream().cuda_stream))
    if r != 0:
        raise RuntimeError(f"welm_hip_dec launch failed: {r}")
    if out.dtype == torch.bfloat16 and out.stride(2) == 1:
        sk = 0 if sinks is None else (1 if sinks.dtype == torch.bfloat16 else 2)
        r = _lib().welm_dec_reduce_launch(
            _I(B), _I(H), _P(mid_o.data_ptr()), _P(mid_lse.data_ptr()), _P(seq_lens.data_ptr()),
            _P(sinks.data_ptr() if sinks is not None else 0), _I(sk), _P(out.data_ptr()),
            _L(mid_o.stride(0)), _L(mid_o.stride(1)), _L(mid_o.stride(2)), _L(mid_lse.stride(0)), _L(mid_lse.stride(1)),
            _L(out.stride(0)), _L(out.stride(1)), _I(cp * PAGE), _I(window_left), ctypes.c_float(vs * f8),
            _P(gate.data_ptr() if gate is not None else 0), _L(gate.stride(0) if gate is not None else 0),
            _P(torch.cuda.current_stream().cuda_stream))
        if r != 0:
            raise RuntimeError(f"welm_hip_dec reduce launch failed: {r}")
        return out
    _welm_dec_attn_reduce_kernel[(B, H)](
        mid_o, mid_lse, seq_lens, sinks if sinks is not None else mid_lse, out,
        mid_o.stride(0), mid_o.stride(1), mid_o.stride(2), mid_lse.stride(0), mid_lse.stride(1),
        out.stride(0), out.stride(1),
        D=D, PAGE=PAGE, CHUNK_PAGES=cp, BLOCK_S=block_s, HAS_SINK=sinks is not None,
        WINDOW_LEFT=window_left, v_scale=vs * f8, num_warps=4)
    if gate is not None:
        from sglang.srt.layers.welmv4_fast_op import sigmoid_mul_strided_

        sigmoid_mul_strided_(gate, out.view(B, H * D), H)
    return out


def hip_prefix(q, k_cache, v_cache, block_table, seqused_k, cu_seqlens_q, max_seqlen_q, softmax_scale,
               sinks=None, out=None, kv_scales=None, num_splits=None):
    """fp8 KV drop-in for welmv4_fast_op.welm_prefix_attention (fp8-mmac partial,
    Triton _welm_pfx_attn_reduce_kernel)."""
    from sglang.srt.layers.welmv4_fast_op import _welm_pfx_attn_reduce_kernel
    total_q, H, D = q.shape
    ks, vs = kv_scales if kv_scales is not None else (1.0, 1.0)
    k8, v8 = k_cache.view(torch.uint8), v_cache.view(torch.uint8)
    KVH = k8.shape[1]
    G = H // KVH
    bs = cu_seqlens_q.shape[0] - 1
    row_tiles = triton.cdiv(max_seqlen_q * G, 64)
    if num_splits is None:  # B4 x 256q L8000: ns1 2.3 ms vs ns32 3.1 ms; small grids want a few splits
        num_splits = min(32, triton.next_power_of_2(triton.cdiv(128, bs * KVH * row_tiles)))  # reduce aranges it
    mid_o = torch.empty((total_q, H, num_splits, D), dtype=torch.float32, device=q.device)
    mid_lse = torch.empty((total_q, H, num_splits), dtype=torch.float32, device=q.device)
    if out is None:
        out = torch.empty_like(q)
    r = _lib().welm_pfx_partial_launch(
        _I(bs * KVH), _I(row_tiles), _I(num_splits), _P(q.data_ptr()), _P(k8.data_ptr()), _P(v8.data_ptr()),
        _P(block_table.data_ptr()), _P(seqused_k.data_ptr()), _P(cu_seqlens_q.data_ptr()),
        _P(mid_o.data_ptr()), _P(mid_lse.data_ptr()), ctypes.c_float(softmax_scale * ks * _LOG2E),
        _L(q.stride(0)), _L(q.stride(1)), _L(k8.stride(0)), _L(k8.stride(1)), _L(k8.stride(2)),
        _L(v8.stride(0)), _L(v8.stride(1)), _L(v8.stride(2)), _L(block_table.stride(0)),
        _L(mid_o.stride(0)), _L(mid_o.stride(1)), _L(mid_o.stride(2)), _L(mid_lse.stride(0)), _L(mid_lse.stride(1)),
        _I(KVH), _I(G), _P(torch.cuda.current_stream().cuda_stream))
    if r != 0:
        raise RuntimeError(f"welm_hip_dec prefix launch failed: {r}")
    _welm_pfx_attn_reduce_kernel[(total_q, H)](
        mid_o, mid_lse, sinks if sinks is not None else mid_lse, out,
        mid_o.stride(0), mid_o.stride(1), mid_o.stride(2), mid_lse.stride(0), mid_lse.stride(1),
        out.stride(0), out.stride(1), D=D, NUM_SPLITS=num_splits, HAS_SINK=sinks is not None, v_scale=vs,
        num_warps=4)
    return out
