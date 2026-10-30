"""WELM_ARN: fused TP2 all-reduce + WeLM RMSNorm(s) for decode CUDA graphs (see welm_arn.hip).

The producer (o_proj / fp8 MoE moe_sum) writes its partial sum into the aiter AR input pool
(welm_ar_pool.view); instead of aiter AR + a separate norm kernel, one kernel syncs with the
peer, sums both ranks' pool rows (peer pool IPC-mapped here) and applies
  mode 0: mmq_style_norm_after_attn (o_norm, + residual, post-attn norm, optional fp8 quant)
  mode 1: ppln input_layernorm (+ fp32 residual, rms norm, fp32 copy)
libwelm_arn.so: hipcc -shared -fPIC -O3 -std=c++17 --offload-arch=gfx938 -fno-gpu-rdc -o libwelm_arn.so welm_arn.hip
"""
import ctypes
import logging
import os

import torch

logger = logging.getLogger(__name__)
ENABLED = os.environ.get("WELM_ARN", "0") == "1"
_P, _I = ctypes.c_void_p, ctypes.c_int
_LIB = None
_HIP = None
_ST = None  # dict(rank, pool=[p0, p1], sig=[s0, s1], self_sig) once set up; False = failed
_MAXB = 128
_pending = None  # (data_ptr, shape) of an un-reduced pool tensor awaiting a fused AR+norm
# WELM_ARN_NOEND: attn / MoE partial sums go to their own pool regions (offset 1 / 2 MiB; aiter AR and
# its copy-in use offset 0). A region is rewritten only after the other kind's fused kernel passed its
# start barrier (=> the peer finished reading it), so the fused kernel can skip its trailing barrier.
NOEND = os.environ.get("WELM_ARN_NOEND", "1") == "1"
_REGION = 1 << 20


class _Handle(ctypes.Structure):
    _fields_ = [("reserved", ctypes.c_char * 64)]


def _lib():
    global _LIB, _HIP
    if _LIB is None:
        so = os.environ.get("WELM_ARN_SO") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "libwelm_arn.so")
        _LIB = ctypes.CDLL(so)
        _LIB.welm_arn_launch.restype = ctypes.c_int
        _LIB.welm_arn_launch.argtypes = [_I, _I, _I, _P, _P, _P, _P, _P, _I, _I, _I, _P, _P, _P, _P, _P, _P, _P, _P,
                                         ctypes.c_float, _I, _P]
        _HIP = ctypes.CDLL("libamdhip64.so")
        _HIP.hipIpcGetMemHandle.argtypes = [ctypes.POINTER(_Handle), _P]
        _HIP.hipIpcOpenMemHandle.argtypes = [ctypes.POINTER(_P), _Handle, ctypes.c_uint]
        _HIP.hipMemset.argtypes = [_P, ctypes.c_int, ctypes.c_size_t]
    return _LIB


def _ipc_handle(ptr):
    h = _Handle()
    r = _HIP.hipIpcGetMemHandle(ctypes.byref(h), _P(ptr))
    if r != 0:
        raise RuntimeError(f"hipIpcGetMemHandle {r}")
    return bytes(memoryview(h))


def _ipc_open(b):
    h = _Handle.from_buffer_copy(b)
    p = _P()
    r = _HIP.hipIpcOpenMemHandle(ctypes.byref(p), h, ctypes.c_uint(1))  # hipIpcMemLazyEnablePeerAccess
    if r != 0:
        raise RuntimeError(f"hipIpcOpenMemHandle {r}")
    return p.value


def _ca():
    from sglang.srt.layers import welm_ar_pool

    ca = welm_ar_pool._ca()
    if ca is None or ca.world_size != 2:
        return None
    return ca


def setup(ca=None):
    """Collective over the TP group (both ranks call it at the same point); eager only."""
    global _ST
    if _ST is not None or not (ENABLED or ca is not None):
        return
    if torch.cuda.is_current_stream_capturing():
        return
    ca = ca if ca is not None else _ca()
    if ca is None:
        _ST = False
        return
    import torch.distributed as dist
    import aiter as ops

    err = None
    try:
        _lib()
        sig_size = _LIB.welm_arn_signal_size()
        sig = ops.allocate_meta_buffer(sig_size, 0)
        _HIP.hipMemset(_P(sig), 0, ctypes.c_size_t(sig_size))
        torch.cuda.synchronize()
        pool = ca._pool["input"].data_ptr
        mine = (_ipc_handle(pool), _ipc_handle(sig))
    except Exception as e:  # noqa: BLE001
        err, mine = repr(e), None
    allh = [None, None]
    dist.all_gather_object(allh, (err, mine), group=ca.group)
    if any(a[0] for a in allh):
        logger.warning("WELM_ARN disabled: %s", [a[0] for a in allh])
        _ST = False
        return
    rank = ca.rank
    pools, sigs = [0, 0], [0, 0]
    for r in range(2):
        if r == rank:
            pools[r], sigs[r] = pool, sig
        else:
            pools[r], sigs[r] = _ipc_open(allh[r][1][0]), _ipc_open(allh[r][1][1])
    torch.cuda.synchronize()
    dist.barrier(group=ca.group)
    _ST = dict(rank=rank, pool=pools, sig=sigs)
    logger.info("WELM_ARN: fused AR+norm ready (rank %d)", rank)


def ready():
    return bool(_ST) and torch.cuda.is_current_stream_capturing()


def region_off(kind, nbytes):
    """Pool byte offset for a deferred (fused) partial sum: kind 1 = attention o_proj, 2 = MoE."""
    if not NOEND or nbytes > _REGION:
        return 0
    from sglang.srt.layers import welm_ar_pool

    ca = welm_ar_pool._ca()
    if ca is None or ca._pool["input"].max_size < 3 * _REGION:
        return 0
    return kind * _REGION


def set_pending(t):
    global _pending
    _pending = (t.data_ptr(), tuple(t.shape))


def take_pending(t):
    """True if t is the un-reduced pool tensor left by set_pending (consumes it)."""
    global _pending
    if _pending is None:
        return False
    ok = _pending == (t.data_ptr(), tuple(t.shape))
    _pending = None
    if not ok:
        raise RuntimeError("WELM_ARN: pending un-reduced tensor was not consumed by the fused norm")
    return True


def check_no_pending():
    if _pending is not None:
        raise RuntimeError("WELM_ARN: un-reduced tensor left pending")


def _launch(mode, x, g0, g1, res, out, res_out, f32_out, xq, xs, eps):
    rows, cols = x.shape
    st = _ST
    off = x.data_ptr() - st["pool"][st["rank"]]
    grid = max(1, min(rows, _MAXB))
    end = 0 if off >= _REGION else 1
    r = _LIB.welm_arn_launch(
        mode, 1 if xq is not None else 0, grid, st["pool"][0] + off, st["pool"][1] + off, st["sig"][0], st["sig"][1],
        st["sig"][st["rank"]], st["rank"], rows, cols, g0.data_ptr(), g1.data_ptr() if g1 is not None else 0,
        res.data_ptr(), out.data_ptr(), res_out.data_ptr() if res_out is not None else 0, f32_out.data_ptr(),
        xq.data_ptr() if xq is not None else 0, xs.data_ptr() if xs is not None else 0, float(eps), end,
        torch.cuda.current_stream().cuda_stream)
    if r != 0:
        raise RuntimeError(f"welm_arn launch failed: {r}")


_logged = set()


def _once(msg):
    if msg not in _logged:
        _logged.add(msg)
        logger.info("WELM_ARN: %s", msg)


def supported(x, res, *gammas):
    ok = _supported(x, res, *gammas)
    if not ok:
        _once(f"unsupported -> AR fallback: x {x.dtype} {tuple(x.shape)} res {res.dtype} gammas {[g.dtype for g in gammas]}")
    return ok


def _supported(x, res, *gammas):
    return (x.dim() == 2 and x.dtype == torch.bfloat16 and x.is_contiguous() and x.shape[1] % 8 == 0
            and x.shape[1] <= 2048 and res.dtype == torch.float32 and res.is_contiguous()
            and res.shape == x.shape and all(g.dtype == torch.bfloat16 and g.is_contiguous() for g in gammas))


def ar_norm_after_attn(x, residual, onorm_w, rnorm_w, eps, quant_out=None):
    """Fused AR + mmq_style_norm_after_attn; x is the local pool view. Returns (out, res_out, fp32_out)."""
    out = torch.empty_like(x)
    res_out = torch.empty_like(x, dtype=torch.float32)
    f32 = torch.empty_like(x, dtype=torch.float32)
    xq, xs = quant_out if quant_out is not None else (None, None)
    _once("fused AR+post-attn norm in use")
    _launch(0, x, onorm_w, rnorm_w, residual, out, res_out, f32, xq, xs, eps)
    return out, res_out, f32


def ar_input_norm(x, residual, w, eps):
    """Fused AR + ppln input_layernorm(clone_fp32_out). Returns (out, out, fp32_out)."""
    out = torch.empty_like(x)
    f32 = torch.empty_like(x, dtype=torch.float32)
    _once("fused AR+input norm in use")
    _launch(1, x, w, None, residual, out, None, f32, None, None, eps)
    return out, out, f32
