"""WELM_AR_POOL: let the producer of a TP all-reduce input write straight into the aiter
custom-AR pre-registered input pool, so the AR skips its copy-in (a ~5 us DtoD memcpy node
per AR in decode graphs on gfx938, where IPC graph-buffer registration is disabled).

Only used while capturing a CUDA graph: the producer gets a view of the pool via view()
(o_proj) or arm()/take() (MoE moe_sum), and the aiter wrapper recognises a pool input by
pointer and calls the AR with reg_inp = 0 (the pool is IPC-registered at init).
"""
import os

import torch

ENABLED = os.environ.get("WELM_AR_POOL", "0") == "1"
_armed = False
_arm_off = 0
_logged = set()


class _CAI:
    def __init__(self, ptr, n):
        self.__cuda_array_interface__ = {"shape": (n,), "typestr": "<u2", "data": (ptr, False), "version": 3}


def _ca():
    from sglang.srt.distributed import get_tp_group

    ca = getattr(get_tp_group(), "ca_comm", None)
    if ca is None or getattr(ca, "disabled", True) or not hasattr(ca, "_pool"):
        return None
    return ca


def _log(what):
    if what not in _logged:
        _logged.add(what)
        import logging

        logging.getLogger(__name__).info("WELM_AR_POOL: %s writes into the AR input pool", what)


def view(shape, dtype, device, what="o_proj", off=0):
    """A (shape, dtype) view of the local AR input pool at byte offset off, or None when not applicable."""
    if not ENABLED or dtype != torch.bfloat16 or not torch.cuda.is_current_stream_capturing():
        return None
    ca = _ca()
    if ca is None:
        return None
    n = 1
    for s in shape:
        n *= s
    nb = n * 2
    if nb == 0 or nb % 16 or off % 16 or nb > min(ca.max_size // 2, ca._all_reduce_max_size, ca._pool["input"].max_size - off):
        return None
    if not (ca.world_size == 2 or ca.fully_connected):
        return None
    buf = ca.__dict__.get("_welm_pool_tensor")
    if buf is None:
        buf = torch.as_tensor(_CAI(ca._pool["input"].data_ptr, ca._pool["input"].max_size // 2), device=device)
        ca._welm_pool_tensor = buf
    _log(what)
    return buf.view(torch.bfloat16)[off // 2 : off // 2 + n].view(*shape)


def arm(off=0):
    global _armed, _arm_off
    _armed = ENABLED
    _arm_off = off


def take(shape, dtype, device, what="moe_sum"):
    global _armed
    if not _armed:
        return None
    _armed = False
    return view(shape, dtype, device, what, _arm_off)


def disarm():
    global _armed
    _armed = False


def is_pool(ca, inp):
    return ENABLED and inp.data_ptr() == ca._pool["input"].data_ptr


def is_pool_tensor(t, off=0):
    ca = _ca() if ENABLED else None
    return ca is not None and t.data_ptr() == ca._pool["input"].data_ptr + off
