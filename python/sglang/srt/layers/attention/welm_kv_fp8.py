"""fp8_e4m3 KV cache support for WeLM on HCU (legacy paged layout).

The KV pool stores e4m3 values x8 with real = x8 * scale, one static scale per
(layer, K/V). The checkpoint carries no KV scales (kv_cache_scheme: None), so
scales come from, in order:
  WELM_KV_FP8_SCALES  json file {"k": [per layer], "v": [per layer]} or
                      {"<layer_id>": [k, v], ...}
  WELM_KV_FP8_K_SCALE / WELM_KV_FP8_V_SCALE  uniform floats (default 1.0)
Attention kernels look scales up by the KV buffer's data_ptr (registered by
the memory pool), so no extra arguments travel through the attention stack.

WELM_KV_AMAX_DUMP=<path>: record per-layer |K|/|V| amax at KV-store time
(bf16 or fp8 pools, eager forwards only) and write it as json every 200 stores.
"""
import json
import os

import torch

FP8_MAX = 448.0
_by_layer = {}   # layer_id -> (k_scale, v_scale)
_by_ptr = {}     # k/v buffer data_ptr -> (k_scale, v_scale)
_loaded = [False]


def _load():
    if _loaded[0]:
        return
    _loaded[0] = True
    path = os.environ.get("WELM_KV_FP8_SCALES")
    if not path:
        return
    with open(path) as f:
        d = json.load(f)
    if "k" in d and "v" in d:
        for i, (ks, vs) in enumerate(zip(d["k"], d["v"])):
            _by_layer[i] = (float(ks), float(vs))
    else:
        for i, kv in d.items():
            _by_layer[int(i)] = (float(kv[0]), float(kv[1]))


def layer_scales(layer_id):
    _load()
    s = _by_layer.get(layer_id)
    if s is None:
        s = (
            float(os.environ.get("WELM_KV_FP8_K_SCALE", "1.0")),
            float(os.environ.get("WELM_KV_FP8_V_SCALE", "1.0")),
        )
    return s


def register_buffer(layer_id, k_buf, v_buf):
    s = layer_scales(layer_id)
    _by_ptr[k_buf.data_ptr()] = s
    _by_ptr[v_buf.data_ptr()] = s
    return s


def is_fp8(t):
    return t.dtype in (torch.float8_e4m3fn, torch.uint8)


def as_fp8(t):
    return t.view(torch.float8_e4m3fn) if t.dtype == torch.uint8 else t


def scales_for(k_cache):
    """(k_scale, v_scale) for an fp8 KV buffer (or a view sharing its storage)."""
    s = _by_ptr.get(k_cache.data_ptr())
    if s is None:
        raise RuntimeError(
            "fp8 KV buffer without registered scales (data_ptr %#x)" % k_cache.data_ptr()
        )
    return s


# ---------------------------------------------------------------- amax dump --
_AMAX_PATH = os.environ.get("WELM_KV_AMAX_DUMP")
_amax = {}
_amax_n = [0]


def record_amax(layer_id, cache_k, cache_v):
    if not _AMAX_PATH or torch.cuda.is_current_stream_capturing():
        return
    if cache_k.dtype not in (torch.bfloat16, torch.float16, torch.float32):
        return
    a = torch.stack([cache_k.abs().amax().float(), cache_v.abs().amax().float()])
    # 99.9th percentile of |x| as a robustness reference (sampled)
    flat_k = cache_k.reshape(-1)[:: max(1, cache_k.numel() // 65536)].abs().float()
    flat_v = cache_v.reshape(-1)[:: max(1, cache_v.numel() // 65536)].abs().float()
    q = torch.stack([torch.quantile(flat_k, 0.999), torch.quantile(flat_v, 0.999)])
    prev = _amax.get(layer_id)
    _amax[layer_id] = (a, q) if prev is None else (torch.maximum(prev[0], a), torch.maximum(prev[1], q))
    _amax_n[0] += 1
    if _amax_n[0] % 200 == 0:
        out = {str(l): {"amax": v[0].tolist(), "p999": v[1].tolist()} for l, v in sorted(_amax.items())}
        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        with open(f"{_AMAX_PATH}.rank{rank}.json", "w") as f:
            json.dump(out, f, indent=0)


# ------------------------------------------------------------------ tracing --
_TRACE = bool(os.environ.get("WELM_KV_FP8_TRACE"))
_seen = set()


def trace(tag, *info):
    """WELM_KV_FP8_TRACE=1: log each distinct code path once (eager only)."""
    if not _TRACE or tag in _seen:
        return
    _seen.add(tag)
    print(f"[welm_kv_fp8] {tag} {info}", flush=True)


# ------------------------------------------------------- in-situ checking --
CHECK = bool(os.environ.get("WELM_KV_FP8_CHECK"))
_nchk = [0]
_ntag = {}


def compact_dequant(k_cache, v_cache, block_table):
    """bf16 dequantized copy of the pages block_table references, plus the
    remapped block table (for re-running a kernel on bf16 KV)."""
    ks, vs = scales_for(k_cache)
    bt = block_table.long().clamp(0, k_cache.shape[0] - 1)
    kc = (k_cache[bt.flatten()].float() * ks).bfloat16()
    vc = (v_cache[bt.flatten()].float() * vs).bfloat16()
    nbt = torch.arange(bt.numel(), device=bt.device, dtype=block_table.dtype).view(bt.shape)
    return kc, vc, nbt


def report(tag, out, ref):
    key = (tag, tuple(out.shape))
    n = _ntag.get(key, 0)
    if n >= 2 or _nchk[0] >= 3000:
        return
    _ntag[key] = n + 1
    _nchk[0] += 1
    o, r = out.float(), ref.float()
    print(f"[welm_kv_fp8 check] {tag} shape {tuple(o.shape)} rel {((o - r).norm() / r.norm().clamp_min(1e-20)).item():.4f} "
          f"|o| {o.abs().max().item():.3g} |ref| {r.abs().max().item():.3g} nan {torch.isnan(o).any().item()}", flush=True)
