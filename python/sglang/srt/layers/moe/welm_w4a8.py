"""WELM_MOE_W4A8: requantize the channel-fp8 routed experts to int4 per-channel at load time (in memory,
the checkpoint is untouched) and run them with aiter moe_c W4A8 (int8 per-token activations).

Packing matches the slimquant W4A8 checkpoint: int8 bytes (even_k << 4) | (odd_k & 0xF), two's-complement
nibbles, then aiter w4a8_moe_repack_shuffle; the scale is passed as scale / 16 (moe_c multiplies by 16).
Per-channel scale picks the clip ratio minimising the weight MSE (fp8 dequant -> int4 RTN).
moe_c W4A8 vs lightop marlin fp8 (E512 H2048 I256 top10, gfx938): M32 154 vs 238 us, M8192 2.05 vs 3.05 ms.
"""
import logging
import os

import torch

logger = logging.getLogger(__name__)
ENABLED = os.environ.get("WELM_MOE_W4A8", "0") == "1"
_RATIOS = [float(r) for r in os.environ.get("WELM_W4A8_CLIP", "1.0,0.95,0.9,0.85,0.8,0.75,0.7,0.65,0.6").split(",")]
_CHUNK = 32


def _q4(w):
    am = w.abs().amax(-1, keepdim=True).clamp(min=1e-12)
    best = bs = None
    for r in _RATIOS:
        s = am * (r / 7.0)
        e = ((torch.round(w / s).clamp(-8, 7) * s - w) ** 2).sum(-1, keepdim=True)
        if best is None:
            best, bs = e, s
        else:
            m = e < best
            best = torch.where(m, e, best)
            bs = torch.where(m, s, bs)
    q = torch.round(w / bs).clamp(-8, 7).to(torch.int8)
    return (((q[..., 0::2] & 0xF) << 4) | (q[..., 1::2] & 0xF)).to(torch.uint8).view(torch.int8), bs


@torch.no_grad()
def requant(w, scale):
    """w: fp8 [E, N, K], scale: per-channel [E, N] / [E, N, 1]. Returns (shuffled int8 [E, N, K/2], fp32 [E, N, 1] scale/16)."""
    from aiter.ops.shuffle import w4a8_moe_repack_shuffle

    E, N, K = w.shape
    scale = scale.reshape(E, N, 1).float()
    packed = torch.empty(E, N, K // 2, dtype=torch.int8, device=w.device)
    s_out = torch.empty(E, N, 1, dtype=torch.float32, device=w.device)
    for e0 in range(0, E, _CHUNK):
        e1 = min(E, e0 + _CHUNK)
        p, s = _q4(w[e0:e1].float() * scale[e0:e1])
        packed[e0:e1] = p
        s_out[e0:e1] = s / 16.0
    return w4a8_moe_repack_shuffle(packed.contiguous()), s_out


_cfg = {}


def _config(m, e, n1, n2, k, topk, dtype):
    key = (m, e, n1, n2, k, topk, dtype)
    c = _cfg.get(key)
    if c is None:
        from aiter.moe import MoeQuantType, get_aiter_moe_config

        st, c = get_aiter_moe_config(M=m, E=e, N1=n1, N2=n2, K=k, top_k=topk, block_size=None, dtype=dtype,
                                     quant_type=MoeQuantType.W4A8)
        if not st:
            raise RuntimeError(f"WELM_MOE_W4A8: no aiter W4A8 config for M={m} E={e} N1={n1} N2={n2} K={k}")
        _cfg[key] = c
        if len(_cfg) == 1:
            logger.info("WELM_MOE_W4A8: aiter %s", c.solution_type)
    return c


def run(layer, x, topk_weights, topk_ids, bias=None, routed_scaling_factor=None, num_experts=None):
    from aiter.moe import aiter_moe

    from sglang.srt.layers import welm_ar_pool as _arp

    e = layer.w13_weight.shape[0]
    cfg = _config(x.shape[0], e, layer.w13_weight.shape[1], layer.w2_weight.shape[1], x.shape[1],
                  topk_ids.shape[1], x.dtype)
    o = aiter_moe(x, w1=layer.w13_weight, w2=layer.w2_weight, topk_weights=topk_weights.float(),
                  topk_ids=topk_ids.to(torch.int32), moe_config=cfg, activation="silu",
                  w1_scale=layer.w13_weight_scale, w2_scale=layer.w2_weight_scale,
                  global_num_experts=num_experts or e,
                  routed_scaling_factor=1.0 if routed_scaling_factor is None else routed_scaling_factor)
    out = _arp.take(x.shape, x.dtype, x.device) if x.dim() == 2 else None
    if out is not None:
        if bias is not None:
            torch.add(o, bias, out=out)
        else:
            out.copy_(o)
        return out
    return o + bias if bias is not None else o
