"""Route the bf16 MoE through aiter W16A16 kernels instead of the serving Triton kernel,
per token-count M.  Everything not routed stays on Triton.

Backends (single-card A/B on the served shape E=513 topk=11 K=2048 I=128, 11-19, us):
      M        8     32     64    128   2018   4003   9603  21239
  triton    110*   276*   378*   588*   1085   1531   2860   5740
  asm_mc     124*   286*   405*   520*    908   1243   2552   4794
  moe_c      ~=triton under graph             944   1441   3117   6463
  (* = CUDA-graph replay; the rest eager.)

So: moe_c wins small/medium M, ASM wins large M, Triton wins decode M<=64 under graph.
ASM needs canonical [gate;up] w13 -- a row permutation of the served interleaved layout --
and reuses w2 UNCHANGED, so it costs one w13 copy (~18 GB/card).  moe_c needs both w13 and
w2 shuffled (~36 GB/card).

ASM uses aiter's module-level moe_activation, which defaults to a generic boltops Triton
elementwise kernel.  Swapping it for aiter.moe_c_silu_and_mul is worth -16% at M=21239
(5640 -> 4754 us) AND moves the output closer to the serving Triton kernel
(rel 2.0e-4 vs 4.3e-3), because it reproduces Triton's bf16 rounding order: silu(gate) is
rounded to bf16 before the multiply.

Env:
  WELM_MOE_AITER_ASM_MIN_M  M > this -> aiter ASM (builds the canonical w13 copy).  0 = off.
  WELM_MOE_AITER_MAX_M      MIN_M < M <= this -> aiter moe_c (builds the shuffled copy).
  WELM_MOE_AITER_MIN_M      lower bound for the moe_c window (default 128).
  WELM_MOE_AITER_INPLACE    shuffle weights in place for moe_c, route EVERY call to moe_c.
  WELM_MOE_AITER_PREP       build the copy even when nothing is routed (memory-matched control).
  WELM_MOE_AITER_CFG        overlay root holding moe_c/<arch>/w16a16/*.json and asm/tuned_fmoe_asm.csv
"""
import inspect
import logging
import os

import torch

logger = logging.getLogger(__name__)
INPLACE = os.environ.get("WELM_MOE_AITER_INPLACE", "0") == "1"
MAX_M = int(os.environ.get("WELM_MOE_AITER_MAX_M", "0"))
MIN_M = int(os.environ.get("WELM_MOE_AITER_MIN_M", "128"))
ASM_MIN_M = int(os.environ.get("WELM_MOE_AITER_ASM_MIN_M", "0"))
ASM_INPLACE = os.environ.get("WELM_MOE_AITER_ASM_INPLACE", "0") == "1"
PREP = (INPLACE or ASM_INPLACE or MAX_M > 0 or ASM_MIN_M > 0
        or os.environ.get("WELM_MOE_AITER_PREP", "0") == "1")
_CFG_ROOT = os.environ.get("WELM_MOE_AITER_CFG", "")
_cfg = {}
_asm_cfg = {}
_moec = None
_asm_ready = False


def _setup():
    global _moec
    if _moec is not None:
        return
    import aiter.fused_moe_c as mc
    if _CFG_ROOT:
        mc._moe_c_config_root = lambda: os.path.join(_CFG_ROOT, "moe_c")
    # moe_c_fused_experts is functools.lru_cache'd on tensor args; bypass it.
    _moec = inspect.unwrap(mc.moe_c_fused_experts)


def _setup_asm():
    """Point aiter at the tuned CSV and install the faster, Triton-matching activation."""
    global _asm_ready
    if _asm_ready:
        return
    import aiter
    import aiter.fused_moe_asm_wna16 as a
    if _CFG_ROOT:
        _orig_csv = a.get_csv_path
        csv = os.path.join(_CFG_ROOT, "asm", "tuned_fmoe_asm.csv")
        if os.path.exists(csv):
            a.get_csv_path = lambda q: csv if q == "no_quant" else _orig_csv(q)
    _orig_act = a.moe_activation

    def _act(activated_out, ffn1_out_2d, activation="silu", is_gated=None, **kw):
        # only the gated-silu case matches moe_c_silu_and_mul; anything else -> aiter default
        if activation == "silu" and is_gated is not False and not kw.get("gemm1_alpha") \
                and not kw.get("gemm1_limit"):
            aiter.moe_c_silu_and_mul(activated_out, ffn1_out_2d)
            return
        _orig_act(activated_out, ffn1_out_2d, activation, is_gated, **kw)

    a.moe_activation = _act
    _asm_ready = True
    logger.info("WELM_MOE_AITER: aiter ASM activation -> moe_c_silu_and_mul")


def _config(m, e, n1, k, topk):
    key = (m, e, n1, k, topk)
    c = _cfg.get(key)
    if c is None:
        _setup()
        from aiter.moe import MoeQuantType, MoeSolutionType, get_aiter_moe_config
        st, c = get_aiter_moe_config(M=m, E=e, N1=n1, N2=k, K=k, top_k=topk, block_size=0,
                                     dtype=torch.bfloat16, quant_type=MoeQuantType.W16A16,
                                     spec_sol_type=MoeSolutionType.MOE_C)
        if not st:
            raise RuntimeError(f"WELM_MOE_AITER: no moe_c W16A16 config E={e} N1={n1} K={k}")
        _cfg[key] = c
    return c


def _config_asm(m, e, n1, k, topk):
    key = (m, e, n1, k, topk)
    c = _asm_cfg.get(key)
    if c is None:
        _setup_asm()
        from aiter.moe import MoeQuantType, MoeSolutionType, get_aiter_moe_config
        st, c = get_aiter_moe_config(M=m, E=e, N1=n1, N2=k, K=k, top_k=topk, block_size=0,
                                     dtype=torch.bfloat16, quant_type=MoeQuantType.W16A16,
                                     spec_sol_type=MoeSolutionType.ASM)
        if not st:
            raise RuntimeError(f"WELM_MOE_AITER: no ASM W16A16 config E={e} N1={n1} K={k} M={m}")
        _asm_cfg[key] = c
    return c


@torch.no_grad()
def prepare(layer, interleaved):
    """Build the aiter-layout copies of the routed+shared expert weights."""
    if not PREP:
        return
    need_moec = (INPLACE or MAX_M > 0 or os.environ.get("WELM_MOE_AITER_PREP", "0") == "1")
    done_moec = getattr(layer, "_welm_moec_w13", None) is not None
    done_asm = getattr(layer, "_welm_asm_w13", None) is not None
    need_asm = ASM_MIN_M > 0 or ASM_INPLACE
    if (done_moec or not need_moec) and (done_asm or not need_asm):
        return
    from aiter.moe import aiter_moe_shfl_weight
    w13, w2 = layer.w13_weight.data, layer.w2_weight.data
    e, n1, k = w13.shape
    if interleaved:  # undo the serving row interleave: even rows = gate, odd rows = up
        w13 = torch.cat([w13[:, 0::2], w13[:, 1::2]], dim=1).contiguous()

    if (ASM_MIN_M > 0 or ASM_INPLACE) and not done_asm:
        # ASM wants canonical [gate;up] w13 and the UNMODIFIED w2 -- no w2 copy needed.
        _setup_asm()
        w13 = w13.contiguous()
        if ASM_INPLACE:
            # Replace the serving (interleaved) w13 outright: zero extra memory, but the
            # Triton kernel can never read these weights again, so use() must always route.
            assert w13.shape == layer.w13_weight.shape, (w13.shape, layer.w13_weight.shape)
            layer.w13_weight.data = w13
            layer._welm_asm_inplace = True
            w13 = layer.w13_weight.data
        layer._welm_asm_w13 = w13

    if need_moec and not done_moec:
        cfg = _config(256, e, n1, k, 11)
        a, b = aiter_moe_shfl_weight(w13, w2, cfg)
        a, b = a.contiguous(), b.contiguous()
        if INPLACE:
            assert a.shape == layer.w13_weight.shape and b.shape == layer.w2_weight.shape, (
                a.shape, layer.w13_weight.shape, b.shape, layer.w2_weight.shape)
            layer.w13_weight.data = a  # drops the Triton-layout storage
            layer.w2_weight.data = b
            layer._welm_moec_inplace = True
            a, b = layer.w13_weight.data, layer.w2_weight.data
        layer._welm_moec_w13, layer._welm_moec_w2 = a, b
    del w13
    if not getattr(prepare, "_logged", False):
        prepare._logged = True
        logger.info("WELM_MOE_AITER: E=%d N1=%d K=%d | moe_c=%s asm_min_m=%d max_m=%d min_m=%d",
                    e, n1, k,
                    "inplace" if INPLACE else ("copy" if need_moec else "off"),
                    ASM_MIN_M, MAX_M, MIN_M)


def use(layer, x):
    if getattr(layer, "_welm_moec_inplace", False):
        return True  # weights are in moe_c layout: Triton must never see them
    if getattr(layer, "_welm_asm_inplace", False):
        return True  # weights are canonical [gate;up]: the interleaved Triton path is gone
    if x.dim() != 2 or x.dtype != torch.bfloat16:
        return False
    m = x.shape[0]
    if (ASM_MIN_M > 0 and m > ASM_MIN_M
            and getattr(layer, "_welm_asm_w13", None) is not None
            and not torch.cuda.is_current_stream_capturing()):
        return True
    return (MAX_M > 0 and MIN_M < m <= MAX_M
            and getattr(layer, "_welm_moec_w13", None) is not None
            and not torch.cuda.is_current_stream_capturing())


def _run_asm(layer, x, topk_weights, topk_ids, routed_scaling_factor):
    from aiter.moe import aiter_moe
    w13 = layer._welm_asm_w13
    e, n1, k = w13.shape
    topk = topk_ids.shape[1]
    cfg = _config_asm(x.shape[0], e, n1, k, topk)
    return aiter_moe(x.contiguous(), w13, layer.w2_weight.data,
                     topk_weights.float().contiguous(), topk_ids.to(torch.int32).contiguous(),
                     moe_config=cfg, activation="silu", global_num_experts=e,
                     routed_scaling_factor=1.0 if routed_scaling_factor is None
                     else routed_scaling_factor)


def run(layer, x, topk_weights, topk_ids, routed_scaling_factor):
    from sglang.srt.layers import welm_ar_pool as _arp
    if x.shape[0] == 0:
        return x.new_empty(x.shape)
    assert x.dim() == 2 and x.dtype == torch.bfloat16, (x.shape, x.dtype)

    if (getattr(layer, "_welm_asm_inplace", False)
            or (ASM_MIN_M > 0 and x.shape[0] > ASM_MIN_M
                and getattr(layer, "_welm_asm_w13", None) is not None)):
        o = _run_asm(layer, x, topk_weights, topk_ids, routed_scaling_factor)
    else:
        w13, w2 = layer._welm_moec_w13, layer._welm_moec_w2
        e, n1, k = w13.shape
        c = _config(x.shape[0], e, n1, k, topk_ids.shape[1]).config
        g1, g2 = c["GEMM1_CONFIG"], c["GEMM2_CONFIG"]
        bm1, bm2 = int(g1["BLOCK_SIZE_M"]), int(g2["BLOCK_SIZE_M"])
        w16 = [int(g1["MODE"]), bm1, int(g1.get("DELTA", 1)), int(g2["MODE"]), bm2,
               int(g2.get("DELTA", bm1 // bm2))]
        o = _moec(x.contiguous(), w13, w2, topk_weights.float().contiguous(),
                  topk_ids.to(torch.int32).contiguous(), inplace=False, use_w16a16=True,
                  activation="silu", global_num_experts=e,
                  routed_scaling_factor=(1.0 if routed_scaling_factor is None
                                         else routed_scaling_factor),
                  compute_dtype=x.dtype, w16a16_config=w16)
    out = _arp.take(x.shape, x.dtype, x.device)
    if out is not None:
        out.copy_(o)
        return out
    return o
