# Copyright (c) 2023, Tri Dao.

from typing import Optional, Union
from typing import List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from flash_attn.utils.sparse_utils import (get_block_map_meansim, block_map_to_block_offset_triton, get_block_map, get_block_map_fast)
import flash_attn_2_cuda as flash_attn_cuda

DEFAULT_FA_VERSION = 2

def maybe_contiguous(x):
    return x.contiguous() if x is not None and x.stride(-1) != 1 else x

def round_multiple(x, m):
    return (x + m - 1) // m * m

def _get_block_size_n(device, head_dim, is_dropout, is_causal):
    # This should match the block sizes in the CUDA kernel
    assert head_dim <= 512
    major, minor = torch.cuda.get_device_capability(device)
    is_sm8x = major == 8 and minor > 0  # Only include sm86 and sm89, exclude sm80 (A100)
    is_sm80 = major == 8 and minor == 0
    is_sm90 = major == 9 and minor == 0
    if head_dim <= 32:
        return 128
    if head_dim <= 64:
        return 128 if not is_dropout else 64
    elif head_dim <= 96:
        return 64
    elif head_dim <= 128:
        if is_sm8x:
            return 64 if (not is_dropout and is_causal) else 32
        else:
            return 64 if not is_dropout else 32
    elif head_dim <= 160:
        if is_sm8x:
            return 64
        else:
            return 32
    elif head_dim <= 192:
        return 64
    elif head_dim <= 224:
        return 64
    elif head_dim <= 256:
        return 64
    elif head_dim <= 512:
        return 64


if torch.__version__ >= "2.4.0":
    _torch_custom_op_wrapper = torch.library.custom_op
    _torch_register_fake_wrapper = torch.library.register_fake
else:
    def noop_custom_op_wrapper(name, fn=None, /, *, mutates_args, device_types=None, schema=None):
        def wrap(func):
            return func
        if fn is None:
            return wrap
        return fn
    def noop_register_fake_wrapper(op, fn=None, /, *, lib=None, _stacklevel=1):
        def wrap(func):
            return func
        if fn is None:
            return wrap
        return fn
    _torch_custom_op_wrapper = noop_custom_op_wrapper
    _torch_register_fake_wrapper = noop_register_fake_wrapper


@_torch_custom_op_wrapper("flash_attn::_flash_attn_forward", mutates_args=(), device_types="cuda")
def _flash_attn_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    dropout_p: float,
    softmax_scale: float,
    causal: bool,
    window_size_left: int,
    window_size_right: int,
    softcap: float,
    alibi_slopes: Optional[torch.Tensor],
    return_softmax: bool,
    bhsd:bool = False,
    s_aux :Optional[torch.Tensor] = None,
) -> Tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    q, k, v = [maybe_contiguous(x) for x in (q, k, v)]
    out, q, k, v, out_padded, softmax_lse, S_dmask, rng_state, skip_info = flash_attn_cuda.fwd(
        q,
        k,
        v,
        None,
        alibi_slopes,
        dropout_p,
        softmax_scale,
        causal,
        window_size_left,
        window_size_right,
        softcap,
        return_softmax,
        None, bhsd, s_aux, 0.0 
    )
    return out, softmax_lse, S_dmask, rng_state


@_torch_register_fake_wrapper("flash_attn::_flash_attn_forward")
def _flash_attn_forward_fake(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    dropout_p: float,
    softmax_scale: float,
    causal: bool,
    window_size_left: int,
    window_size_right: int,
    softcap: float,
    alibi_slopes: Optional[torch.Tensor],
    return_softmax: bool,
    bhsd:bool = False,
    s_aux :Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    q, k, v = [maybe_contiguous(x) for x in (q, k, v)]
    batch_size, seqlen_q, num_heads, head_size = q.shape
    seqlen_k = k.shape[1]
    out = torch.empty_like(q)
    softmax_lse = torch.empty((batch_size, num_heads, seqlen_q), dtype=torch.float32, device=q.device, layout=q.layout)
    p = torch.empty((0,), dtype=q.dtype, device=q.device, layout=q.layout)
    if return_softmax:
        p = torch.empty((batch_size, num_heads, round_multiple(seqlen_q, 128), round_multiple(seqlen_k, 128)), dtype=q.dtype, device=q.device, layout=q.layout)
    rng_state = torch.empty((2,), dtype=torch.int64, device=q.device)
    return out, softmax_lse, p, rng_state


if torch.__version__ >= "2.4.0":
    _wrapped_flash_attn_forward = torch.ops.flash_attn._flash_attn_forward
else:
    _wrapped_flash_attn_forward = _flash_attn_forward


def _flash_attn_varlen_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q:float,
    max_seqlen_k:float,
    dropout_p:float,
    softmax_scale:float,
    causal:bool,
    window_size_left:int,
    window_size_right:int,
    softcap:float,
    alibi_slopes:Optional[torch.Tensor],
    return_softmax:bool,
    q_descale:Optional[torch.Tensor] = None,
    k_descale:Optional[torch.Tensor] = None,
    v_descale:Optional[torch.Tensor] = None,
    block_table:Optional[torch.Tensor]=None,
    leftpad_k:Optional[torch.Tensor]=None,
    s_aux:Optional[torch.Tensor]=None,
)-> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    q, k, v = [maybe_contiguous(x) for x in (q, k, v)]
    out, q, k, v, out_padded, softmax_lse, S_dmask, rng_state = flash_attn_cuda.varlen_fwd(
        q,
        k,
        v,
        None,
        cu_seqlens_q,
        cu_seqlens_k,
        None,
        leftpad_k,
        block_table,
        alibi_slopes,
        max_seqlen_q,
        max_seqlen_k,
        dropout_p,
        softmax_scale,
        False,
        causal,
        window_size_left,
        window_size_right,
        softcap,
        return_softmax,
        q_descale,
        k_descale,
        v_descale,
        None,
        s_aux,
    )
    # if out.isnan().any() or softmax_lse.isnan().any():
    #     breakpoint()
    return out, softmax_lse, S_dmask, rng_state

@_torch_register_fake_wrapper("flash_attn2_c_op::varlen_fwd")
def varlen_fwd_fake(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        out: Optional[torch.Tensor],
        cu_seqlens_q: torch.Tensor,
        cu_seqlens_k: torch.Tensor,
        seqused_k: Optional[torch.Tensor],
        leftpad_k_: Optional[torch.Tensor],
        block_table: Optional[torch.Tensor],
        alibi_slopes: Optional[torch.Tensor],
        max_seqlen_q: int,
        max_seqlen_k: int,
        p_dropout: float,
        softmax_scale: float,
        zero_tensors: bool,
        is_causal: bool,
        window_size_left: int,
        window_size_right: int,
        softcap: float,
        return_softmax: bool,) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    # q shape:total_q x num_heads x head_size
    num_heads = q.size(1)
    total_q = q.size(0)
    batch_size = cu_seqlens_q.size(0) - 1
    out = torch.empty_like(q)
    softmax_lse = torch.empty([num_heads, total_q], dtype=q.dtype, device=q.device)
    seqlen_q_rounded = round_multiple(max_seqlen_q, 128)
    seqlen_k_rounded = round_multiple(max_seqlen_k, 128)
    if return_softmax:
        p = torch.empty([batch_size, num_heads, seqlen_q_rounded, seqlen_k_rounded], dtype=q.dtype, device=q.device)
    else:
        p = torch.empty([0], dtype=q.dtype, device=q.device)
    rng_state = torch.empty([2], dtype=torch.int64, device=q.device)
    return out, softmax_lse, p, rng_state


_wrapped_flash_attn_varlen_forward = torch.ops.flash_attn2_c_op.varlen_fwd

@_torch_register_fake_wrapper("flash_attn2_c_op::fwd_kvcache")
def fwd_kvcache_fake(
    q: torch.Tensor,
    kcache: torch.Tensor,
    vcache: torch.Tensor,
    k_: Optional[torch.Tensor],
    v_: Optional[torch.Tensor],
    seqlens_k_: Optional[torch.Tensor],
    rotary_cos_: Optional[torch.Tensor],
    rotary_sin_: Optional[torch.Tensor],
    cache_batch_idx_: Optional[torch.Tensor],
    leftpad_k_: Optional[torch.Tensor],
    block_table_: Optional[torch.Tensor],
    alibi_slopes_: Optional[torch.Tensor],
    out_: Optional[torch.Tensor],
    softmax_scale: float,
    is_causal: bool,
    window_size_left: int,
    window_size_right: int,
    softcap: float,
    is_rotary_interleaved: bool,
    num_splits: int,
) -> Tuple[torch.Tensor, torch.Tensor]:

    batch_size, seqlen_q, num_heads, head_size = q.shape
    
    # 创建输出张量的形状
    # out: batch_size x seqlen_q x num_heads x head_size
    out_shape = q.shape
    out = torch.empty(out_shape, dtype=q.dtype, device=q.device)
    
    # softmax_lse: batch_size x num_heads x seqlen_q
    softmax_lse_shape = (batch_size, num_heads, seqlen_q)
    softmax_lse = torch.empty(softmax_lse_shape, dtype=torch.float32, device=q.device)
    
    return out, softmax_lse


@_torch_custom_op_wrapper("flash_attn::_flash_attn_backward", mutates_args=("dq", "dk", "dv"), device_types="cuda")
def _flash_attn_backward(
    dout: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    softmax_lse: torch.Tensor,
    dq: Optional[torch.Tensor],
    dk: Optional[torch.Tensor],
    dv: Optional[torch.Tensor],
    dropout_p: float,
    softmax_scale: float,
    causal: bool,
    window_size_left: int,
    window_size_right: int,
    softcap: float,
    alibi_slopes: Optional[torch.Tensor],
    deterministic: bool,
    rng_state: Optional[torch.Tensor] = None,
    bhsd:bool = False,
) -> torch.Tensor:
    dout, q, k, v, out = [maybe_contiguous(x) for x in (dout, q, k, v, out)]
    return flash_attn_cuda.bwd(
        dout,
        q,
        k,
        v,
        out,
        softmax_lse,
        dq,
        dk,
        dv,
        alibi_slopes,
        dropout_p,
        softmax_scale,
        causal,
        window_size_left,
        window_size_right,
        softcap,
        deterministic,
        None,
        rng_state,
        bhsd
    )[3]


@_torch_register_fake_wrapper("flash_attn::_flash_attn_backward")
def _flash_attn_backward_fake(
    dout: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    softmax_lse: torch.Tensor,
    dq: Optional[torch.Tensor],
    dk: Optional[torch.Tensor],
    dv: Optional[torch.Tensor],
    dropout_p: float,
    softmax_scale: float,
    causal: bool,
    window_size_left: int,
    window_size_right: int,
    softcap: float,
    alibi_slopes: Optional[torch.Tensor],
    deterministic: bool,
    rng_state: Optional[torch.Tensor] = None,
    bhsd:bool = False,
) -> torch.Tensor:
    dout, q, k, v, out = [maybe_contiguous(x) for x in (dout, q, k, v, out)]
    if dq is None:
        dq = torch.empty_like(q)
    if dk is None:
        dk = torch.empty_like(k)
    if dv is None:
        dv = torch.empty_like(v)
    batch_size, seqlen_q, num_heads, _ = q.shape
    softmax_d = torch.empty((batch_size, num_heads, round_multiple(seqlen_q, 128)), device=q.device, dtype=torch.float32)
    return softmax_d


if torch.__version__ >= "2.4.0":
    _wrapped_flash_attn_backward = torch.ops.flash_attn._flash_attn_backward
else:
    _wrapped_flash_attn_backward = _flash_attn_backward


@_torch_custom_op_wrapper("flash_attn::_flash_attn_varlen_backward",mutates_args=("dq", "dk", "dv"),device_types="cuda",)
def _flash_attn_varlen_backward(
    dout: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    softmax_lse: torch.Tensor,
    dq: Optional[torch.Tensor],
    dk: Optional[torch.Tensor],
    dv: Optional[torch.Tensor],
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    dropout_p: float,
    softmax_scale: float,
    causal: bool,
    window_size_left: int,
    window_size_right: int,
    softcap: float,
    alibi_slopes: Optional[torch.Tensor],
    deterministic: bool,
    rng_state: Optional[torch.Tensor] = None,
    zero_tensors: bool = False,
) -> torch.Tensor:
    dout, q, k, v, out = [maybe_contiguous(x) for x in (dout, q, k, v, out)]
    (
        dq,
        dk,
        dv,
        softmax_d,
    ) = flash_attn_cuda.varlen_bwd(
        dout,
        q,
        k,
        v,
        out,
        softmax_lse,
        dq,
        dk,
        dv,
        cu_seqlens_q,
        cu_seqlens_k,
        alibi_slopes,
        max_seqlen_q,
        max_seqlen_k,
        dropout_p,
        softmax_scale,
        zero_tensors,
        causal,
        window_size_left,
        window_size_right,
        softcap,
        deterministic,
        None,
        rng_state,
    )
    # if dk.isnan().any() or dk.isnan().any() or dv.isnan().any() or softmax_d.isnan().any():
    #     breakpoint()
    return softmax_d


@_torch_register_fake_wrapper("flash_attn::_flash_attn_varlen_backward")
def _flash_attn_varlen_backward_fake(
    dout: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    softmax_lse: torch.Tensor,
    dq: Optional[torch.Tensor],
    dk: Optional[torch.Tensor],
    dv: Optional[torch.Tensor],
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    dropout_p: float,
    softmax_scale: float,
    causal: bool,
    window_size_left: int,
    window_size_right: int,
    softcap: float,
    alibi_slopes: Optional[torch.Tensor],
    deterministic: bool,
    rng_state: Optional[torch.Tensor] = None,
    zero_tensors: bool = False,
) -> torch.Tensor:
    batch_size = cu_seqlens_q.numel() - 1
    total_q, num_heads, _ = q.shape
    return torch.empty(
        (num_heads, total_q + 128 * batch_size),
        device=q.device,
        dtype=torch.float32,
    )


if torch.__version__ >= "2.4.0":
    _wrapped_flash_attn_varlen_backward = torch.ops.flash_attn._flash_attn_varlen_backward
else:
    _wrapped_flash_attn_varlen_backward = _flash_attn_varlen_backward


####################################################################################################
# Attnmask functions - FlashAttention with explicit attention mask
####################################################################################################

def _flash_attn_attnmask_forward(
    q,
    k,
    v,
    attn_mask,
    dropout_p,
    softmax_scale,
    causal,
    window_size,
    softcap,
    alibi_slopes,
    return_softmax,
    bhsd=False,
    s_aux=None,
    bias=None
):
    """
    Forward pass for FlashAttention with explicit attention mask.

    Args:
        q: Query tensor, shape [batch, seqlen_q, nheads, headdim] or [batch, nheads, seqlen_q, headdim] if bhsd
        k: Key tensor, shape [batch, seqlen_k, nheads_k, headdim]
        v: Value tensor, shape [batch, seqlen_k, nheads_k, headdim]
        attn_mask: Attention mask, shape [batch, nheads, seqlen_q, seqlen_k], dtype bool or uint8.
                   True/1 = attend, False/0 = mask out.
        dropout_p: Dropout probability
        softmax_scale: Softmax scale factor
        causal: Whether to apply causal masking
        window_size: (left, right) window sizes
        softcap: Softmax cap value
        alibi_slopes: ALiBi slopes
        return_softmax: Whether to return softmax probabilities
        bhsd: Whether input is in [batch, heads, seq, dim] format
        s_aux: (nheads,), optional. Precomputed LogSumExp values for Attention Sinks.

    Returns:
        out, q, k, v, out_padded, softmax_lse, S_dmask, rng_state
    """
    q, k, v = [maybe_contiguous(x) for x in (q, k, v)]
    if attn_mask.stride(-1) != 1:
        attn_mask = attn_mask.contiguous()

    # masked_value must be -inf for backward correctness
    masked_value = float('-inf')

    out, q, k, v, out_padded, softmax_lse, S_dmask, rng_state = flash_attn_cuda.fwd_attnmask(
        q,
        k,
        v,
        attn_mask,
        None,  # out_
        alibi_slopes,
        dropout_p,
        softmax_scale,
        causal,
        window_size[0],
        window_size[1],
        softcap,
        masked_value,
        return_softmax,
        None,  # generator
        bhsd,
        s_aux,  # Attention Sinks
        bias
    )
    return out, q, k, v, out_padded, softmax_lse, S_dmask, rng_state


def _flash_attn_attnmask_backward(
    dout,
    q,
    k,
    v,
    out,
    softmax_lse,
    attn_mask,
    dq,
    dk,
    dv,
    dropout_p,
    softmax_scale,
    causal,
    window_size,
    softcap,
    alibi_slopes,
    deterministic,
    rng_state=None,
    bhsd=False,
):
    """
    Backward pass for FlashAttention with explicit attention mask.
    """
    dout, q, k, v, out = [maybe_contiguous(x) for x in (dout, q, k, v, out)]
    if attn_mask.stride(-1) != 1:
        attn_mask = attn_mask.contiguous()

    (
        dq,
        dk,
        dv,
        softmax_d,
    ) = flash_attn_cuda.bwd_attnmask(
        dout,
        q,
        k,
        v,
        out,
        softmax_lse,
        attn_mask,
        dq,
        dk,
        dv,
        alibi_slopes,
        dropout_p,
        softmax_scale,
        causal,
        window_size[0],
        window_size[1],
        softcap,
        deterministic,
        None,  # generator
        rng_state,
        bhsd,
    )
    return dq, dk, dv, softmax_d


def _flash_attn_varlen_attnmask_forward(
    q,
    k,
    v,
    attn_mask,
    cu_seqlens_q,
    cu_seqlens_k,
    max_seqlen_q,
    max_seqlen_k,
    dropout_p,
    softmax_scale,
    causal,
    window_size,
    softcap,
    alibi_slopes,
    return_softmax,
    zero_tensors=False,
    seqused_k=None,
    s_aux=None,
):
    """
    Variable-length forward pass for FlashAttention with explicit attention mask.

    Args:
        q: Query tensor, shape [total_q, nheads, headdim]
        k: Key tensor, shape [total_k, nheads_k, headdim]
        v: Value tensor, shape [total_k, nheads_k, headdim]
        attn_mask: Attention mask, shape [batch, nheads, max_seqlen_q, max_seqlen_k]
        cu_seqlens_q: Cumulative sequence lengths for queries, shape [batch + 1]
        cu_seqlens_k: Cumulative sequence lengths for keys, shape [batch + 1]
        max_seqlen_q: Maximum query sequence length
        max_seqlen_k: Maximum key sequence length
        s_aux: (nheads,), optional. Precomputed LogSumExp values for Attention Sinks.
        ...
    """
    q, k, v = [maybe_contiguous(x) for x in (q, k, v)]
    if attn_mask.stride(-1) != 1:
        attn_mask = attn_mask.contiguous()

    masked_value = float('-inf')

    out, q, k, v, out_padded, softmax_lse, S_dmask, rng_state = flash_attn_cuda.varlen_fwd_attnmask(
        q,
        k,
        v,
        attn_mask,
        None,  # out_
        cu_seqlens_q,
        cu_seqlens_k,
        seqused_k,
        alibi_slopes,
        max_seqlen_q,
        max_seqlen_k,
        dropout_p,
        softmax_scale,
        zero_tensors,
        causal,
        window_size[0],
        window_size[1],
        softcap,
        masked_value,
        return_softmax,
        None,  # generator
        s_aux,  # Attention Sinks
    )
    return out, q, k, v, out_padded, softmax_lse, S_dmask, rng_state


def _flash_attn_varlen_attnmask_backward(
    dout,
    q,
    k,
    v,
    out,
    softmax_lse,
    attn_mask,
    dq,
    dk,
    dv,
    cu_seqlens_q,
    cu_seqlens_k,
    max_seqlen_q,
    max_seqlen_k,
    dropout_p,
    softmax_scale,
    causal,
    window_size,
    softcap,
    alibi_slopes,
    deterministic,
    rng_state=None,
    zero_tensors=False,
):
    """
    Variable-length backward pass for FlashAttention with explicit attention mask.
    """
    dout, q, k, v, out = [maybe_contiguous(x) for x in (dout, q, k, v, out)]
    if attn_mask.stride(-1) != 1:
        attn_mask = attn_mask.contiguous()

    (
        dq,
        dk,
        dv,
        softmax_d,
    ) = flash_attn_cuda.varlen_bwd_attnmask(
        dout,
        q,
        k,
        v,
        out,
        softmax_lse,
        attn_mask,
        dq,
        dk,
        dv,
        cu_seqlens_q,
        cu_seqlens_k,
        alibi_slopes,
        max_seqlen_q,
        max_seqlen_k,
        dropout_p,
        softmax_scale,
        zero_tensors,
        causal,
        window_size[0],
        window_size[1],
        softcap,
        deterministic,
        None,  # generator
        rng_state,
    )
    return dq, dk, dv, softmax_d


class FlashAttnQKVPackedFunc(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        qkv,
        dropout_p,
        softmax_scale,
        causal,
        window_size,
        softcap,
        alibi_slopes,
        deterministic,
        return_softmax,
        is_grad_enabled,
        bhsd = False
    ):
        is_grad = is_grad_enabled and qkv.requires_grad
        if softmax_scale is None:
            softmax_scale = qkv.shape[-1] ** (-0.5)
        q, k, v = qkv[:, :, 0].detach(), qkv[:, :, 1].detach(), qkv[:, :, 2].detach()
        head_size_og = q.size(3)
        if head_size_og % 8 != 0:
            q = torch.nn.functional.pad(q, [0, 8 - head_size_og % 8])
            k = torch.nn.functional.pad(k, [0, 8 - head_size_og % 8])
            v = torch.nn.functional.pad(v, [0, 8 - head_size_og % 8])
        out_padded, softmax_lse, S_dmask, rng_state =  _wrapped_flash_attn_forward(
            q,
            k,
            v,
            dropout_p,
            softmax_scale,
            causal=causal,
            window_size_left=window_size[0],
            window_size_right=window_size[1],
            softcap=softcap,
            alibi_slopes=alibi_slopes,
            return_softmax=return_softmax and dropout_p > 0 and is_grad,
            bhsd=bhsd,
        )
        if is_grad:
            ctx.save_for_backward(q, k, v, out_padded, softmax_lse, rng_state)
            ctx.dropout_p = dropout_p
            ctx.softmax_scale = softmax_scale
            ctx.causal = causal
            ctx.window_size = window_size
            ctx.softcap = softcap
            ctx.alibi_slopes = alibi_slopes
            ctx.deterministic = deterministic
            ctx.bhsd = bhsd
        out = out_padded[..., :head_size_og]
        return out if not return_softmax else (out, softmax_lse, S_dmask)

    @staticmethod
    def backward(ctx, dout, *args):
        q, k, v, out, softmax_lse, rng_state = ctx.saved_tensors
        qkv_shape = q.shape[:-2] + (3, *q.shape[-2:])
        dqkv = torch.empty(qkv_shape, dtype=q.dtype, device=q.device)
        _wrapped_flash_attn_backward(
            dout,
            q,
            k,
            v,
            out,
            softmax_lse,
            dqkv[:, :, 0],
            dqkv[:, :, 1],
            dqkv[:, :, 2],
            ctx.dropout_p,
            ctx.softmax_scale,
            ctx.causal,
            ctx.window_size[0],
            ctx.window_size[1],
            ctx.softcap,
            ctx.alibi_slopes,
            ctx.deterministic,
            rng_state=rng_state,
            bhsd=ctx.bhsd
        )
        dqkv = dqkv[..., : dout.shape[-1]]  # We could have padded the head dimension
        return dqkv, None, None, None, None, None, None, None, None, None, None


class FlashAttnVarlenQKVPackedFunc(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        qkv,
        cu_seqlens,
        max_seqlen,
        dropout_p,
        softmax_scale,
        causal,
        window_size,
        softcap,
        alibi_slopes,
        deterministic,
        return_softmax,
    ):
        is_grad = torch.is_grad_enabled() and qkv.requires_grad 
        if is_grad:
            cu_seqlens=cu_seqlens[cu_seqlens<= qkv.shape[0]]

        if softmax_scale is None:
            softmax_scale = qkv.shape[-1] ** (-0.5)
        out, softmax_lse, S_dmask, rng_state = _wrapped_flash_attn_varlen_forward(
            qkv[:, 0],
            qkv[:, 1],
            qkv[:, 2],
            None,
            cu_seqlens,
            cu_seqlens,
            None,
            None,
            None,
            alibi_slopes,
            max_seqlen,
            max_seqlen,
            dropout_p,
            softmax_scale,
            False,
            causal,
            window_size[0],
            window_size[1],
            softcap=softcap,
            return_softmax=return_softmax and dropout_p > 0,
        )
        ctx.save_for_backward(qkv[:, 0], qkv[:, 1], qkv[:, 2], out, softmax_lse, cu_seqlens, rng_state)
        ctx.dropout_p = dropout_p
        ctx.max_seqlen = max_seqlen
        ctx.softmax_scale = softmax_scale
        ctx.causal = causal
        ctx.window_size = window_size
        ctx.softcap = softcap
        ctx.alibi_slopes = alibi_slopes
        ctx.deterministic = deterministic
        return out if not return_softmax else (out, softmax_lse, S_dmask)

    @staticmethod
    def backward(ctx, dout, *args):
        q, k, v, out, softmax_lse, cu_seqlens, rng_state = ctx.saved_tensors
        qkv_shape = q.shape[:-2] + (3, *q.shape[-2:])
        dqkv = torch.empty(qkv_shape, dtype=q.dtype, device=q.device)
        _wrapped_flash_attn_varlen_backward(
            dout,
            q,
            k,
            v,
            out,
            softmax_lse,
            dqkv[:, 0],
            dqkv[:, 1],
            dqkv[:, 2],
            cu_seqlens,
            cu_seqlens,
            ctx.max_seqlen,
            ctx.max_seqlen,
            ctx.dropout_p,
            ctx.softmax_scale,
            ctx.causal,
            ctx.window_size[0],
            ctx.window_size[1],
            ctx.softcap,
            ctx.alibi_slopes,
            ctx.deterministic,
            rng_state=rng_state,
        )
        dqkv = dqkv[..., : dout.shape[-1]]  # We could have padded the head dimension
        return dqkv, None, None, None, None, None, None, None, None, None, None


class FlashAttnKVPackedFunc(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q,
        kv,
        dropout_p,
        softmax_scale,
        causal,
        window_size,
        softcap,
        alibi_slopes,
        deterministic,
        return_softmax,
        is_grad_enabled,
        bhsd=False
    ):
        is_grad = is_grad_enabled and any(
            x.requires_grad for x in [q, kv]
        )
        if softmax_scale is None:
            softmax_scale = q.shape[-1] ** (-0.5)
        k, v = kv[:, :, 0].detach(), kv[:, :, 1].detach()
        head_size_og = q.size(3)
        if head_size_og % 8 != 0:
            q = torch.nn.functional.pad(q, [0, 8 - head_size_og % 8])
            k = torch.nn.functional.pad(k, [0, 8 - head_size_og % 8])
            v = torch.nn.functional.pad(v, [0, 8 - head_size_og % 8])
        out_padded, softmax_lse, S_dmask, rng_state = _wrapped_flash_attn_forward(
            q,
            k,
            v,
            dropout_p,
            softmax_scale,
            causal=causal,
            window_size_left=window_size[0],
            window_size_right=window_size[1],
            softcap=softcap,
            alibi_slopes=alibi_slopes,
            return_softmax=return_softmax and dropout_p > 0 and is_grad,
            bhsd=bhsd
        )
        if is_grad:
            ctx.save_for_backward(q, k, v, out_padded, softmax_lse, rng_state)
            ctx.dropout_p = dropout_p
            ctx.softmax_scale = softmax_scale
            ctx.causal = causal
            ctx.window_size = window_size
            ctx.softcap = softcap
            ctx.alibi_slopes = alibi_slopes
            ctx.deterministic = deterministic
            ctx.bhsd = bhsd
        out = out_padded[..., :head_size_og]
        return out if not return_softmax else (out, softmax_lse, S_dmask)

    @staticmethod
    def backward(ctx, dout, *args):
        q, k, v, out, softmax_lse, rng_state = ctx.saved_tensors
        dq = torch.empty_like(q)
        kv_shape = k.shape[:-2] + (2, *k.shape[-2:])
        dkv = torch.empty(kv_shape, dtype=k.dtype, device=k.device)
        _wrapped_flash_attn_backward(
            dout,
            q,
            k,
            v,
            out,
            softmax_lse,
            dq,
            dkv[:, :, 0],
            dkv[:, :, 1],
            ctx.dropout_p,
            ctx.softmax_scale,
            ctx.causal,
            ctx.window_size[0],
            ctx.window_size[1],
            ctx.softcap,
            ctx.alibi_slopes,
            ctx.deterministic,
            rng_state=rng_state,
            bhsd=ctx.bhsd
        )
        dq = dq[..., : dout.shape[-1]]  # We could have padded the head dimension
        dkv = dkv[..., : dout.shape[-1]]
        return dq, dkv, None, None, None, None, None, None, None, None, None, None


class FlashAttnVarlenKVPackedFunc(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q,
        kv,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        dropout_p,
        softmax_scale,
        causal,
        window_size,
        softcap,
        alibi_slopes,
        deterministic,
        return_softmax,
    ):
        is_grad = torch.is_grad_enabled() and any(x.requires_grad for x in [q, kv])
        if is_grad:
            cu_seqlens_q=cu_seqlens_q[cu_seqlens_q<= q.shape[0]]
            cu_seqlens_k=cu_seqlens_k[cu_seqlens_k<= kv.shape[0]]

        if softmax_scale is None:
            softmax_scale = q.shape[-1] ** (-0.5)
        out, softmax_lse, S_dmask, rng_state = _wrapped_flash_attn_varlen_forward(
            q,
            kv[:, 0],
            kv[:, 1],
            None,
            cu_seqlens_q,
            cu_seqlens_k,
            None,
            None,
            None,
            alibi_slopes,
            max_seqlen_q,
            max_seqlen_k,
            dropout_p,
            softmax_scale,
            False,
            causal,
            window_size[0],
            window_size[1],
            softcap=softcap,
            return_softmax=return_softmax and dropout_p > 0,
        )
        ctx.save_for_backward(
            q, kv[:, 0], kv[:, 1], out, softmax_lse, cu_seqlens_q, cu_seqlens_k, rng_state
        )
        ctx.dropout_p = dropout_p
        ctx.max_seqlen_q = max_seqlen_q
        ctx.max_seqlen_k = max_seqlen_k
        ctx.softmax_scale = softmax_scale
        ctx.causal = causal
        ctx.window_size = window_size
        ctx.softcap = softcap
        ctx.alibi_slopes = alibi_slopes
        ctx.deterministic = deterministic
        return out if not return_softmax else (out, softmax_lse, S_dmask)

    @staticmethod
    def backward(ctx, dout, *args):
        q, k, v, out, softmax_lse, cu_seqlens_q, cu_seqlens_k, rng_state = ctx.saved_tensors
        dq = torch.empty_like(q)
        kv_shape = k.shape[:-2] + (2, *k.shape[-2:])
        dkv = torch.empty(kv_shape, dtype=k.dtype, device=k.device)
        _wrapped_flash_attn_varlen_backward(
            dout,
            q,
            k,
            v,
            out,
            softmax_lse,
            dq,
            dkv[:, 0],
            dkv[:, 1],
            cu_seqlens_q,
            cu_seqlens_k,
            ctx.max_seqlen_q,
            ctx.max_seqlen_k,
            ctx.dropout_p,
            ctx.softmax_scale,
            ctx.causal,
            ctx.window_size[0],
            ctx.window_size[1],
            ctx.softcap,
            ctx.alibi_slopes,
            ctx.deterministic,
            rng_state=rng_state,
        )
        dq = dq[..., : dout.shape[-1]]  # We could have padded the head dimension
        dkv = dkv[..., : dout.shape[-1]]
        return dq, dkv, None, None, None, None, None, None, None, None, None, None, None, None


class FlashAttnFunc(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q,
        k,
        v,
        dropout_p,
        softmax_scale,
        causal,
        window_size,
        softcap,
        alibi_slopes,
        deterministic,
        return_softmax,
        is_grad_enabled,
        bhsd = False,
        s_aux = None  # Attention Sinks
    ):
        is_grad = is_grad_enabled and any(
            x.requires_grad for x in [q, k, v]
        )
        if softmax_scale is None:
            softmax_scale = q.shape[-1] ** (-0.5)
        head_size_og = q.size(3)
        if head_size_og % 8 != 0:
            q = torch.nn.functional.pad(q, [0, 8 - head_size_og % 8])
            k = torch.nn.functional.pad(k, [0, 8 - head_size_og % 8])
            v = torch.nn.functional.pad(v, [0, 8 - head_size_og % 8])
        out_padded, softmax_lse, S_dmask, rng_state = _wrapped_flash_attn_forward(
            q,
            k,
            v,
            dropout_p,
            softmax_scale,
            causal=causal,
            window_size_left=window_size[0],
            window_size_right=window_size[1],
            softcap=softcap,
            alibi_slopes=alibi_slopes,
            return_softmax=return_softmax and dropout_p > 0 and is_grad,
            bhsd = bhsd,
            s_aux = s_aux,
        )
        if is_grad:
            ctx.save_for_backward(q, k, v, out_padded, softmax_lse, rng_state)
            ctx.dropout_p = dropout_p
            ctx.softmax_scale = softmax_scale
            ctx.causal = causal
            ctx.window_size = window_size
            ctx.softcap = softcap
            ctx.alibi_slopes = alibi_slopes
            ctx.deterministic = deterministic
            ctx.bhsd = bhsd
            ctx.qk_headdim = q.shape[-1]
        out = out_padded[..., :head_size_og]
        return out if not return_softmax else (out, softmax_lse, S_dmask)

    @staticmethod
    def backward(ctx, dout, *args):
        q, k, v, out, softmax_lse, rng_state = ctx.saved_tensors
        dq, dk, dv = torch.empty_like(q), torch.empty_like(k), torch.empty_like(v)
        _wrapped_flash_attn_backward(
            dout,
            q,
            k,
            v,
            out,
            softmax_lse,
            dq,
            dk,
            dv,
            ctx.dropout_p,
            ctx.softmax_scale,
            ctx.causal,
            ctx.window_size[0],
            ctx.window_size[1],
            ctx.softcap,
            ctx.alibi_slopes,
            ctx.deterministic,
            rng_state=rng_state,
            bhsd=ctx.bhsd
        )
        # dq = dq[..., : dout.shape[-1]]  # We could have padded the head dimension
        # dk = dk[..., : dout.shape[-1]]
        # dv = dv[..., : dout.shape[-1]]
        dq = dq[..., : ctx.qk_headdim]  # We could have padded the head dimension
        dk = dk[..., : ctx.qk_headdim]
        dv = dv[..., : dout.shape[-1]]
        return dq, dk, dv, None, None, None, None, None, None, None, None, None, None, None


class FlashAttnVarlenFunc(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q,
        k,
        v,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        dropout_p,
        softmax_scale,
        causal,
        window_size,
        softcap,
        alibi_slopes,
        deterministic,
        return_softmax,
        q_descale,
        k_descale,
        v_descale,
        block_table,
        s_aux,
        is_grad_enabled
    ):
        is_grad = is_grad_enabled and any(x.requires_grad for x in [q, k, v])
        if is_grad:
            cu_seqlens_q=cu_seqlens_q[cu_seqlens_q<= q.shape[0]]
            cu_seqlens_k=cu_seqlens_k[cu_seqlens_k<= k.shape[0]]      
        if softmax_scale is None:
            softmax_scale = q.shape[-1] ** (-0.5)
        # The torch.compile custom op path only exposes the reduced varlen_fwd schema
        # from flash_attn2_c_op::varlen_fwd. Keep using it for the supported subset,
        # but fall back to the original pybind path when extra features are requested.
        if any(x is not None for x in (q_descale, k_descale, v_descale, block_table, s_aux)):
            out, softmax_lse, S_dmask, rng_state = _flash_attn_varlen_forward(
                q,
                k,
                v,
                cu_seqlens_q,
                cu_seqlens_k,
                max_seqlen_q,
                max_seqlen_k,
                dropout_p,
                softmax_scale,
                causal,
                window_size[0],
                window_size[1],
                softcap,
                alibi_slopes,
                return_softmax and dropout_p > 0 and is_grad,
                q_descale=q_descale,
                k_descale=k_descale,
                v_descale=v_descale,
                block_table=block_table,
                s_aux=s_aux,
            )
        else:
            out, softmax_lse, S_dmask, rng_state = _wrapped_flash_attn_varlen_forward(
                q,
                k,
                v,
                None,
                cu_seqlens_q,
                cu_seqlens_k,
                None,
                None,
                None,
                alibi_slopes,
                max_seqlen_q,
                max_seqlen_k,
                dropout_p,
                softmax_scale,
                False,
                causal,
                window_size[0],
                window_size[1],
                softcap=softcap,
                return_softmax=return_softmax and dropout_p > 0 and is_grad,
            )
        if is_grad:
            ctx.qk_headdim = q.shape[-1]
            ctx.save_for_backward(
                q, k, v, out, softmax_lse, cu_seqlens_q, cu_seqlens_k, rng_state
            )
            ctx.dropout_p = dropout_p
            ctx.max_seqlen_q = max_seqlen_q
            ctx.max_seqlen_k = max_seqlen_k
            ctx.softmax_scale = softmax_scale
            ctx.causal = causal
            ctx.window_size = window_size
            ctx.softcap = softcap
            ctx.alibi_slopes = alibi_slopes
            ctx.deterministic = deterministic
        return out if not return_softmax else (out, softmax_lse, S_dmask)

    @staticmethod
    def backward(ctx, dout, *args):
        q, k, v, out, softmax_lse, cu_seqlens_q, cu_seqlens_k, rng_state = ctx.saved_tensors
        dq, dk, dv = torch.empty_like(q), torch.empty_like(k), torch.empty_like(v)
        _wrapped_flash_attn_varlen_backward(
            dout,
            q,
            k,
            v,
            out,
            softmax_lse,
            dq,
            dk,
            dv,
            cu_seqlens_q,
            cu_seqlens_k,
            ctx.max_seqlen_q,
            ctx.max_seqlen_k,
            ctx.dropout_p,
            ctx.softmax_scale,
            ctx.causal,
            ctx.window_size[0],
            ctx.window_size[1],
            ctx.softcap,
            ctx.alibi_slopes,
            ctx.deterministic,
            rng_state=rng_state,
        )
        dq = dq[..., : ctx.qk_headdim]  # We could have padded the head dimension
        dk = dk[..., : ctx.qk_headdim]
        dv = dv[..., : dout.shape[-1]]
        return dq, dk, dv, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None


####################################################################################################
# Attnmask autograd.Function classes
####################################################################################################

class FlashAttnAttnmaskFunc(torch.autograd.Function):
    """
    FlashAttention with explicit attention mask - fixed length version.
    """

    @staticmethod
    def forward(
        ctx,
        q,
        k,
        v,
        attn_mask,
        dropout_p,
        softmax_scale,
        causal,
        window_size,
        softcap,
        alibi_slopes,
        deterministic,
        return_softmax,
        bhsd,
        s_aux,
        bias
    ):
        if softmax_scale is None:
            softmax_scale = q.shape[-1] ** (-0.5)
        out, q, k, v, out_padded, softmax_lse, S_dmask, rng_state = _flash_attn_attnmask_forward(
            q,
            k,
            v,
            attn_mask,
            dropout_p,
            softmax_scale,
            causal=causal,
            window_size=window_size,
            softcap=softcap,
            alibi_slopes=alibi_slopes,
            return_softmax=return_softmax and dropout_p > 0,
            bhsd=bhsd,
            s_aux=s_aux,
            bias=bias
        )
        ctx.qk_headdim = q.shape[-1]
        ctx.save_for_backward(q, k, v, out_padded, softmax_lse, attn_mask, rng_state)
        ctx.dropout_p = dropout_p
        ctx.softmax_scale = softmax_scale
        ctx.causal = causal
        ctx.window_size = window_size
        ctx.softcap = softcap
        ctx.alibi_slopes = alibi_slopes
        ctx.deterministic = deterministic
        ctx.bhsd = bhsd
        return out if not return_softmax else (out, softmax_lse, S_dmask)

    @staticmethod
    def backward(ctx, dout, *args):
        q, k, v, out, softmax_lse, attn_mask, rng_state = ctx.saved_tensors
        dq, dk, dv = torch.empty_like(q), torch.empty_like(k), torch.empty_like(v)
        _flash_attn_attnmask_backward(
            dout,
            q,
            k,
            v,
            out,
            softmax_lse,
            attn_mask,
            dq,
            dk,
            dv,
            ctx.dropout_p,
            ctx.softmax_scale,
            ctx.causal,
            ctx.window_size,
            ctx.softcap,
            ctx.alibi_slopes,
            ctx.deterministic,
            rng_state=rng_state,
            bhsd=ctx.bhsd,
        )
        dq = dq[..., : ctx.qk_headdim]
        dk = dk[..., : ctx.qk_headdim]
        dv = dv[..., : dout.shape[-1]]
        return dq, dk, dv, None, None, None, None, None, None, None, None, None, None, None, None


class FlashAttnVarlenAttnmaskFunc(torch.autograd.Function):
    """
    FlashAttention with explicit attention mask - variable length version.
    """

    @staticmethod
    def forward(
        ctx,
        q,
        k,
        v,
        attn_mask,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        dropout_p,
        softmax_scale,
        causal,
        window_size,
        softcap,
        alibi_slopes,
        deterministic,
        return_softmax,
        zero_tensors,
        seqused_k,
        s_aux,
    ):
        if softmax_scale is None:
            softmax_scale = q.shape[-1] ** (-0.5)
        out, q, k, v, out_padded, softmax_lse, S_dmask, rng_state = _flash_attn_varlen_attnmask_forward(
            q,
            k,
            v,
            attn_mask,
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q,
            max_seqlen_k,
            dropout_p,
            softmax_scale,
            causal=causal,
            window_size=window_size,
            softcap=softcap,
            alibi_slopes=alibi_slopes,
            return_softmax=return_softmax and dropout_p > 0,
            zero_tensors=zero_tensors,
            seqused_k=seqused_k,
            s_aux=s_aux,
        )
        ctx.qk_headdim = q.shape[-1]
        ctx.save_for_backward(
            q, k, v, out_padded, softmax_lse, attn_mask, cu_seqlens_q, cu_seqlens_k, rng_state
        )
        ctx.dropout_p = dropout_p
        ctx.max_seqlen_q = max_seqlen_q
        ctx.max_seqlen_k = max_seqlen_k
        ctx.softmax_scale = softmax_scale
        ctx.causal = causal
        ctx.window_size = window_size
        ctx.softcap = softcap
        ctx.alibi_slopes = alibi_slopes
        ctx.deterministic = deterministic
        ctx.zero_tensors = zero_tensors
        return out if not return_softmax else (out, softmax_lse, S_dmask)

    @staticmethod
    def backward(ctx, dout, *args):
        q, k, v, out, softmax_lse, attn_mask, cu_seqlens_q, cu_seqlens_k, rng_state = ctx.saved_tensors
        dq, dk, dv = torch.empty_like(q), torch.empty_like(k), torch.empty_like(v)
        _flash_attn_varlen_attnmask_backward(
            dout,
            q,
            k,
            v,
            out,
            softmax_lse,
            attn_mask,
            dq,
            dk,
            dv,
            cu_seqlens_q,
            cu_seqlens_k,
            ctx.max_seqlen_q,
            ctx.max_seqlen_k,
            ctx.dropout_p,
            ctx.softmax_scale,
            ctx.causal,
            ctx.window_size,
            ctx.softcap,
            ctx.alibi_slopes,
            ctx.deterministic,
            rng_state=rng_state,
            zero_tensors=ctx.zero_tensors,
        )
        dq = dq[..., : ctx.qk_headdim]
        dk = dk[..., : ctx.qk_headdim]
        dv = dv[..., : dout.shape[-1]]
        return dq, dk, dv, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None


def flash_attn_qkvpacked_func(
    qkv,
    dropout_p=0.0,
    softmax_scale=None,
    causal=False,
    window_size=(-1, -1),  # -1 means infinite context window
    softcap=0.0,  # <=0.0 means deactivate
    alibi_slopes=None,
    deterministic=False,
    return_attn_probs=False,
    bhsd = False
):
    """dropout_p should be set to 0.0 during evaluation
    If Q, K, V are already stacked into 1 tensor, this function will be faster than
    calling flash_attn_func on Q, K, V since the backward pass avoids explicit concatenation
    of the gradients of Q, K, V.
    For multi-query and grouped-query attention (MQA/GQA), please see
    flash_attn_kvpacked_func and flash_attn_func.

    If window_size != (-1, -1), implements sliding window local attention. Query at position i
    will only attend to keys between [i - window_size[0], i + window_size[1]] inclusive.

    Arguments:
        qkv: (batch_size, seqlen, 3, nheads, headdim)
        dropout_p: float. Dropout probability.
        softmax_scale: float. The scaling of QK^T before applying softmax.
            Default to 1 / sqrt(headdim).
        causal: bool. Whether to apply causal attention mask (e.g., for auto-regressive modeling).
        window_size: (left, right). If not (-1, -1), implements sliding window local attention.
        softcap: float. Anything > 0 activates softcapping attention.
        alibi_slopes: (nheads,) or (batch_size, nheads), fp32. A bias of (-alibi_slope * |i - j|) is added to
            the attention score of query i and key j.
        deterministic: bool. Whether to use the deterministic implementation of the backward pass,
            which is slightly slower and uses more memory. The forward pass is always deterministic.
        return_attn_probs: bool. Whether to return the attention probabilities. This option is for
           testing only. The returned probabilities are not guaranteed to be correct
           (they might not have the right scaling).
    Return:
        out: (batch_size, seqlen, nheads, headdim).
        softmax_lse [optional, if return_attn_probs=True]: (batch_size, nheads, seqlen). The
            logsumexp of each row of the matrix QK^T * scaling (e.g., log of the softmax
            normalization factor).
        S_dmask [optional, if return_attn_probs=True]: (batch_size, nheads, seqlen, seqlen).
            The output of softmax (possibly with different scaling). It also encodes the dropout
            pattern (negative means that location was dropped, nonnegative means it was kept).
    """
    return FlashAttnQKVPackedFunc.apply(
        qkv,
        dropout_p,
        softmax_scale,
        causal,
        window_size,
        softcap,
        alibi_slopes,
        deterministic,
        return_attn_probs,
        torch.is_grad_enabled(),
        bhsd
    )


def flash_attn_kvpacked_func(
    q,
    kv,
    dropout_p=0.0,
    softmax_scale=None,
    causal=False,
    window_size=(-1, -1),  # -1 means infinite context window
    softcap=0.0,  # 0.0 means deactivated
    alibi_slopes=None,
    deterministic=False,
    return_attn_probs=False,
    bhsd = False
):
    """dropout_p should be set to 0.0 during evaluation
    If K, V are already stacked into 1 tensor, this function will be faster than
    calling flash_attn_func on Q, K, V since the backward pass avoids explicit concatenation
    of the gradients of K, V.
    Supports multi-query and grouped-query attention (MQA/GQA) by passing in KV with fewer heads
    than Q. Note that the number of heads in Q must be divisible by the number of heads in KV.
    For example, if Q has 6 heads and K, V have 2 heads, head 0, 1, 2 of Q will attention to head
    0 of K, V, and head 3, 4, 5 of Q will attention to head 1 of K, V.

    If causal=True, the causal mask is aligned to the bottom right corner of the attention matrix.
    For example, if seqlen_q = 2 and seqlen_k = 5, the causal mask (1 = keep, 0 = masked out) is:
        1 1 1 1 0
        1 1 1 1 1
    If seqlen_q = 5 and seqlen_k = 2, the causal mask is:
        0 0
        0 0
        0 0
        1 0
        1 1
    If the row of the mask is all zero, the output will be zero.

    If window_size != (-1, -1), implements sliding window local attention. Query at position i
    will only attend to keys between
    [i + seqlen_k - seqlen_q - window_size[0], i + seqlen_k - seqlen_q + window_size[1]] inclusive.

    Arguments:
        q: (batch_size, seqlen, nheads, headdim)
        kv: (batch_size, seqlen, 2, nheads_k, headdim)
        dropout_p: float. Dropout probability.
        softmax_scale: float. The scaling of QK^T before applying softmax.
            Default to 1 / sqrt(headdim).
        causal: bool. Whether to apply causal attention mask (e.g., for auto-regressive modeling).
        window_size: (left, right). If not (-1, -1), implements sliding window local attention.
        softcap: float. Anything > 0 activates softcapping attention.
        alibi_slopes: (nheads,) or (batch_size, nheads), fp32. A bias of
            (-alibi_slope * |i + seqlen_k - seqlen_q - j|)
            is added to the attention score of query i and key j.
        deterministic: bool. Whether to use the deterministic implementation of the backward pass,
            which is slightly slower and uses more memory. The forward pass is always deterministic.
        return_attn_probs: bool. Whether to return the attention probabilities. This option is for
           testing only. The returned probabilities are not guaranteed to be correct
           (they might not have the right scaling).
    Return:
        out: (batch_size, seqlen, nheads, headdim).
        softmax_lse [optional, if return_attn_probs=True]: (batch_size, nheads, seqlen). The
            logsumexp of each row of the matrix QK^T * scaling (e.g., log of the softmax
            normalization factor).
        S_dmask [optional, if return_attn_probs=True]: (batch_size, nheads, seqlen, seqlen).
            The output of softmax (possibly with different scaling). It also encodes the dropout
            pattern (negative means that location was dropped, nonnegative means it was kept).
    """
    return FlashAttnKVPackedFunc.apply(
        q,
        kv,
        dropout_p,
        softmax_scale,
        causal,
        window_size,
        softcap,
        alibi_slopes,
        deterministic,
        return_attn_probs,
        torch.is_grad_enabled(),
        bhsd
    )


def flash_attn_func(
    q,
    k,
    v,
    dropout_p=0.0,
    softmax_scale=None,
    causal=False,
    window_size=(-1, -1),  # -1 means infinite context window
    softcap=0.0, # 0.0 means deactivated
    alibi_slopes=None,
    deterministic=False,
    return_attn_probs=False,
    bhsd = False,
    s_aux=None  # Attention Sinks: [nheads]
):
    """dropout_p should be set to 0.0 during evaluation
    Supports multi-query and grouped-query attention (MQA/GQA) by passing in KV with fewer heads
    than Q. Note that the number of heads in Q must be divisible by the number of heads in KV.
    For example, if Q has 6 heads and K, V have 2 heads, head 0, 1, 2 of Q will attention to head
    0 of K, V, and head 3, 4, 5 of Q will attention to head 1 of K, V.

    If causal=True, the causal mask is aligned to the bottom right corner of the attention matrix.
    For example, if seqlen_q = 2 and seqlen_k = 5, the causal mask (1 = keep, 0 = masked out) is:
        1 1 1 1 0
        1 1 1 1 1
    If seqlen_q = 5 and seqlen_k = 2, the causal mask is:
        0 0
        0 0
        0 0
        1 0
        1 1
    If the row of the mask is all zero, the output will be zero.

    If window_size != (-1, -1), implements sliding window local attention. Query at position i
    will only attend to keys between
    [i + seqlen_k - seqlen_q - window_size[0], i + seqlen_k - seqlen_q + window_size[1]] inclusive.

    Arguments:
        q: (batch_size, seqlen, nheads, headdim)
        k: (batch_size, seqlen, nheads_k, headdim)
        v: (batch_size, seqlen, nheads_k, headdim)
        dropout_p: float. Dropout probability.
        softmax_scale: float. The scaling of QK^T before applying softmax.
            Default to 1 / sqrt(headdim).
        causal: bool. Whether to apply causal attention mask (e.g., for auto-regressive modeling).
        window_size: (left, right). If not (-1, -1), implements sliding window local attention.
        alibi_slopes: (nheads,) or (batch_size, nheads), fp32. A bias of
            (-alibi_slope * |i + seqlen_k - seqlen_q - j|)
            is added to the attention score of query i and key j.
        deterministic: bool. Whether to use the deterministic implementation of the backward pass,
            which is slightly slower and uses more memory. The forward pass is always deterministic.
        return_attn_probs: bool. Whether to return the attention probabilities. This option is for
           testing only. The returned probabilities are not guaranteed to be correct
           (they might not have the right scaling).
        s_aux: (nheads,), optional, dtype float16 or bfloat16 (must match Q/K/V dtype).
            Precomputed LogSumExp values for sink tokens in Streaming LLM.
            Used to implement Attention Sinks for maintaining attention to initial tokens.
            IMPORTANT: s_aux.dtype must equal q.dtype (enforced by runtime check).
            Storage uses FP16/BF16 for bandwidth efficiency (50% reduction vs FP32).
            Computation uses FP32 internally for numerical stability.
            If provided, enables Attention Sinks with <0.5% overhead. Maximum 64 heads supported.
    Return:
        out: (batch_size, seqlen, nheads, headdim).
        softmax_lse [optional, if return_attn_probs=True]: (batch_size, nheads, seqlen). The
            logsumexp of each row of the matrix QK^T * scaling (e.g., log of the softmax
            normalization factor).
        S_dmask [optional, if return_attn_probs=True]: (batch_size, nheads, seqlen, seqlen).
            The output of softmax (possibly with different scaling). It also encodes the dropout
            pattern (negative means that location was dropped, nonnegative means it was kept).
    """
    return FlashAttnFunc.apply(
        q,
        k,
        v,
        dropout_p,
        softmax_scale,
        causal,
        window_size,
        softcap,
        alibi_slopes,
        deterministic,
        return_attn_probs,
        torch.is_grad_enabled(),
        bhsd,
        s_aux  # Attention Sinks
    )

####################################################################################################
# Attnmask high-level user interface functions
####################################################################################################

def flash_attn_with_mask_func(
    q,
    k,
    v,
    attn_mask,
    dropout_p=0.0,
    softmax_scale=None,
    causal=False,
    window_size=(-1, -1),
    softcap=0.0,
    alibi_slopes=None,
    deterministic=False,
    return_attn_probs=False,
    bhsd=False,
    s_aux=None,
    bias=None
):
    """FlashAttention with explicit attention mask.

    This function applies FlashAttention with a user-provided attention mask.
    The mask allows fine-grained control over which query-key pairs attend to each other.

    Constraints:
        - headdim must be 128
        - attn_mask's last dimension (seqlen_k) must be contiguous
        - q/k/v dtype must be fp16 or bf16
        - True in mask = attend, False = mask out. Non-bool is converted: integer 1/0 -> True/False; float 0/-inf -> True/False.

    Arguments:
        q: (batch_size, seqlen_q, nheads, headdim) or (batch_size, nheads, seqlen_q, headdim) if bhsd
        k: (batch_size, seqlen_k, nheads_k, headdim)
        v: (batch_size, seqlen_k, nheads_k, headdim)
        attn_mask: (batch_size, nheads, seqlen_q, seqlen_k), dtype bool, integer, or float.
            If bool: True = attend, False = mask out.
            If integer (uint8/int8/...): 1 = attend, 0 = mask out.
            If float: 0 = attend, -inf = mask out (converted to bool internally).
        dropout_p: float. Dropout probability.
        softmax_scale: float. The scaling of QK^T before applying softmax.
            Default to 1 / sqrt(headdim).
        causal: bool. Whether to apply causal attention mask (combined with attn_mask).
        window_size: (left, right). If not (-1, -1), implements sliding window local attention.
        softcap: float. Softmax cap value. 0.0 means deactivated.
        alibi_slopes: (nheads,) or (batch_size, nheads), fp32. ALiBi slopes.
        deterministic: bool. Whether to use the deterministic implementation of the backward pass.
        return_attn_probs: bool. Whether to return the attention probabilities.
        bhsd: bool. Whether input is in [batch, heads, seq, dim] format.
        s_aux: (nheads,), optional, dtype float16 or bfloat16 (must match Q/K/V dtype).
            Precomputed LogSumExp values for sink tokens in Streaming LLM.
            Used to implement Attention Sinks for maintaining attention to initial tokens.

    Returns:
        out: (batch_size, seqlen_q, nheads, headdim) or (batch_size, nheads, seqlen_q, headdim) if bhsd.
        softmax_lse [optional]: (batch_size, nheads, seqlen_q). The logsumexp values.
        S_dmask [optional]: The softmax output (for testing only).

    Example:
        >>> q = torch.randn(2, 128, 8, 128, dtype=torch.float16, device='cuda')
        >>> k = torch.randn(2, 256, 8, 128, dtype=torch.float16, device='cuda')
        >>> v = torch.randn(2, 256, 8, 128, dtype=torch.float16, device='cuda')
        >>> # Create mask: first 128 K positions are visible
        >>> attn_mask = torch.zeros(2, 8, 128, 256, dtype=torch.bool, device='cuda')
        >>> attn_mask[:, :, :, :128] = True
        >>> out = flash_attn_with_mask_func(q, k, v, attn_mask)
    """
    # Kernel expects bool mask. Convert non-bool:
    # - Integer (uint8/int8/...): 1 -> True (attend), 0 -> False (mask out).
    # - Float: 0 -> True (attend), -inf -> False (mask out).
    if attn_mask.dtype != torch.bool:
        if attn_mask.dtype in (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64):
            attn_mask = (attn_mask != 0)
        else:
            attn_mask = (attn_mask == 0)
    return FlashAttnAttnmaskFunc.apply(
        q,
        k,
        v,
        attn_mask,
        dropout_p,
        softmax_scale,
        causal,
        window_size,
        softcap,
        alibi_slopes,
        deterministic,
        return_attn_probs,
        bhsd,
        s_aux,
        bias
    )


def flash_attn_varlen_with_mask_func(
    q,
    k,
    v,
    attn_mask,
    cu_seqlens_q,
    cu_seqlens_k,
    max_seqlen_q,
    max_seqlen_k,
    dropout_p=0.0,
    softmax_scale=None,
    causal=False,
    window_size=(-1, -1),
    softcap=0.0,
    alibi_slopes=None,
    deterministic=False,
    return_attn_probs=False,
    zero_tensors=False,
    seqused_k=None,
    s_aux=None,
):
    """Variable-length FlashAttention with explicit attention mask.

    This function applies FlashAttention with variable-length sequences and a user-provided
    attention mask.

    Constraints:
        - headdim must be 128
        - attn_mask's last dimension must be contiguous
        - q/k/v dtype must be fp16 or bf16

    Arguments:
        q: (total_q, nheads, headdim), where total_q = sum of seqlen_q across batch.
        k: (total_k, nheads_k, headdim), where total_k = sum of seqlen_k across batch.
        v: (total_k, nheads_k, headdim).
        attn_mask: (batch_size, nheads, max_seqlen_q, max_seqlen_k), dtype bool, integer, or float.
            If bool: True = attend, False = mask out. If integer: 1 = attend, 0 = mask out. If float: 0 = attend, -inf = mask out.
        cu_seqlens_q: (batch_size + 1,), dtype int32. Cumulative sequence lengths for queries.
        cu_seqlens_k: (batch_size + 1,), dtype int32. Cumulative sequence lengths for keys.
        max_seqlen_q: int. Maximum query sequence length in the batch.
        max_seqlen_k: int. Maximum key sequence length in the batch.
        dropout_p: float. Dropout probability.
        softmax_scale: float. The scaling of QK^T before applying softmax.
        causal: bool. Whether to apply causal attention mask.
        window_size: (left, right). Sliding window size.
        softcap: float. Softmax cap value.
        alibi_slopes: ALiBi slopes.
        deterministic: bool. Whether to use deterministic backward.
        return_attn_probs: bool. Whether to return attention probabilities.
        zero_tensors: bool. Whether to zero output tensors before computation.
        seqused_k: (batch_size,), optional. Actual K length used per sample.
        s_aux: (nheads,), optional, dtype float16 or bfloat16 (must match Q/K/V dtype).
            Precomputed LogSumExp values for sink tokens in Streaming LLM.
            Used to implement Attention Sinks for maintaining attention to initial tokens.

    Returns:
        out: (total_q, nheads, headdim).
    """
    # Kernel expects bool mask. Convert non-bool:
    # - Integer (uint8/int8/...): 1 -> True (attend), 0 -> False (mask out).
    # - Float: 0 -> True (attend), -inf -> False (mask out).
    if attn_mask.dtype != torch.bool:
        if attn_mask.dtype in (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64):
            attn_mask = (attn_mask != 0)
        else:
            attn_mask = (attn_mask == 0)
    return FlashAttnVarlenAttnmaskFunc.apply(
        q,
        k,
        v,
        attn_mask,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        dropout_p,
        softmax_scale,
        causal,
        window_size,
        softcap,
        alibi_slopes,
        deterministic,
        return_attn_probs,
        zero_tensors,
        seqused_k,
        s_aux,
    )


def flash_attn_func_blasst(
    q,
    k,
    v,
    dropout_p=0.0,
    softmax_scale=None,
    causal=False,
    window_size=(-1, -1),  # -1 means infinite context window
    softcap=0.0, # 0.0 means deactivated
    alibi_slopes=None,
    deterministic=False,
    return_attn_probs=False,
    bhsd = False,
    s_aux=None,  # Attention Sinks: [nheads]
    blasst_threshold_scale_factor = 1.0e-9
):
    assert blasst_threshold_scale_factor > 0, "blasst_threshold_scale_factor must be great than 0.0f"
    if softmax_scale is None:
        softmax_scale = q.shape[-1] ** (-0.5)
    return_softmax=return_attn_probs and dropout_p > 0
    out, q, k, v, out_padded, softmax_lse, S_dmask, rng_state, skip_info = flash_attn_cuda.fwd(
        q,
        k,
        v,
        None,
        alibi_slopes,
        dropout_p,
        softmax_scale,
        causal,
        window_size[0],
        window_size[1],
        softcap,
        return_softmax,
        None, bhsd, s_aux,  # Attention Sinks
        blasst_threshold_scale_factor
    )
    return (out, skip_info) if not return_attn_probs else (out, softmax_lse, S_dmask, skip_info)

def flash_attn_varlen_qkvpacked_func(
    qkv,
    cu_seqlens,
    max_seqlen,
    dropout_p=0.0,
    softmax_scale=None,
    causal=False,
    window_size=(-1, -1),  # -1 means infinite context window
    softcap=0.0, # 0.0 means deactivated
    alibi_slopes=None,
    deterministic=False,
    return_attn_probs=False,
):
    """dropout_p should be set to 0.0 during evaluation
    If Q, K, V are already stacked into 1 tensor, this function will be faster than
    calling flash_attn_varlen_func on Q, K, V since the backward pass avoids explicit concatenation
    of the gradients of Q, K, V.
    For multi-query and grouped-query attention (MQA/GQA), please see
    flash_attn_varlen_kvpacked_func and flash_attn_varlen_func.

    If window_size != (-1, -1), implements sliding window local attention. Query at position i
    will only attend to keys between [i - window_size[0], i + window_size[1]] inclusive.

    Arguments:
        qkv: (total, 3, nheads, headdim), where total = total number of tokens in the batch.
        cu_seqlens: (batch_size + 1,), dtype torch.int32. The cumulative sequence lengths
           of the sequences in the batch, used to index into qkv.
        max_seqlen: int. Maximum sequence length in the batch.
        dropout_p: float. Dropout probability.
        softmax_scale: float. The scaling of QK^T before applying softmax.
            Default to 1 / sqrt(headdim).
        causal: bool. Whether to apply causal attention mask (e.g., for auto-regressive modeling).
        window_size: (left, right). If not (-1, -1), implements sliding window local attention.
        softcap: float. Anything > 0 activates softcapping attention.
        alibi_slopes: (nheads,) or (batch_size, nheads), fp32. A bias of (-alibi_slope * |i - j|)
            is added to the attention score of query i and key j.
        deterministic: bool. Whether to use the deterministic implementation of the backward pass,
            which is slightly slower and uses more memory. The forward pass is always deterministic.
        return_attn_probs: bool. Whether to return the attention probabilities. This option is for
           testing only. The returned probabilities are not guaranteed to be correct
           (they might not have the right scaling).
    Return:
        out: (total, nheads, headdim).
        softmax_lse [optional, if return_attn_probs=True]: (nheads, total_q_seqlen). The
            logsumexp of each row of the matrix QK^T * scaling (e.g., log of the softmax
            normalization factor).
        S_dmask [optional, if return_attn_probs=True]: (batch_size, nheads, seqlen, seqlen).
            The output of softmax (possibly with different scaling). It also encodes the dropout
            pattern (negative means that location was dropped, nonnegative means it was kept).
    """
    return FlashAttnVarlenQKVPackedFunc.apply(
        qkv,
        cu_seqlens,
        max_seqlen,
        dropout_p,
        softmax_scale,
        causal,
        window_size,
        softcap,
        alibi_slopes,
        deterministic,
        return_attn_probs,
    )


def flash_attn_varlen_kvpacked_func(
    q,
    kv,
    cu_seqlens_q,
    cu_seqlens_k,
    max_seqlen_q,
    max_seqlen_k,
    dropout_p=0.0,
    softmax_scale=None,
    causal=False,
    window_size=(-1, -1),  # -1 means infinite context window
    softcap=0.0, # 0.0 means deactivated
    alibi_slopes=None,
    deterministic=False,
    return_attn_probs=False,
):
    """dropout_p should be set to 0.0 during evaluation
    If K, V are already stacked into 1 tensor, this function will be faster than
    calling flash_attn_func on Q, K, V since the backward pass avoids explicit concatenation
    of the gradients of K, V.
    Supports multi-query and grouped-query attention (MQA/GQA) by passing in KV with fewer heads
    than Q. Note that the number of heads in Q must be divisible by the number of heads in KV.
    For example, if Q has 6 heads and K, V have 2 heads, head 0, 1, 2 of Q will attention to head
    0 of K, V, and head 3, 4, 5 of Q will attention to head 1 of K, V.

    If causal=True, the causal mask is aligned to the bottom right corner of the attention matrix.
    For example, if seqlen_q = 2 and seqlen_k = 5, the causal mask (1 = keep, 0 = masked out) is:
        1 1 1 1 0
        1 1 1 1 1
    If seqlen_q = 5 and seqlen_k = 2, the causal mask is:
        0 0
        0 0
        0 0
        1 0
        1 1
    If the row of the mask is all zero, the output will be zero.

    If window_size != (-1, -1), implements sliding window local attention. Query at position i
    will only attend to keys between
    [i + seqlen_k - seqlen_q - window_size[0], i + seqlen_k - seqlen_q + window_size[1]] inclusive.

    Arguments:
        q: (total_q, nheads, headdim), where total_q = total number of query tokens in the batch.
        kv: (total_k, 2, nheads_k, headdim), where total_k = total number of key tokens in the batch.
        cu_seqlens_q: (batch_size + 1,), dtype torch.int32. The cumulative sequence lengths
           of the sequences in the batch, used to index into q.
        cu_seqlens_k: (batch_size + 1,), dtype torch.int32. The cumulative sequence lengths
           of the sequences in the batch, used to index into kv.
        max_seqlen_q: int. Maximum query sequence length in the batch.
        max_seqlen_k: int. Maximum key sequence length in the batch.
        dropout_p: float. Dropout probability.
        softmax_scale: float. The scaling of QK^T before applying softmax.
            Default to 1 / sqrt(headdim).
        causal: bool. Whether to apply causal attention mask (e.g., for auto-regressive modeling).
        window_size: (left, right). If not (-1, -1), implements sliding window local attention.
        softcap: float. Anything > 0 activates softcapping attention.
        alibi_slopes: (nheads,) or (batch_size, nheads), fp32. A bias of
            (-alibi_slope * |i + seqlen_k - seqlen_q - j|)
            is added to the attention score of query i and key j.
        deterministic: bool. Whether to use the deterministic implementation of the backward pass,
            which is slightly slower and uses more memory. The forward pass is always deterministic.
        return_attn_probs: bool. Whether to return the attention probabilities. This option is for
           testing only. The returned probabilities are not guaranteed to be correct
           (they might not have the right scaling).
    Return:
        out: (total, nheads, headdim).
        softmax_lse [optional, if return_attn_probs=True]: (nheads, total_q_seqlen). The
            logsumexp of each row of the matrix QK^T * scaling (e.g., log of the softmax
            normalization factor).
        S_dmask [optional, if return_attn_probs=True]: (batch_size, nheads, seqlen, seqlen).
            The output of softmax (possibly with different scaling). It also encodes the dropout
            pattern (negative means that location was dropped, nonnegative means it was kept).
    """
    return FlashAttnVarlenKVPackedFunc.apply(
        q,
        kv,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        dropout_p,
        softmax_scale,
        causal,
        window_size,
        softcap,
        alibi_slopes,
        deterministic,
        return_attn_probs,
    )

# This interface references vLLM flash attention and is compatible with both FA2 and FA3. 
def flash_attn_varlen_func(
    q,
    k,
    v,
    cu_seqlens_q=None,
    cu_seqlens_k=None,# only used for non-paged prefill
    max_seqlen_q=0,
    max_seqlen_k=0,
    dropout_p=0.0,
    seqused_k=None,
    q_v=None,
    softmax_scale=None,
    causal=False,
    window_size: Optional[List[int]] = (-1, -1),
    softcap=0.0, # 0.0 means deactivated
    alibi_slopes=None,
    deterministic=False,
    return_attn_probs=False,
    block_table=None,
    return_softmax_lse=False,
    out=None,
    layout="bshd",# bhsd,bshd,custom. The bhsd layout is recommended
    # FA3 Only
    scheduler_metadata=None,
    q_descale=None,
    k_descale=None,
    v_descale=None,
    num_splits: int = 0,
    # Version selector
    fa_version: int = DEFAULT_FA_VERSION,
    s_aux=None,
    cp_world_size=1,
    cp_rank=0,
    cp_tot_seqused_k=None,
):
    """dropout_p should be set to 0.0 during evaluation
    Supports multi-query and grouped-query attention (MQA/GQA) by passing in K, V with fewer heads
    than Q. Note that the number of heads in Q must be divisible by the number of heads in KV.
    For example, if Q has 6 heads and K, V have 2 heads, head 0, 1, 2 of Q will attention to head
    0 of K, V, and head 3, 4, 5 of Q will attention to head 1 of K, V.

    If causal=True, the causal mask is aligned to the bottom right corner of the attention matrix.
    For example, if seqlen_q = 2 and seqlen_k = 5, the causal mask (1 = keep, 0 = masked out) is:
        1 1 1 1 0
        1 1 1 1 1
    If seqlen_q = 5 and seqlen_k = 2, the causal mask is:
        0 0
        0 0
        0 0
        1 0
        1 1
    If the row of the mask is all zero, the output will be zero.

    If window_size != (-1, -1), implements sliding window local attention. Query at position i
    will only attend to keys between
    [i + seqlen_k - seqlen_q - window_size[0], i + seqlen_k - seqlen_q + window_size[1]] inclusive.

    Arguments:
        q: (total_q, nheads, headdim), where total_q = total number of query tokens in the batch.
        k: (total_k, nheads_k, headdim), where total_k = total number of key tokens in the batch.
        v: (total_k, nheads_k, headdim), where total_k = total number of key tokens in the batch.
        cu_seqlens_q: (batch_size + 1,), dtype torch.int32. The cumulative sequence lengths
           of the sequences in the batch, used to index into q.
        cu_seqlens_k: (batch_size + 1,), dtype torch.int32. The cumulative sequence lengths
           of the sequences in the batch, used to index into kv.
        max_seqlen_q: int. Maximum query sequence length in the batch.
        max_seqlen_k: int. Maximum key sequence length in the batch.
        dropout_p: float. Dropout probability.
        softmax_scale: float. The scaling of QK^T before applying softmax.
            Default to 1 / sqrt(headdim).
        causal: bool. Whether to apply causal attention mask (e.g., for auto-regressive modeling).
        window_size: (left, right). If not (-1, -1), implements sliding window local attention.
        softcap: float. Anything > 0 activates softcapping attention.
        alibi_slopes: (nheads,) or (batch_size, nheads), fp32. A bias of
            (-alibi_slope * |i + seqlen_k - seqlen_q - j|)
            is added to the attention score of query i and key j.
        deterministic: bool. Whether to use the deterministic implementation of the backward pass,
            which is slightly slower and uses more memory. The forward pass is always deterministic.
        return_attn_probs: bool. Whether to return the attention probabilities. This option is for
           testing only. The returned probabilities are not guaranteed to be correct
           (they might not have the right scaling).
    Return:
        out: (total, nheads, headdim).
        softmax_lse [optional, if return_attn_probs=True]: (nheads, total_q_seqlen). The
            logsumexp of each row of the matrix QK^T * scaling (e.g., log of the softmax
            normalization factor).
        S_dmask [optional, if return_attn_probs=True]: (batch_size, nheads, seqlen, seqlen).
            The output of softmax (possibly with different scaling). It also encodes the dropout
            pattern (negative means that location was dropped, nonnegative means it was kept).
    """
    if block_table is None:
        return FlashAttnVarlenFunc.apply(
            q,
            k,
            v,
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q,
            max_seqlen_k,
            dropout_p,
            softmax_scale,
            causal,
            window_size,
            softcap,
            alibi_slopes,
            deterministic,
            return_attn_probs,
            q_descale, k_descale, v_descale,
            block_table,
            s_aux,
            torch.is_grad_enabled()
        )
    if layout=='custom':
        pa_layout = 0 
        block_size = k.shape[-2]
        head_dim_v = v.shape[-2]
        nheads_kv = k.shape[1]
    elif layout == "bhsd":
        pa_layout = 1
        block_size = k.shape[-2]
        head_dim_v = v.shape[-1]
        nheads_kv = k.shape[1]
    else :
        pa_layout = 2
        block_size = k.shape[1]
        head_dim_v = v.shape[-1]
        nheads_kv = k.shape[-2]
    head_dim_qk = q.shape[-1]
    nheads_q = q.shape[-2]
    assert head_dim_qk %64 == 0 and head_dim_qk<=512, f"UnSupport q.shape[-1]: {head_dim_qk}"
    assert not (layout != "bshd" and head_dim_qk == 192 and head_dim_v == 128),f"Only support bshd for qkdim 192 vdim 128"
    if softmax_scale is None:
        softmax_scale = q.shape[-1] ** (-0.5)
    dummy_cu_seqlens_k = torch.empty_like(cu_seqlens_q)
    real_window_size: Tuple[int, int]
    if window_size is None:
        real_window_size = (-1, -1)
    else:
        assert len(window_size) == 2
        real_window_size = (window_size[0], window_size[1])
    bs = cu_seqlens_q.shape[0] - 1
    total_q = q.shape[0]
    # max_seqlen_q*bs==total_q and max_seqlen_q<=4 means mtp
    is_mtp = (max_seqlen_q*bs==total_q and max_seqlen_q>1 and max_seqlen_q<16 and max_seqlen_q*nheads_q/nheads_kv<=64)
    gcn_arch_name = torch.cuda.get_device_properties("cuda").gcnArchName
    is_gfx936 = "gfx936" in gcn_arch_name
    is_gfx938 = "gfx938" in gcn_arch_name
    use_mixed_pa_192_128 = (
        is_gfx936
        and is_mtp
        and window_size == (-1, -1)
        and (causal or max_seqlen_q == 1)
        and layout == "bshd"
        and q.dtype in (torch.bfloat16, torch.float8_e5m2)
        and k.dtype == torch.float8_e5m2
        and head_dim_qk == 192
        and head_dim_v == 128
        and block_size in (64, 128)
        and nheads_q * max_seqlen_q <= nheads_kv * 64
    )
    # Keep the existing prefix decode/prefill boundary.  This path only
    # replaces prefix prefill; prefix decode remains on its original backend.
    is_prefix_prefill = max_seqlen_q >= 16
    use_b16_d192_contiguous_prefill = (
        (is_gfx936 or is_gfx938)
        and is_prefix_prefill
        and block_size in (64, 128)
        and head_dim_qk == 192
        and (head_dim_v == 192 or (is_gfx936 and head_dim_v == 128 and layout == "bshd"))
        and q.dtype in (torch.float16, torch.bfloat16)
        and k.dtype == q.dtype
        and v.dtype == q.dtype
        and dropout_p == 0.0
        and softcap == 0.0
        and alibi_slopes is None
        and q_descale is None
        and k_descale is None
        and v_descale is None
    )
    use_fp8_d192_d128_contiguous_prefill = (
        is_gfx938
        and is_prefix_prefill
        and causal
        and real_window_size[0] < 0
        and max_seqlen_k > 0
        and block_size in (64, 128)
        and layout == "bshd"
        and head_dim_qk == 192
        and head_dim_v == 128
        and q.dtype == torch.float8_e4m3fn
        and k.dtype == q.dtype
        and v.dtype == q.dtype
        and dropout_p == 0.0
        and softcap == 0.0
        and alibi_slopes is None
        # The existing HG varlen interface limits attention sinks to 64 heads.
        and (s_aux is None or nheads_q <= 64)
        and not return_attn_probs
        and (
            (q_descale is None and k_descale is None and v_descale is None)
            or all(s is not None for s in (q_descale, k_descale, v_descale))
        )
    )
    use_direct_paged = (
        (max_seqlen_q == 1 or is_mtp)
        and block_size == 64
        and head_dim_qk == head_dim_v
    ) or use_mixed_pa_192_128
    if use_direct_paged:
        if out==None:
            if q.dtype == torch.float8_e4m3fn or q.dtype == torch.float8_e5m2:
                out = torch.empty(total_q, nheads_q, head_dim_v, device=q.device, dtype=torch.bfloat16)
            else :
                out = torch.empty(total_q, nheads_q, head_dim_v, device=q.device, dtype=q.dtype)
        if q.dtype == torch.float8_e5m2:
            q=q.to(out.dtype)
        if pa_layout==2 and  nheads_kv==1 and not use_mixed_pa_192_128:# bshd且khead=1时的特殊优化
            flash_attn_cuda.paged_attention(out,q.reshape(bs,max_seqlen_q,nheads_q,head_dim_qk),k.permute(0, 2, 1, 3),v.permute(0, 2, 1, 3),
                                            softmax_scale,block_table,seqused_k,alibi_slopes,"",q_descale,k_descale,v_descale,max_seqlen_k,s_aux,1,
                                            real_window_size[0],real_window_size[1],causal)
        else:
            flash_attn_cuda.paged_attention(out,q.reshape(bs,max_seqlen_q,nheads_q,head_dim_qk),k,v,softmax_scale,block_table,
                seqused_k,alibi_slopes,"",q_descale,k_descale,v_descale,max_seqlen_k,s_aux,pa_layout,
                real_window_size[0],real_window_size[1],causal)
        return out
    use_kv_contiguous = (
        block_size == 64 and head_dim_qk == head_dim_v
    ) or use_b16_d192_contiguous_prefill or use_fp8_d192_d128_contiguous_prefill or (head_dim_qk==512 and block_size==32)
    if use_kv_contiguous:
        kv_dtype = k.dtype
        if k.dtype == torch.float8_e5m2:
            if out!=None:
                kv_dtype = out.dtype
            elif q.dtype==torch.float8_e5m2:
                kv_dtype = torch.bfloat16
            else :
                kv_dtype = q.dtype
        q=q.to(kv_dtype)
        total_k = 0
        if not torch.cuda.is_current_stream_capturing():
            total_k = seqused_k.sum()
        if total_k < 256*1024 : #total_k不设置最小值，Qwen2.5 7B 会出现core dump，是崩溃在框架侧，而不是fa中，原因不明
            total_k = 256*1024 
        kt = torch.zeros([total_k,nheads_kv,head_dim_qk],device = q.device,dtype=kv_dtype)
        vt = (torch.zeros_like(kt) if head_dim_qk == head_dim_v else
              torch.zeros([total_k, nheads_kv, head_dim_v], device=q.device, dtype=kv_dtype))
        cu_seqlens_k = torch.empty_like(cu_seqlens_q)
        flash_attn_cuda.pagedkv_to_contiguouskv(kt,vt,k,v,block_table,seqused_k,cu_seqlens_k,max_seqlen_k,pa_layout)
        k = kt
        v = vt
        block_table = None
        seqused_k = None
    q=q.to(k.dtype)
    out, q, k, v, out_padded, softmax_lse, S_dmask, rng_state = flash_attn_cuda.varlen_fwd(
        q,
        k,
        v,
        out,
        cu_seqlens_q,
        # cu_seqlens_k not used since we use seqused_k, but flash_api.cpp
        # still wants it so we pass all zeros
        dummy_cu_seqlens_k if cu_seqlens_k is None else cu_seqlens_k,
        seqused_k,
        #leftpad_k,
        None,
        block_table,
        alibi_slopes,
        max_seqlen_q,
        max_seqlen_k,
        dropout_p,
        softmax_scale,
        False,
        causal,
        real_window_size[0],
        real_window_size[1],
        softcap,
        False,
        q_descale,
        k_descale,
        v_descale,
        None,
        s_aux,
    )
    return out if not return_softmax_lse else (out, softmax_lse, S_dmask)

hg_flash_attn_varlen_func = flash_attn_varlen_func
varlen_fwd_unified = flash_attn_varlen_func

def vllm_flash_attn_varlen_func(
    q,
    k,
    v,
    max_seqlen_q,
    cu_seqlens_q,
    max_seqlen_k,
    cu_seqlens_k=None, # only used for non-paged prefill
    seqused_k=None,
    q_v=None,
    dropout_p=0.0,
    softmax_scale=None,
    causal=False,
    window_size: Optional[List[int]] = None,
    softcap=0.0, # 0.0 means deactivated
    alibi_slopes=None,
    deterministic=False,
    return_attn_probs=False,
    block_table=None,
    return_softmax_lse=False,
    out=None,
    is_prefix_cache = True,
    # FA3 Only
    scheduler_metadata=None,
    q_descale=None,
    k_descale=None,
    v_descale=None,
    kv_cache_dtype = "auto",
    # Version selector
    fa_version: int = DEFAULT_FA_VERSION,
    s_aux=None
):
    return flash_attn_varlen_func(
            q,
            k,
            v,
            max_seqlen_q=max_seqlen_q,
            cu_seqlens_q=cu_seqlens_q,
            max_seqlen_k=max_seqlen_k,
            cu_seqlens_k=cu_seqlens_k, # only used for non-paged prefill
            seqused_k=seqused_k,
            q_v=q_v,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            causal=causal,
            window_size=window_size,
            softcap=softcap, # 0.0 means deactivated
            alibi_slopes=alibi_slopes,
            deterministic=deterministic,
            return_attn_probs=return_attn_probs,
            block_table=block_table,
            return_softmax_lse=return_softmax_lse,
            out=out,
            layout="custom",# bhsd,bshd,custom. The bhsd layout is recommended
            # FA3 Only
            scheduler_metadata=scheduler_metadata,
            q_descale=q_descale,
            k_descale=k_descale,
            v_descale=v_descale,
            num_splits = 0,
            # Version selector
            fa_version = fa_version,
            s_aux=s_aux,
            cp_world_size=1,
            cp_rank=0,
            cp_tot_seqused_k=None,
        )
    
def flash_attn_with_kvcache(
    q,
    k_cache,
    v_cache,
    k=None,
    v=None,
    rotary_cos=None,
    rotary_sin=None,
    cache_seqlens: Optional[Union[(int, torch.Tensor)]] = None,
    cache_batch_idx: Optional[torch.Tensor] = None,
    cache_leftpad: Optional[torch.Tensor] = None,
    block_table: Optional[torch.Tensor] = None,
    softmax_scale=None,
    causal=False,
    window_size=(-1, -1),  # -1 means infinite context window
    softcap=0.0, # 0.0 means deactivated
    rotary_interleaved=True,
    alibi_slopes=None,
    num_splits=0,
    s_aux=None,  # Attention Sinks: [nheads]
    return_softmax_lse=False,
    *,

    out = None
):
    """
    If k and v are not None, k_cache and v_cache will be updated *inplace* with the new values from
    k and v. This is useful for incremental decoding: you can pass in the cached keys/values from
    the previous step, and update them with the new keys/values from the current step, and do
    attention with the updated cache, all in 1 kernel.

    If you pass in k / v, you must make sure that the cache is large enough to hold the new values.
    For example, the KV cache could be pre-allocated with the max sequence length, and you can use
    cache_seqlens to keep track of the current sequence lengths of each sequence in the batch.

    Also apply rotary embedding if rotary_cos and rotary_sin are passed in. The key @k will be
    rotated by rotary_cos and rotary_sin at indices cache_seqlens, cache_seqlens + 1, etc.
    If causal or local (i.e., window_size != (-1, -1)), the query @q will be rotated by rotary_cos
    and rotary_sin at indices cache_seqlens, cache_seqlens + 1, etc.
    If not causal and not local, the query @q will be rotated by rotary_cos and rotary_sin at
    indices cache_seqlens only (i.e. we consider all tokens in @q to be at position cache_seqlens).

    See tests/test_flash_attn.py::test_flash_attn_kvcache for examples of how to use this function.

    Supports multi-query and grouped-query attention (MQA/GQA) by passing in KV with fewer heads
    than Q. Note that the number of heads in Q must be divisible by the number of heads in KV.
    For example, if Q has 6 heads and K, V have 2 heads, head 0, 1, 2 of Q will attention to head
    0 of K, V, and head 3, 4, 5 of Q will attention to head 1 of K, V.

    If causal=True, the causal mask is aligned to the bottom right corner of the attention matrix.
    For example, if seqlen_q = 2 and seqlen_k = 5, the causal mask (1 = keep, 0 = masked out) is:
        1 1 1 1 0
        1 1 1 1 1
    If seqlen_q = 5 and seqlen_k = 2, the causal mask is:
        0 0
        0 0
        0 0
        1 0
        1 1
    If the row of the mask is all zero, the output will be zero.

    If window_size != (-1, -1), implements sliding window local attention. Query at position i
    will only attend to keys between
    [i + seqlen_k - seqlen_q - window_size[0], i + seqlen_k - seqlen_q + window_size[1]] inclusive.

    Note: Does not support backward pass.

    Arguments:
        q: (batch_size, seqlen, nheads, headdim)
        k_cache: (batch_size_cache, seqlen_cache, nheads_k, headdim) if there's no block_table,
            or (num_blocks, page_block_size, nheads_k, headdim) if there's a block_table (i.e. paged KV cache)
            page_block_size must be a multiple of 256.
        v_cache: (batch_size_cache, seqlen_cache, nheads_k, headdim) if there's no block_table,
            or (num_blocks, page_block_size, nheads_k, headdim) if there's a block_table (i.e. paged KV cache)
        k [optional]: (batch_size, seqlen_new, nheads_k, headdim). If not None, we concatenate
            k with k_cache, starting at the indices specified by cache_seqlens.
        v [optional]: (batch_size, seqlen_new, nheads_k, headdim). Similar to k.
        rotary_cos [optional]: (seqlen_ro, rotary_dim / 2). If not None, we apply rotary embedding
            to k and q. Only applicable if k and v are passed in. rotary_dim must be divisible by 16.
        rotary_sin [optional]: (seqlen_ro, rotary_dim / 2). Similar to rotary_cos.
        cache_seqlens: int, or (batch_size,), dtype torch.int32. The sequence lengths of the
            KV cache.
        cache_batch_idx: (batch_size,), dtype torch.int32. The indices used to index into the KV cache.
            If None, we assume that the batch indices are [0, 1, 2, ..., batch_size - 1].
            If the indices are not distinct, and k and v are provided, the values updated in the cache
                 might come from any of the duplicate indices.
        cache_leftpad: (batch_size,), dtype torch.int32. The index that the KV cache starts. If None, assume 0.
        block_table [optional]: (batch_size, max_num_blocks_per_seq), dtype torch.int32.
        softmax_scale: float. The scaling of QK^T before applying softmax.
            Default to 1 / sqrt(headdim).
        causal: bool. Whether to apply causal attention mask (e.g., for auto-regressive modeling).
        window_size: (left, right). If not (-1, -1), implements sliding window local attention.
        softcap: float. Anything > 0 activates softcapping attention.
        rotary_interleaved: bool. Only applicable if rotary_cos and rotary_sin are passed in.
            If True, rotary embedding will combine dimensions 0 & 1, 2 & 3, etc. If False,
            rotary embedding will combine dimensions 0 & rotary_dim / 2, 1 & rotary_dim / 2 + 1
            (i.e. GPT-NeoX style).
        alibi_slopes: (nheads,) or (batch_size, nheads), fp32. A bias of
            (-alibi_slope * |i + seqlen_k - seqlen_q - j|)
            is added to the attention score of query i and key j.
        num_splits: int. If > 1, split the key/value into this many chunks along the sequence.
           If num_splits == 1, we don't split the key/value. If num_splits == 0, we use a heuristic
           to automatically determine the number of splits.
           Don't change this unless you know what you are doing.
        return_softmax_lse: bool. Whether to return the logsumexp of the attention scores.

    Return:
        out: (batch_size, seqlen, nheads, headdim).
        softmax_lse [optional, if return_softmax_lse=True]: (batch_size, nheads, seqlen). The
            logsumexp of each row of the matrix QK^T * scaling (e.g., log of the softmax
            normalization factor).
    """
    assert k_cache.stride(-1) == 1, "k_cache must have contiguous last dimension"
    assert v_cache.stride(-1) == 1, "v_cache must have contiguous last dimension"
    q, k, v = [maybe_contiguous(x) for x in (q, k, v)]
    s_aux = maybe_contiguous(s_aux)
    if softmax_scale is None:
        softmax_scale = q.shape[-1] ** (-0.5)
    if cache_seqlens is not None and isinstance(cache_seqlens, int):
        cache_seqlens = torch.full(
            (k_cache.shape[0],), cache_seqlens, dtype=torch.int32, device=k_cache.device
        )
        cache_seqlens = maybe_contiguous(cache_seqlens)
    cache_batch_idx = maybe_contiguous(cache_batch_idx)
    block_table = maybe_contiguous(block_table)
    out, softmax_lse = flash_attn_cuda.fwd_kvcache(
        q,
        k_cache,
        v_cache,
        k,
        v,
        cache_seqlens,
        rotary_cos,
        rotary_sin,
        cache_batch_idx,
        cache_leftpad,
        block_table,
        alibi_slopes,
        out,
        softmax_scale,
        causal,
        window_size[0],
        window_size[1],
        softcap,
        rotary_interleaved,
        num_splits,
        s_aux,  # Attention Sinks
    )
    return (out, softmax_lse) if return_softmax_lse else out


def vllm_flash_attn_with_kvcache(
    q,
    k_cache,
    v_cache,
    k=None,
    v=None,
    rotary_cos=None,
    rotary_sin=None,
    cache_seqlens: Optional[Union[(int, torch.Tensor)]] = None,
    cache_batch_idx: Optional[torch.Tensor] = None,
    cache_leftpad: Optional[torch.Tensor] = None,
    block_table: Optional[torch.Tensor] = None,
    softmax_scale=None,
    causal=False,
    window_size=(-1, -1),  # -1 means infinite context window
    softcap=0.0, # 0.0 means deactivated
    rotary_interleaved=True,
    alibi_slopes=None,
    num_splits=0,
    s_aux=None,  # Attention Sinks: [nheads]
    return_softmax_lse=False,
    *,
    q_scale=None,
    k_scale=None,
    v_scale=None,
    kv_cache_dtype = "auto",
    out = None,
    max_seqlen_k = 0,
    layout = 0
):
    """
    If k and v are not None, k_cache and v_cache will be updated *inplace* with the new values from
    k and v. This is useful for incremental decoding: you can pass in the cached keys/values from
    the previous step, and update them with the new keys/values from the current step, and do
    attention with the updated cache, all in 1 kernel.

    If you pass in k / v, you must make sure that the cache is large enough to hold the new values.
    For example, the KV cache could be pre-allocated with the max sequence length, and you can use
    cache_seqlens to keep track of the current sequence lengths of each sequence in the batch.

    Also apply rotary embedding if rotary_cos and rotary_sin are passed in. The key @k will be
    rotated by rotary_cos and rotary_sin at indices cache_seqlens, cache_seqlens + 1, etc.
    If causal or local (i.e., window_size != (-1, -1)), the query @q will be rotated by rotary_cos
    and rotary_sin at indices cache_seqlens, cache_seqlens + 1, etc.
    If not causal and not local, the query @q will be rotated by rotary_cos and rotary_sin at
    indices cache_seqlens only (i.e. we consider all tokens in @q to be at position cache_seqlens).

    See tests/test_flash_attn.py::test_flash_attn_kvcache for examples of how to use this function.

    Supports multi-query and grouped-query attention (MQA/GQA) by passing in KV with fewer heads
    than Q. Note that the number of heads in Q must be divisible by the number of heads in KV.
    For example, if Q has 6 heads and K, V have 2 heads, head 0, 1, 2 of Q will attention to head
    0 of K, V, and head 3, 4, 5 of Q will attention to head 1 of K, V.

    If causal=True, the causal mask is aligned to the bottom right corner of the attention matrix.
    For example, if seqlen_q = 2 and seqlen_k = 5, the causal mask (1 = keep, 0 = masked out) is:
        1 1 1 1 0
        1 1 1 1 1
    If seqlen_q = 5 and seqlen_k = 2, the causal mask is:
        0 0
        0 0
        0 0
        1 0
        1 1
    If the row of the mask is all zero, the output will be zero.

    If window_size != (-1, -1), implements sliding window local attention. Query at position i
    will only attend to keys between
    [i + seqlen_k - seqlen_q - window_size[0], i + seqlen_k - seqlen_q + window_size[1]] inclusive.

    Note: Does not support backward pass.

    Arguments:
        q: (batch_size, seqlen, nheads, headdim)
        k_cache: (batch_size_cache, seqlen_cache, nheads_k, headdim) if there's no block_table,
            or (num_blocks, page_block_size, nheads_k, headdim) if there's a block_table (i.e. paged KV cache)
            page_block_size must be a multiple of 256.
        v_cache: (batch_size_cache, seqlen_cache, nheads_k, headdim) if there's no block_table,
            or (num_blocks, page_block_size, nheads_k, headdim) if there's a block_table (i.e. paged KV cache)
        k [optional]: (batch_size, seqlen_new, nheads_k, headdim). If not None, we concatenate
            k with k_cache, starting at the indices specified by cache_seqlens.
        v [optional]: (batch_size, seqlen_new, nheads_k, headdim). Similar to k.
        rotary_cos [optional]: (seqlen_ro, rotary_dim / 2). If not None, we apply rotary embedding
            to k and q. Only applicable if k and v are passed in. rotary_dim must be divisible by 16.
        rotary_sin [optional]: (seqlen_ro, rotary_dim / 2). Similar to rotary_cos.
        cache_seqlens: int, or (batch_size,), dtype torch.int32. The sequence lengths of the
            KV cache.
        cache_batch_idx: (batch_size,), dtype torch.int32. The indices used to index into the KV cache.
            If None, we assume that the batch indices are [0, 1, 2, ..., batch_size - 1].
            If the indices are not distinct, and k and v are provided, the values updated in the cache
                 might come from any of the duplicate indices.
        cache_leftpad: (batch_size,), dtype torch.int32. The index that the KV cache starts. If None, assume 0.
        block_table [optional]: (batch_size, max_num_blocks_per_seq), dtype torch.int32.
        softmax_scale: float. The scaling of QK^T before applying softmax.
            Default to 1 / sqrt(headdim).
        causal: bool. Whether to apply causal attention mask (e.g., for auto-regressive modeling).
        window_size: (left, right). If not (-1, -1), implements sliding window local attention.
        softcap: float. Anything > 0 activates softcapping attention.
        rotary_interleaved: bool. Only applicable if rotary_cos and rotary_sin are passed in.
            If True, rotary embedding will combine dimensions 0 & 1, 2 & 3, etc. If False,
            rotary embedding will combine dimensions 0 & rotary_dim / 2, 1 & rotary_dim / 2 + 1
            (i.e. GPT-NeoX style).
        alibi_slopes: (nheads,) or (batch_size, nheads), fp32. A bias of
            (-alibi_slope * |i + seqlen_k - seqlen_q - j|)
            is added to the attention score of query i and key j.
        num_splits: int. If > 1, split the key/value into this many chunks along the sequence.
           If num_splits == 1, we don't split the key/value. If num_splits == 0, we use a heuristic
           to automatically determine the number of splits.
           Don't change this unless you know what you are doing.
        return_softmax_lse: bool. Whether to return the logsumexp of the attention scores.

    Return:
        out: (batch_size, seqlen, nheads, headdim).
        softmax_lse [optional, if return_softmax_lse=True]: (batch_size, nheads, seqlen). The
            logsumexp of each row of the matrix QK^T * scaling (e.g., log of the softmax
            normalization factor).
    """
    assert k_cache.stride(-1) == 1, "k_cache must have contiguous last dimension"
    assert v_cache.stride(-1) == 1, "v_cache must have contiguous last dimension"
    q, k, v = [maybe_contiguous(x) for x in (q, k, v)]
    if softmax_scale is None:
        softmax_scale = q.shape[-1] ** (-0.5)
    if cache_seqlens is not None and isinstance(cache_seqlens, int):
        cache_seqlens = torch.full(
            (k_cache.shape[0],), cache_seqlens, dtype=torch.int32, device=k_cache.device
        )
        cache_seqlens = maybe_contiguous(cache_seqlens)
    cache_batch_idx = maybe_contiguous(cache_batch_idx)
    block_table = maybe_contiguous(block_table)
    assert q.shape[-1] %64 == 0 and q.shape[-1]<=256, f"UnSupport q.shape[-1]: {q.shape[-1]}"
    if out==None:
        if q.dtype == torch.float8_e4m3fn or q.dtype == torch.float8_e5m2:
            out = torch.empty(q.size(),device = q.device,dtype=torch.bfloat16)
        else :
            out = torch.empty_like(q)
    flash_attn_cuda.paged_attention(out,q,k_cache,v_cache,softmax_scale,block_table,
        cache_seqlens,alibi_slopes,kv_cache_dtype,q_scale,k_scale,v_scale,max_seqlen_k,s_aux,layout,
        window_size[0],window_size[1],causal)
    return out


def sparse_attn_func(
    q,
    k,
    v,
    block_count,
    block_offset,
    column_count,
    column_index,
    dropout_p=0.0,
    softmax_scale=None,
    causal=False,
    softcap=0.0, # 0.0 means deactivated
    alibi_slopes=None,
    deterministic=False,
    return_attn_probs=False,
    *,
    return_softmax_lse=False,
    out=None,
    pv_threshold=50.0,         # Dynamic PV skip threshold
    enable_dynamic_skip=False, # Enable dynamic PV skip optimization (disabled by default for dense scenarios)
    is_sla=False
):
    """Compute attention with vertical and slash sparsity patterns.
    Most Arguments are the same with the flash_attn_func interface, except for 4 extra args:
    block_count and block_offset for slash sparsity patterns, and
    column_count and column_index for vertical sparsity patterns.
    For more details please refer to Appendix C.4.2 of paper https://arxiv.org/abs/2407.02490.

    Arguments:
        q: (batch_size, seqlen, nheads, headdim)
        k: (batch_size, seqlen, nheads_k, headdim)
        v: (batch_size, seqlen, nheads_k, headdim)
        block_count: (batch_size, nheads, cdiv(seqlen, BLOCK_M))
        block_offset: (batch_size, nheads, cdiv(seqlen, BLOCK_M), NNZ_S)
        column_count: (batch_size, nheads, cdiv(seqlen, BLOCK_M))
        column_index: (batch_size, nheads, cdiv(seqlen, BLOCK_M), NNZ_V)
        dropout_p: float. Dropout probability.
        softmax_scale: float. The scaling of QK^T before applying softmax.
            Default to 1 / sqrt(headdim).
        causal: bool. Whether to apply causal attention mask (e.g., for auto-regressive modeling).
        alibi_slopes: (nheads,) or (batch_size, nheads), fp32. A bias of
            (-alibi_slope * |i + seqlen_k - seqlen_q - j|)
            is added to the attention score of query i and key j.
        deterministic: bool. Whether to use the deterministic implementation of the backward pass,
            which is slightly slower and uses more memory. The forward pass is always deterministic.
        return_attn_probs: bool. Whether to return the attention probabilities. This option is for
           testing only. The returned probabilities are not guaranteed to be correct
           (they might not have the right scaling).
    Return:
        out: (batch_size, seqlen, nheads, headdim).
        softmax_lse [optional, if return_softmax_lse=True]: (batch_size, nheads, seqlen). The
            logsumexp of each row of the matrix QK^T * scaling (e.g., log of the softmax
            normalization factor).
    """
    if softmax_scale is None:
        softmax_scale = q.shape[-1] ** (-0.5)
    q, k, v = [maybe_contiguous(x) for x in (q, k, v)]
    out, softmax_lse = flash_attn_cuda.fwd_sparse(
        q,
        k,
        v,
        block_count,
        block_offset,
        column_count,
        column_index,
        out,
        alibi_slopes,
        dropout_p,
        softmax_scale,
        causal,
        softcap,
        return_attn_probs and dropout_p > 0,
        None,
        is_sla,
        pv_threshold,
        enable_dynamic_skip,
    )
    return (out, softmax_lse) if return_softmax_lse else out

def _prepare_sparse_gqa_metadata(indices, cu_q, cu_k, kv_lens, max_seqlen_q):
    metadata = (indices, cu_q, cu_k, kv_lens)
    if any(t.dtype not in (torch.int32, torch.int64) for t in metadata):
        raise TypeError("indices and sequence metadata must be int32 or int64")
    if cu_q.ndim != 1 or cu_k.ndim != 1 or kv_lens.ndim != 1:
        raise ValueError("cu_q, cu_k and kv_lens must be one-dimensional")
    if max_seqlen_q is None:
        # Empty query batches are a supported extension of the target wrapper.
        max_seqlen_q = (max(1, int((cu_q[1:] - cu_q[:-1]).max().item()))
                       if kv_lens.numel() and cu_q.numel() > 1 else 1)
    # Valid metadata must fit int32 and describe in-bounds packed segments.
    # Avoid a value-checking host synchronization on the explicit-max/graph path.
    indices, cu_q, cu_k, kv_lens = [t.to(dtype=torch.int32).contiguous() for t in metadata]
    return indices, cu_q, cu_k, kv_lens, max_seqlen_q


def fa_sparse_attention_prefill(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    packed_regions: torch.Tensor,
    region_counts: torch.Tensor,
    *,
    num_kv_groups: int,
    region_size: int = 8,
    softmax_scale: Optional[float] = None,
    block_regions: int = 16,
    logical_regions: Optional[torch.Tensor] = None,
    q_positions: Optional[torch.Tensor] = None,
    request_indices: Optional[torch.Tensor] = None,
    block_q: int = 128,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """BF16 Stage-5 region attention, with the standard Triton call signature.

    Q is [M,H,D], K/V [tokens,G,D], packed int32 [G*M,topk] and counts
    int32 [G*M]. Packing is physical_region | (valid_tokens << 24).
    Returns BF16 output and FP32 natural-log LSE; empty rows are (0,-inf).
    Requires D=192; supports 1..16 heads/group and vector-aligned rows.
    The optional query-tiled path is not part of this Stage-5/6 port.
    block_regions is a compatibility hint; the FA kernel uses a fixed tile.
    """
    if logical_regions is not None:
        raise NotImplementedError("query-tiled sparse prefill is not included in the Stage-5/6 port")
    if block_regions <= 0:
        raise ValueError("block_regions must be positive")
    out, lse = flash_attn_cuda.fa_sparse_attention_prefill(
        query, key_cache, value_cache, packed_regions, region_counts,
        num_kv_groups, region_size, softmax_scale,
    )
    return out, lse


def fa_sparse_attention_decode(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    packed_regions: torch.Tensor,
    region_counts: torch.Tensor,
    kv_seqlens: torch.Tensor,
    *,
    num_kv_groups: int,
    region_size: int = 8,
    softmax_scale: Optional[float] = None,
    num_splits: int = 1,
    block_regions: int = 16,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Stage-6 decode with BF16 Q/K/V and output.

    num_splits is accepted for call compatibility but ignored. Internal parts
    are selected and merged before returning final BF16 [B,H,192] output and
    FP32 natural-log [B,H] LSE. No public partial states are returned.
    Requires D=192; other shape/alignment rules match fa_sparse_attention_prefill.
    Empty states are (0,-inf). block_regions is accepted as a compatibility tuning hint.
    """
    if block_regions <= 0:
        raise ValueError("block_regions must be positive")
    # The C++ entry point also enforces final-state output for direct callers.
    num_splits = 1
    out, lse = flash_attn_cuda.fa_sparse_attention_decode(
        query, key_cache, value_cache, packed_regions, region_counts, kv_seqlens,
        num_kv_groups, region_size, softmax_scale, num_splits,
    )
    return out, lse


def fa_sparse_attention_prefill_fp8(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    packed_regions: torch.Tensor,
    region_counts: torch.Tensor,
    *,
    num_kv_groups: int,
    region_size: int = 8,
    softmax_scale: Optional[float] = None,
    block_regions: int = 16,
    logical_regions: Optional[torch.Tensor] = None,
    q_positions: Optional[torch.Tensor] = None,
    request_indices: Optional[torch.Tensor] = None,
    block_q: int = 128,
    k_scale: float = 1.0,
    v_scale: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Stage-5 region attention with BF16 Q/output and E4M3FN K/V.

    Q is [M,H,D], K/V [tokens,G,D], packed int32 [G*M,topk] and counts
    int32 [G*M]. Packing is physical_region | (valid_tokens << 24).
    FP8 values represent cache * k_scale/v_scale (positive finite host scalars).
    Returns BF16 output and FP32 natural-log LSE; empty rows are (0,-inf).
    Requires D=192; supports 1..16 heads/group and vector-aligned rows.
    The optional query-tiled path is not part of this Stage-5/6 port.
    block_regions is a compatibility hint; the FA kernel uses a fixed tile.
    """
    if logical_regions is not None:
        raise NotImplementedError("query-tiled sparse prefill is not included in the Stage-5/6 port")
    if block_regions <= 0:
        raise ValueError("block_regions must be positive")
    out, lse = flash_attn_cuda.fa_sparse_attention_prefill_fp8(
        query, key_cache, value_cache, packed_regions, region_counts,
        num_kv_groups, region_size, softmax_scale, k_scale, v_scale,
    )
    return out, lse


def fa_sparse_attention_decode_fp8(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    packed_regions: torch.Tensor,
    region_counts: torch.Tensor,
    kv_seqlens: torch.Tensor,
    *,
    num_kv_groups: int,
    region_size: int = 8,
    softmax_scale: Optional[float] = None,
    num_splits: int = 1,
    block_regions: int = 16,
    k_scale: float = 1.0,
    v_scale: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Stage-6 decode with BF16 Q/output and E4M3FN K/V.

    num_splits is accepted for call compatibility but ignored. Internal parts
    are selected and merged before returning final BF16 [B,H,192] output and
    FP32 natural-log [B,H] LSE. No public partial states are returned.
    Requires D=192; other shape/alignment rules match fa_sparse_attention_prefill_fp8.
    FP8 values represent cache * k_scale/v_scale (positive finite host scalars).
    Empty states are (0,-inf). block_regions is accepted as a compatibility tuning hint.
    """
    if block_regions <= 0:
        raise ValueError("block_regions must be positive")
    # The C++ entry point also enforces final-state output for direct callers.
    num_splits = 1
    out, lse = flash_attn_cuda.fa_sparse_attention_decode_fp8(
        query, key_cache, value_cache, packed_regions, region_counts, kv_seqlens,
        num_kv_groups, region_size, softmax_scale, num_splits, k_scale, v_scale,
    )
    return out, lse


def mqa_paged_score_func(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    page_table: torch.Tensor,
    token_to_req: torch.Tensor,
    query_positions: torch.Tensor,
    sequence_lengths: torch.Tensor,
    compress_ratio: int,
    num_columns: Optional[int] = None,
    score_scale: Optional[float] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """BF16 paged MQA scores and visible compressed-block counts.

    Matches QSA's qsa_mqa_paged argument order. Each head's FP32 dot product
    is rectified before summing; score_scale is a DIVISOR (default sqrt(D)).
    Returns FP32 [M, columns] logits and int32 [M] counts. Invalid scores
    are -inf. Q/K and page-table strides are preserved, not copied.
    Positions are contiguous int64; request IDs/lengths are contiguous int32.
    Zero columns return zero counts (the source leaves those uninitialized).
    """
    logits, visible_blocks = flash_attn_cuda.mqa_paged_score(
        q, k_cache, page_table, token_to_req, query_positions,
        sequence_lengths, compress_ratio, num_columns, score_scale,
    )
    return logits, visible_blocks


def sparse_gqa_paged_attn_func(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    logical_indices: torch.Tensor,
    block_table: torch.Tensor,
    token_to_req: torch.Tensor,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """BF16 sparse GQA directly over paged K/V, without repacking.

    Matches QSA's qsa_sparse_paged_attention argument order and out writeback.
    Q/out are [M,Hq,D]; caches [pages,page_size,Hkv,D]; indices [M,topk].
    Indices are request-relative, retain duplicates, and already encode the
    causal selection. Negative indices, invalid requests/pages are masked;
    empty rows return zero. Only the last dimensions must be contiguous.
    Metadata is int32. D is a power of two >=16. Scale is fixed at D**-0.5
    and applied after FP32 QK. Split-K uses FP32 partials and log2 LSE.
    No host synchronization; uses the current stream and supports graphs.
    """
    result = flash_attn_cuda.sparse_gqa_paged_attention(
        q, k_cache, v_cache, logical_indices, block_table, token_to_req, out,
    )
    return result if out is None else out


def sparse_gqa_paged_attn_fp8_func(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    logical_indices: torch.Tensor,
    block_table: torch.Tensor,
    token_to_req: torch.Tensor,
    out: Optional[torch.Tensor] = None,
    k_scale: Optional[torch.Tensor] = None,
    v_scale: Optional[torch.Tensor] = None,
    *,
    kv_cache_dtype: Optional[str] = None,
) -> torch.Tensor:
    """Paged QSA with BF16 Q/output, E4M3FN or E5M2 KV, and device scales.

    Preserves sparse_gqa_paged_attn_func metadata, strides and out identity.
    k_scale/v_scale are required same-device contiguous FP32 tensors, numel=1;
    valid finite scales are the caller/writer's responsibility. No host read.
    K scale multiplies FP32 QK; V scale multiplies normalized FP32 partials.
    Native FP8 dtype selects the format. Raw uint8 requires kv_cache_dtype
    ('fp8_e4m3' or 'fp8_e5m2'); bytes are reinterpreted, never re-quantized.
    FNUZ and per-head scales are not supported. Uses the current stream and
    reads scales on every graph replay. BF16 Q/P MMAC, not FP8 Q/P compute.
    """
    result = flash_attn_cuda.sparse_gqa_paged_attention_fp8(
        q, k_cache, v_cache, logical_indices, block_table, token_to_req,
        out, k_scale, v_scale, kv_cache_dtype,
    )
    return result if out is None else out


def sparse_gqa_attn_func(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    indices: torch.Tensor,
    cu_q: torch.Tensor,
    cu_k: torch.Tensor,
    kv_lens: torch.Tensor,
    scale: Optional[float] = None,
    k_scale: Optional[float] = None,
    v_scale: Optional[float] = None,
    *,
    max_seqlen_q: Optional[int] = None,
) -> torch.Tensor:
    """Sparse GQA chunk-prefill for matching BF16/FP16 Q, K and V.

    The argument order through ``v_scale`` matches QSA's
    ``sparse_gqa_fwd_interface_triton_ck``. As in that target, k_scale and
    v_scale are ignored for non-FP8 K/V; scale defaults to head_dim**-0.5.
    For FP8 K/V use sparse_gqa_attn_fp8_func instead.

    Supports D=256, GQA groups 1..16, and shared [total_q, topk] or per-head
    [total_q, Hkv, topk] indices. Indices are request-relative and retain
    their order/duplicates. The slot window is block-rounded like the target;
    pad unused slots with -1. Empty rows or empty leading reference blocks
    return NaN. Positive out-of-range indices are safely masked.

    Metadata may be int32/int64 but must fit int32 and describe valid packed
    segments. Noncontiguous tensors are copied. Output follows Q's dtype.
    max_seqlen_q must be at least the largest query length when supplied.
    Supply it during graph capture to avoid the default max().item() sync.
    Empty query batches return an empty output. See docs/sparse_gqa.md.
    """
    if scale is None:
        scale = q.shape[-1] ** -0.5
    indices, cu_q, cu_k, kv_lens, max_seqlen_q = _prepare_sparse_gqa_metadata(
        indices, cu_q, cu_k, kv_lens, max_seqlen_q
    )
    return flash_attn_cuda.sparse_gqa_attention(
        q.contiguous(), k.contiguous(), v.contiguous(), indices,
        cu_q, cu_k, kv_lens, float(scale), max_seqlen_q,
    )


def sparse_gqa_attn_fp8_func(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    indices: torch.Tensor,
    cu_q: torch.Tensor,
    cu_k: torch.Tensor,
    kv_lens: torch.Tensor,
    scale: Optional[float] = None,
    k_scale: Optional[float] = None,
    v_scale: Optional[float] = None,
    *,
    max_seqlen_q: Optional[int] = None,
) -> torch.Tensor:
    """FP8-KV counterpart of QSA's sparse_gqa_fwd_interface_triton_ck.

    Q/output are BF16 or FP16; K/V share one of PyTorch's four FP8 dtypes.
    Supports D=256, GQA groups 1..16, and 2D or per-KV-head 3D indices.
    Scales are scalar descales: FP32 QK is multiplied by k_scale, and FP32
    PV contributions by v_scale. Q/P are not quantized to FP8.

    Argument names and order match the target through v_scale.
    Pass max_seqlen_q explicitly during graph capture to avoid the target
    wrapper's host synchronization. Metadata can be int32 or int64 and is
    normalized to contiguous int32; valid sequence offsets must fit int32.
    The target's rounded slot window and empty-leading-block NaN behavior are retained.
    max_seqlen_q must cover every query segment when explicitly supplied.
    Q/K/V and metadata share a device; metadata offsets must describe valid
    packed segments and fit int32. Empty query batches return an empty output.

    FP8 decoding follows each storage format, including NaN/Inf. It does not
    emulate the current gfx936 Triton FP8-to-BF16 special-value conversion bug.
    FNUZ is a C++ extension where the installed Triton compiler rejects it.
    FP8 scales must be finite float32 scalars. On gfx936, E4M3FN may use a
    bounded per-call predecode workspace; see docs/sparse_gqa.md.
    """
    if scale is None:
        scale = q.shape[-1] ** -0.5
    indices, cu_q, cu_k, kv_lens, max_seqlen_q = _prepare_sparse_gqa_metadata(
        indices, cu_q, cu_k, kv_lens, max_seqlen_q
    )
    return flash_attn_cuda.sparse_gqa_attention_fp8(
        q.contiguous(), k.contiguous(), v.contiguous(), indices,
        cu_q, cu_k, kv_lens,
        float(scale), 1.0 if k_scale is None else float(k_scale),
        1.0 if v_scale is None else float(v_scale), max_seqlen_q,
    )

def _auto_sparse_attn_core(
    *,
    q,
    k,
    v,
    dropout_p=0.0,
    softmax_scale=None,
    causal=False,
    softcap=0.0,
    alibi_slopes=None,
    deterministic=False,
    return_attn_probs=False,
    # Auto-sparsity parameters
    smooth_k=True,
    simthreshd1=0.6,
    cdfthreshd=None,
    topk=None,
    attention_sink=False,
    block_m=64,
    block_k=64,
    # PV skip optimization parameters
    pv_threshold=50.0,
    enable_dynamic_skip=False,
    # Return options
    return_softmax_lse=False,
    return_sparsity=False,
    out=None,
):
    """
    Core implementation for auto-generating sparsity (CDF or TopK) and calling sparse_attn_func.
    Exactly one of cdfthreshd / topk must be provided.
    """
    assert (cdfthreshd is None) ^ (topk is None), "Provide exactly one of cdfthreshd or topk"
    # FA2 sparse kernel uses a fixed block size (BLOCK_M = 64). Keep q/k block sizes aligned.
    assert block_k == block_m, "block_k must equal block_m to match FA2 sparse kernel"

    # Input validation and conversion
    assert q.dim() == 4, f"q must be 4D (B, L, H, D), got shape {q.shape}"
    assert q.size(1) >= 128, "seqlen should be not less than 128."

    torch.cuda.set_device(q.device)

    # Convert to BHLD layout for processing
    q_bhld = q.transpose(1, 2).contiguous()  # (B, H, L, D)
    k_bhld = k.transpose(1, 2).contiguous()
    v_bhld = v.transpose(1, 2).contiguous()

    # Ensure correct dtype
    dtype = q_bhld.dtype
    if dtype == torch.float32 or dtype == torch.float16:
        q_bhld = q_bhld.to(torch.float16)
        k_bhld = k_bhld.to(torch.float16)
        v_bhld = v_bhld.to(torch.float16)
    else:
        q_bhld = q_bhld.to(torch.bfloat16)
        k_bhld = k_bhld.to(torch.bfloat16)
        v_bhld = v_bhld.to(torch.bfloat16)

    # Apply K smoothing (subtract mean directly)
    if smooth_k:
        k_bhld = k_bhld - k_bhld.mean(dim=-2, keepdim=True)

    B, H, seqlen_q, headdim = q_bhld.shape
    seqlen_k = k_bhld.size(-2)

    # Generate sparsity pattern with block_offset format
    block_offset, block_count = get_block_map_meansim(
        q_bhld, k_bhld,
        is_causal=causal,
        simthreshd1=simthreshd1,
        cdfthreshd=cdfthreshd,
        topk=topk,
        return_block_offset=True,
        attention_sink=attention_sink,
        BLKQ=block_m,
        BLKK=block_k,
    )

    if softmax_scale is None:
        softmax_scale = headdim ** (-0.5)

    # Disable Vertical sparsity (set column_count to 0)
    num_blocks_q = (seqlen_q + block_m - 1) // block_m
    num_blocks_k = (seqlen_k + block_k - 1) // block_k

    column_count = torch.zeros(
        (B, H, num_blocks_q), dtype=torch.int32, device=q.device
    )
    column_index = torch.zeros(
        (B, H, num_blocks_q, 1), dtype=torch.int32, device=q.device
    )

    # Convert block indices to row offsets (kernel expects row offsets, not block indices)
    block_offset = block_offset * block_k

    # Call sparse attention kernel (expects BLHD layout)
    result = sparse_attn_func(
        q, k, v,  # Use original BLHD layout
        block_count=block_count,
        block_offset=block_offset,
        column_count=column_count,
        column_index=column_index,
        dropout_p=dropout_p,
        softmax_scale=softmax_scale,
        causal=causal,
        softcap=softcap,
        alibi_slopes=alibi_slopes,
        deterministic=deterministic,
        return_attn_probs=return_attn_probs,
        return_softmax_lse=return_softmax_lse,
        out=out,
        pv_threshold=pv_threshold,
        enable_dynamic_skip=enable_dynamic_skip,
    )

    # Calculate sparsity if requested
    if return_sparsity:
        if not causal:
            qk_sparsity = 1 - (block_count.float().sum()) / (
                num_blocks_k * num_blocks_q * B * H
            )
        else:
            # For causal mask, expected number of blocks is (num_blocks_k + 1) / 2 per query block
            qk_sparsity = 1 - (block_count.float().sum()) / (
                ((num_blocks_k + 1) // 2) * num_blocks_q * B * H
            )

        if return_softmax_lse:
            out, softmax_lse = result
            return out, softmax_lse, qk_sparsity.item()
        else:
            return result, qk_sparsity.item()

    return result


def spas_fa2_attn_meansim_cuda(
    q,
    k,
    v,
    *,
    dropout_p=0.0,
    softmax_scale=None,
    causal=False,
    softcap=0.0,
    alibi_slopes=None,
    deterministic=False,
    return_attn_probs=False,
    smooth_k=True,
    simthreshd1=0.6,
    cdfthreshd=0.98,
    attention_sink=False,
    block_m=64,
    block_k=64,
    pv_threshold=50.0,
    enable_dynamic_skip=False,
    return_softmax_lse=False,
    return_sparsity=False,
    out=None,
):
    """
    Auto-sparse attention using CDF thresholding (cdfthreshd). Mirrors SpargeAttn's mean-sim CDF path.
    """
    return _auto_sparse_attn_core(
        q=q,
        k=k,
        v=v,
        dropout_p=dropout_p,
        softmax_scale=softmax_scale,
        causal=causal,
        softcap=softcap,
        alibi_slopes=alibi_slopes,
        deterministic=deterministic,
        return_attn_probs=return_attn_probs,
        smooth_k=smooth_k,
        simthreshd1=simthreshd1,
        cdfthreshd=cdfthreshd,
        topk=None,
        attention_sink=attention_sink,
        block_m=block_m,
        block_k=block_k,
        pv_threshold=pv_threshold,
        enable_dynamic_skip=enable_dynamic_skip,
        return_softmax_lse=return_softmax_lse,
        return_sparsity=return_sparsity,
        out=out,
    )


def spas_fa2_attn_meansim_topk_cuda(
    q,
    k,
    v,
    *,
    dropout_p=0.0,
    softmax_scale=None,
    causal=False,
    softcap=0.0,
    alibi_slopes=None,
    deterministic=False,
    return_attn_probs=False,
    smooth_k=True,
    simthreshd1=-0.1,
    topk=0.5,
    attention_sink=False,
    block_m=64,
    block_k=64,
    pv_threshold=50.0,
    enable_dynamic_skip=False,
    return_softmax_lse=False,
    return_sparsity=False,
    out=None,
):
    """
    Auto-sparse attention using Top-K ratio (topk). Mirrors SpargeAttn's mean-sim topk path.
    """
    return _auto_sparse_attn_core(
        q=q,
        k=k,
        v=v,
        dropout_p=dropout_p,
        softmax_scale=softmax_scale,
        causal=causal,
        softcap=softcap,
        alibi_slopes=alibi_slopes,
        deterministic=deterministic,
        return_attn_probs=return_attn_probs,
        smooth_k=smooth_k,
        simthreshd1=simthreshd1,
        cdfthreshd=None,
        topk=topk,
        attention_sink=attention_sink,
        block_m=block_m,
        block_k=block_k,
        pv_threshold=pv_threshold,
        enable_dynamic_skip=enable_dynamic_skip,
        return_softmax_lse=return_softmax_lse,
        return_sparsity=return_sparsity,
        out=out,
    )


class SparseLinearAttention(nn.Module):
    def __init__(self, head_dim, topk, feature_map='softmax', use_bf16=True, use_fp8=False, tie_feature_map_qk=True):
        R'''
        Args:
            head_dim: dimension of each head.
            topk: ratio of keys selected for sparse attention, shared across all queries.
            feature_map: feature map for linear attention, one of ['hedgehog', 'elu', 'relu', 'softmax'].
            use_bf16: whether to use bfloat16 (default) or float16 for computation. The conversion to bf16/fp16 is done inside the module.
            tie_feature_map_qk: whether to use the same feature map for query and key.
        '''

        super().__init__()
        assert not (use_bf16 and use_fp8), "Only one of bf16 and fp8 can be used."
        assert head_dim in (64, 128), "Dtype fp16/bf16 only support dim (64, 128)."
        assert not (use_fp8 and head_dim==64), "Dtype fp8 only support dim 128."
        self.dtype = torch.bfloat16 if use_bf16 else torch.float16
        self.dtype = torch.float8_e4m3fn if use_fp8 else self.dtype
        self.topk = topk
        self.proj_l = nn.Linear(head_dim, head_dim, dtype=torch.float32)

        if feature_map == 'elu':
            def elu_feature_map(x):
                return F.elu(x) + 1
            self.feature_map_q = elu_feature_map
            self.feature_map_k = elu_feature_map
        elif feature_map == 'relu':
            self.feature_map_q = nn.ReLU()
            self.feature_map_k = nn.ReLU()
        elif feature_map == 'softmax':
            def softmax_feature_map(x):
                return F.softmax(x, dim=-1)
            self.feature_map_q = softmax_feature_map
            self.feature_map_k = softmax_feature_map
        else:
            raise NotImplementedError(f'Not supported feature map {feature_map}.')

        if tie_feature_map_qk:
            self.feature_map_k = self.feature_map_q

        self.init_weights_()

    def init_weights_(self):
        with torch.no_grad():
            nn.init.zeros_(self.proj_l.weight)
            nn.init.zeros_(self.proj_l.bias)

    def forward(self, q, k, v, return_sparsity=False):
        R'''
        Args:
            q: queries of shape (B, L, H, D).
            k: keys of shape (B, L, H, D).
            v: values of shape (B, L, H, D).
            return_sparsity: whether to return the actual sparsity.
        '''

        B, seqlen_q, H, headdim = q.shape
        if headdim == 64:
            block_m = 64 if seqlen_q <= 2048 else 128
        elif headdim == 128:
            block_m = 64 if seqlen_q <= 2048 else 128
        block_k = 64
        if headdim == 64:
            sparse_map, lut, real_topk = get_block_map(q.transpose(1, 2).contiguous(), k.transpose(1, 2).contiguous(), topk_ratio=self.topk, BLKQ=block_m, BLKK=block_k)
        else:
            sparse_map, lut, real_topk = get_block_map_fast(q, k, topk_ratio=self.topk, BLKQ=block_m, BLKK=block_k)

        q = q.to(self.dtype)
        k = k.to(self.dtype)
        v = v.to(self.dtype)

        ########## SPARGE BEGIN ##########
        headdim = q.size(-1)
        block_offset, block_count = block_map_to_block_offset_triton(sparse_map)
        block_offset = block_offset * block_k
        softmax_scale = 1.0 / (headdim ** 0.5)
        assert headdim in [64, 128], "headdim should be in [64, 128]. For other headdim, you can use padding and specify the softmax scale."

        seqlen_k = k.size(1)
        num_blocks_q = (seqlen_q + block_m - 1) // block_m
        num_blocks_k = (seqlen_k + block_k - 1) // block_k
        column_count = torch.empty(
            (B, H, num_blocks_q), dtype=torch.int32, device=q.device
        )
        column_index = torch.empty(
            (B, H, num_blocks_q, 1), dtype=torch.int32, device=q.device
        )

        o_s = sparse_attn_func(
            q, k, v,  # Use original BLHD layout
            block_count=block_count,
            block_offset=block_offset,
            column_count=column_count,
            column_index=column_index,
            softmax_scale=softmax_scale,
            is_sla=True,
        )
        # pdb.set_trace()

        ########## SPARGE END ##########

        '''
        q = self.feature_map_q(q).contiguous().to(self.dtype) # c_q
        k = self.feature_map_k(k).contiguous().to(self.dtype) # c_k
        def calc_linear(q, k, v):
            kvsum = k.transpose(-1, -2) @ v
            ksum = torch.sum(k, dim=-2, keepdim=True)
            return (q @ kvsum) / (1e-5 + (q * ksum).sum(dim=-1, keepdim=True))
        o_l = calc_linear(q, k, v)

        with torch.amp.autocast('cuda', dtype=self.dtype):
            o_l = self.proj_l(o_l)
        o = (o_s + o_l).to(dtype)
        '''

        if return_sparsity:
            return o_s, real_topk / sparse_map.shape[-1]
        else:
            return o_s

def sparse_attn_with_sla(
    q,
    k,
    v,
    topk=0.2,
    feature_map="softmax",
    use_bf16=True,
    use_fp8=False,
    *,
    return_sparsity=False,
):
    """Compute attention with vertical and slash sparsity patterns.
    Most Arguments are the same with the flash_attn_func interface, except for 4 extra args:
    block_count and block_offset for slash sparsity patterns, and
    column_count and column_index for vertical sparsity patterns.
    For more details please refer to Appendix C.4.2 of paper https://arxiv.org/abs/2407.02490.

    Arguments:
        q: (batch_size, seqlen, nheads, headdim)
        k: (batch_size, seqlen, nheads_k, headdim)
        v: (batch_size, seqlen, nheads_k, headdim)
        return_sparsity: bool. Whether to return the sparsity. This option is for
           testing only. The returned probabilities are not guaranteed to be correct
           (they might not have the right scaling).
    Return:
        out: (batch_size, seqlen, nheads, headdim).
        softmax_lse [optional, if return_softmax_lse=True]: (batch_size, nheads, seqlen). The
            logsumexp of each row of the matrix QK^T * scaling (e.g., log of the softmax
            normalization factor).
    """

    q, k, v = [maybe_contiguous(x) for x in (q, k, v)]
    dtype = torch.bfloat16 if use_bf16 else torch.float16
    dtype = torch.float8_e4m3fn if use_fp8 else dtype
    B, seqlen_q, H, headdim = q.shape
    assert not (use_bf16 and use_fp8), "Only one of bf16 and fp8 can be used."
    assert headdim in (64, 128), "Dtype fp16/bf16 only support dim (64, 128)."
    assert not (use_fp8 and headdim==64), "Dtype fp8 only support dim 128."
    if headdim == 64:
        block_m = 64 if seqlen_q <= 2048 else 128
    elif headdim == 128:
        block_m = 64 if seqlen_q <= 2048 else 128
    block_k = 64
    if headdim == 64:
        sparse_map, lut, real_topk = get_block_map(q.transpose(1, 2).contiguous(), k.transpose(1, 2).contiguous(), topk_ratio=topk, BLKQ=block_m, BLKK=block_k)
    else:
        sparse_map, lut, real_topk = get_block_map_fast(q, k, topk_ratio=topk, BLKQ=block_m, BLKK=block_k)

    q = q.to(dtype)
    k = k.to(dtype)
    v = v.to(dtype)

    ########## SPARGE BEGIN ##########
    headdim = q.size(-1)
    block_offset, block_count = block_map_to_block_offset_triton(sparse_map)
    block_offset = block_offset * block_k
    softmax_scale = 1.0 / (headdim ** 0.5)
    assert headdim in [64, 128], "headdim should be in [64, 128]. For other headdim, you can use padding and specify the softmax scale."

    seqlen_k = k.size(1)
    num_blocks_q = (seqlen_q + block_m - 1) // block_m
    num_blocks_k = (seqlen_k + block_k - 1) // block_k
    column_count = torch.empty(
        (B, H, num_blocks_q), dtype=torch.int32, device=q.device
    )
    column_index = torch.empty(
        (B, H, num_blocks_q, 1), dtype=torch.int32, device=q.device
    )
    o_s = sparse_attn_func(
        q, k, v,  # Use original BLHD layout
        block_count=block_count,
        block_offset=block_offset,
        column_count=column_count,
        column_index=column_index,
        softmax_scale=softmax_scale,
        is_sla=True,
    )

    if return_sparsity:
        return o_s, real_topk / sparse_map.shape[-1]
    else:
        return o_s


def _require_hg_varlen_symbol(name: str):
    symbol = getattr(flash_attn_cuda, name, None)
    if symbol is None:
        raise RuntimeError(
            f"{name} is unavailable in this build. Rebuild flash_attn with HAS_HG_DISPATCH enabled."
        )
    return symbol


def _validate_hg_paged_kv_contract(k_cache, v_cache) -> None:
    if k_cache.dim() != 4 or v_cache.dim() != 4:
        raise ValueError("HG paged KV cache expects k and v to both be 4D tensors")
    if k_cache.shape[0] != v_cache.shape[0]:
        raise ValueError("HG paged KV cache expects k and v to agree on num_blocks")
    if k_cache.shape[1] != v_cache.shape[1] or k_cache.shape[2] != v_cache.shape[2]:
        raise ValueError(
            "HG paged KV cache expects "
            "k=[num_blocks, page_block_size, num_heads_k, d_qk], "
            "v=[num_blocks, page_block_size, num_heads_k, d_v]"
        )

def get_mla_metadata(
    cache_seqlens: torch.Tensor,
    num_heads_per_head_k: int,
    num_heads_k: int,
    is_fp8_kvcache: bool = False,
):
    return None, None


def flash_mla_with_kvcache(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    block_table: torch.Tensor,
    cache_seqlens: torch.Tensor,
    head_dim_v: int,
    tile_scheduler_metadata: Optional[torch.Tensor],
    num_splits: Optional[torch.Tensor],
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    use_cuda_graph: bool = True,
    out: Optional[torch.Tensor] = None,
):
    if k_cache.dtype not in (torch.float16, torch.bfloat16):
        raise NotImplementedError(
            "HG MLA dispatch in the main repository supports fp16/bf16 only; "
            "fp8/int8 MLA is not supported."
        )
    if softmax_scale is None:
        softmax_scale = q.shape[-1] ** (-0.5)

    hg_mla = _require_hg_varlen_symbol("hg_fwd_kvcache_mla")
    max_seqlen_k = 1 if use_cuda_graph else cache_seqlens.max().item()
    result = hg_mla(
        q,
        k_cache,
        None,
        head_dim_v,
        cache_seqlens,
        block_table,
        softmax_scale,
        causal,
        tile_scheduler_metadata,
        num_splits,
        out,
        max_seqlen_k,
    )
    if len(result) < 2:
        raise RuntimeError("hg_fwd_kvcache_mla did not return softmax_lse")
    return result[0], result[1]
