"""Inference-only WeLMV4 NextN (MTP) speculative draft model.

Ported from the official WeLM SGLang tree. The draft shares the target's
token embedding, over-encoding (OE) embeddings and LM head through
``set_embed_and_head`` instead of allocating its own copies (the OE tables
alone are ~15 GiB per TP rank).
"""

import logging
import os
from typing import Iterable, Optional, Tuple

import torch
from torch import nn
from transformers import PretrainedConfig

from sglang.srt.distributed import get_pp_group, get_tensor_model_parallel_world_size
from sglang.srt.eplb.expert_distribution import get_global_expert_distribution_recorder
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.models.welmv4 import (
    Qwen2MoeDecoderLayer,
    WelmV4FusedRMSNorm,
    WeLMV4MoeForCausalLM,
    hash_input_ids_vectorized,
    welm_mtp_mirror_kv,
    welm_spec_n_gram_input_ids,
)
from sglang.srt.utils import add_prefix, is_cuda

logger = logging.getLogger(__name__)

_is_cuda = is_cuda()


class WeLMV4ModelNextN(nn.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size

        # Shared with the target model via set_embed_and_head.
        self.embed_tokens = None
        self.oe_embed = None
        self.oe_gate_up_proj = None
        self.oe_grams = config.oe_grams
        self.oe_vocab_sizes = config.oe_vocab_sizes

        self.enorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.hnorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.eh_proj = nn.Linear(2 * config.hidden_size, config.hidden_size, bias=True)

        self.alt_stream = torch.cuda.Stream() if _is_cuda else None
        self.decoder_layers = nn.ModuleList(
            [
                Qwen2MoeDecoderLayer(
                    config,
                    i,
                    quant_config=quant_config,
                    is_nextn=True,
                    prefix=add_prefix("decoder", prefix),
                    alt_stream=self.alt_stream,
                )
                for i in range(config.num_nextn_predict_layers)
            ]
        )

        self.shared_head = nn.Module()
        self.shared_head.norm = WelmV4FusedRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def _oe_embedding(self, input_ids, forward_batch, hidden_states):
        return self._oe_embedding_from(
            input_ids, forward_batch.n_gram_input_ids, hidden_states
        )

    def _oe_embedding_from(self, input_ids, n_gram_input_ids, hidden_states):
        max_n = max(self.oe_grams)
        grams = []
        for n in range(2, max_n + 1):
            gram = n_gram_input_ids.get_gram(n) if n_gram_input_ids is not None else None
            grams.append(gram if gram is not None else torch.zeros_like(input_ids))
        input_ids_ngram = []
        input_ids_ngram_tmp = input_ids
        for g in range(1, max_n):
            input_ids_ngram_tmp = input_ids_ngram_tmp + grams[g - 1] * (
                self.vocab_size**g
            )
            input_ids_ngram.append(hash_input_ids_vectorized(input_ids_ngram_tmp))
        emb_ngram = []
        for i, vs in enumerate(self.oe_vocab_sizes):
            emb_ngram.append(self.oe_embed[i](input_ids_ngram[self.oe_grams[i] - 2] % vs))
        emb_new, _ = self.oe_gate_up_proj(torch.cat(emb_ngram, dim=-1))
        return (hidden_states + emb_new) / 2.0

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: torch.Tensor = None,
    ) -> torch.Tensor:
        if input_embeds is None:
            hidden_states = self.embed_tokens(input_ids)
        else:
            hidden_states = input_embeds

        if len(self.oe_grams) > 0:
            forward_batch.n_gram_input_ids = welm_spec_n_gram_input_ids(
                input_ids, positions, forward_batch, is_draft=True
            )
            hidden_states = self._oe_embedding(input_ids, forward_batch, hidden_states)
            mode = forward_batch.forward_mode
            if (
                os.environ.get("WELM_MTP_DECODE_MIRROR", "1") == "1"
                and not mode.is_idle()
                and not mode.is_extend_without_speculative()
                and input_ids.shape[0] > 0
            ):
                welm_mtp_mirror_kv(
                    input_ids,
                    positions,
                    forward_batch,
                    self.embed_tokens,
                    self._oe_embedding_from,
                )

        if (
            forward_batch.enable_kv_mirror
            and forward_batch.forward_mode.is_extend_without_speculative()
        ):
            # The KV-mirror target prefill only returns the last hidden state
            # of each request; keep the matching embedding rows.
            main_hidden_states = forward_batch.spec_info.hidden_states
            if (
                main_hidden_states is not None
                and hidden_states.shape[0] != main_hidden_states.shape[0]
            ):
                if forward_batch.custom_last_index is None:
                    forward_batch.custom_last_index = (
                        torch.cumsum(forward_batch.extend_seq_lens, dim=0) - 1
                    )
                hidden_states = hidden_states[forward_batch.custom_last_index]

        if hidden_states.shape[0] > 0:
            hidden_states = self.eh_proj(
                torch.cat(
                    (
                        self.enorm(hidden_states),
                        self.hnorm(forward_batch.spec_info.hidden_states),
                    ),
                    dim=-1,
                )
            )

        residual = None
        final_experts_output = None
        final_shared_output = None
        with get_global_expert_distribution_recorder().disable_this_region():
            for layer in self.decoder_layers:
                hidden_states, residual = layer(
                    positions, hidden_states, forward_batch, residual
                )
                final_experts_output = getattr(layer, "final_mlp_experts_output", None)
                final_shared_output = getattr(layer, "final_mlp_shared_output", None)

        pre_norm_hidden_states = hidden_states
        if not forward_batch.forward_mode.is_idle():
            if residual is not None:
                if final_experts_output is not None:
                    hidden_states = final_experts_output.float() + residual.float()
                    if final_shared_output is not None:
                        hidden_states = hidden_states + final_shared_output.float()
                else:
                    hidden_states = hidden_states.float() + residual.float()
                hidden_states = hidden_states.to(self.shared_head.norm.weight.dtype)
            pre_norm_hidden_states = hidden_states
            hidden_states, _ = self.shared_head.norm(hidden_states)
        return hidden_states, pre_norm_hidden_states


class WeLMV4MoeForCausalLMNextN(WeLMV4MoeForCausalLM):
    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        nn.Module.__init__(self)
        self.config = config
        self.tp_size = get_tensor_model_parallel_world_size()
        self.quant_config = quant_config
        self.pp_group = get_pp_group()

        self.model = WeLMV4ModelNextN(
            config, quant_config, prefix=add_prefix("model", prefix)
        )
        self.lm_head = None
        self.logits_processor = LogitsProcessor(config)
        super().post_init_after_load_weights(is_nextn=True)

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: torch.Tensor = None,
        **kwargs,
    ):
        hidden_states, pre_norm_hidden_states = self.model(
            input_ids, positions, forward_batch, input_embeds
        )
        # Like the target model, hand the next draft step the hidden state
        # before the final norm: that is what hnorm was trained on.
        aux_hidden_states = None
        if (
            forward_batch.capture_hidden_mode.need_capture()
            and os.environ.get("WELM_MTP_DRAFT_PRENORM", "1") == "1"
        ):
            aux_hidden_states = [pre_norm_hidden_states]
        return self.logits_processor(
            input_ids, hidden_states, self.lm_head, forward_batch, aux_hidden_states
        )

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        super().load_weights(weights, is_nextn=True)


EntryClass = WeLMV4MoeForCausalLMNextN
