"""Mini-Qwen3 Transformer with QK-Norm, GQA/MHA, removable RoPE."""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .components import RMSNorm, RoPE, SwiGLU

from config import ModelConfig


class Attention(nn.Module):
    """Multi-head attention with QK-Norm and GQA support.

    Supports:
        - Standard MHA (num_kv_heads == num_q_heads)
        - Grouped Query Attention (num_kv_heads < num_q_heads)
        - QK-Norm: RMSNorm on Q,K before RoPE (cosine similarity attention)
        - Removable RoPE for DroPE experiments
        - Returning raw attention weights for mech interp
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.num_q_heads = config.num_q_heads
        self.num_kv_heads = config.num_kv_heads
        self.head_dim = config.head_dim
        self.num_kv_groups = config.num_kv_groups
        self.hidden_size = config.hidden_size

        self.q_proj = nn.Linear(config.hidden_size, self.num_q_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_q_heads * self.head_dim, config.hidden_size, bias=False)

        self.use_qk_norm = config.qk_norm
        if self.use_qk_norm:
            self.q_norm = RMSNorm(self.head_dim, config.rms_norm_eps)
            self.k_norm = RMSNorm(self.head_dim, config.rms_norm_eps)

        self.scale = 1.0 / math.sqrt(self.head_dim)

    def forward(
        self,
        x: torch.Tensor,
        rope: Optional[RoPE] = None,
        mask: Optional[torch.Tensor] = None,
        return_attention: bool = False,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        B, T, C = x.shape

        q = self.q_proj(x).view(B, T, self.num_q_heads, self.head_dim)
        k = self.k_proj(x).view(B, T, self.num_kv_heads, self.head_dim)
        v = self.v_proj(x).view(B, T, self.num_kv_heads, self.head_dim)

        # QK-Norm: normalize before RoPE (forces cosine similarity)
        if self.use_qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        # RoPE (skipped when rope=None, i.e. DroPE mode)
        if rope is not None:
            q = rope(q)
            k = rope(k)

        # GQA: expand KV heads to match Q heads
        if self.num_kv_groups > 1:
            k = k.unsqueeze(3).expand(B, T, self.num_kv_heads, self.num_kv_groups, self.head_dim)
            k = k.reshape(B, T, self.num_q_heads, self.head_dim)
            v = v.unsqueeze(3).expand(B, T, self.num_kv_heads, self.num_kv_groups, self.head_dim)
            v = v.reshape(B, T, self.num_q_heads, self.head_dim)

        # Transpose: (B, heads, T, head_dim)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        if return_attention:
            # Manual attention for mech interp (returns weights)
            attn = (q @ k.transpose(-2, -1)) * self.scale
            if mask is not None:
                attn = attn.masked_fill(~mask, float("-inf"))
            attn_weights = F.softmax(attn, dim=-1)
            out = attn_weights @ v
        else:
            # Use SDPA for training efficiency
            out = F.scaled_dot_product_attention(
                q, k, v, attn_mask=mask, is_causal=(mask is None), scale=self.scale
            )
            attn_weights = None

        out = out.transpose(1, 2).contiguous().view(B, T, self.num_q_heads * self.head_dim)
        out = self.o_proj(out)
        return out, attn_weights


class TransformerBlock(nn.Module):
    """Pre-norm transformer block: RMSNorm → Attention → residual → RMSNorm → MLP → residual."""

    def __init__(self, config: ModelConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.attn_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.attn = Attention(config)
        self.mlp_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.mlp = SwiGLU(config.hidden_size, config.intermediate_size)

    def forward(
        self,
        x: torch.Tensor,
        rope: Optional[RoPE] = None,
        mask: Optional[torch.Tensor] = None,
        return_attention: bool = False,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        attn_out, attn_weights = self.attn(
            self.attn_norm(x), rope=rope, mask=mask, return_attention=return_attention
        )
        x = x + attn_out
        x = x + self.mlp(self.mlp_norm(x))
        return x, attn_weights


class Transformer(nn.Module):
    """Full Mini-Qwen3 transformer for causal language modeling.

    Features:
        - QK-Norm (cosine similarity attention)
        - GQA or MHA (configurable)
        - Removable RoPE (for DroPE experiments)
        - Tied input/output embeddings
        - Returns attention weights on demand for mech interp
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        self.embed = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            [TransformerBlock(config, i) for i in range(config.num_layers)]
        )
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)

        if not config.tie_embeddings:
            self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        else:
            self.lm_head = None

        # RoPE (set to None to disable for DroPE)
        if config.use_rope:
            self.rope = RoPE(config.head_dim, config.max_seq_len, config.rope_theta)
        else:
            self.rope = None

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module):
        std = 0.02
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)

    def forward(
        self,
        input_ids: torch.Tensor,
        return_attention: bool = False,
        return_hidden_states: bool = False,
    ) -> dict[str, torch.Tensor]:
        """Forward pass.

        Returns dict with:
            - logits: (B, T, vocab_size)
            - attention_weights: list of (B, heads, T, T) per layer (if return_attention)
            - hidden_states: list of (B, T, hidden) per layer (if return_hidden_states)
        """
        B, T = input_ids.shape
        h = self.embed(input_ids)

        attn_weights_all = []
        hidden_states_all = [h] if return_hidden_states else []

        for layer in self.layers:
            h, attn_w = layer(h, rope=self.rope, return_attention=return_attention)
            if return_attention:
                attn_weights_all.append(attn_w)
            if return_hidden_states:
                hidden_states_all.append(h)

        h = self.norm(h)

        if self.lm_head is not None:
            logits = self.lm_head(h)
        else:
            logits = F.linear(h, self.embed.weight)

        result = {"logits": logits}
        if return_attention:
            result["attention_weights"] = attn_weights_all
        if return_hidden_states:
            result["hidden_states"] = hidden_states_all
        return result

    def disable_rope(self):
        """DroPE: remove positional embeddings for inference/recalibration."""
        self.rope = None

    def enable_rope(self, theta: Optional[float] = None):
        """Re-enable RoPE (e.g. after DroPE experiment)."""
        theta = theta or self.config.rope_theta
        self.rope = RoPE(self.config.head_dim, self.config.max_seq_len, theta)
        if next(self.parameters()).is_cuda:
            self.rope = self.rope.to(next(self.parameters()).device)

    def num_parameters(self, trainable_only: bool = True) -> int:
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())

    def get_param_groups(self) -> dict[str, list[nn.Parameter]]:
        """Split parameters for Muon (2D hidden weights) vs AdamW (everything else).

        Returns:
            dict with 'muon_params' and 'adam_params' keys.
        """
        muon_params = []
        adam_params = []

        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            # Muon: 2D hidden weights (not embedding or lm_head)
            if param.ndim >= 2 and "embed" not in name and "lm_head" not in name:
                muon_params.append(param)
            else:
                adam_params.append(param)

        return {"muon_params": muon_params, "adam_params": adam_params}
