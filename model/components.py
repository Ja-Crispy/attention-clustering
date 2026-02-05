"""Core model components: RoPE, RMSNorm, QK-Norm, SwiGLU."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization (Qwen3/Llama-style)."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)
        return (x.float() * norm).type_as(x) * self.weight


class RoPE(nn.Module):
    """Rotary Position Embeddings. Can be disabled for DroPE experiments."""

    def __init__(self, head_dim: int, max_seq_len: int = 2048, theta: float = 1_000_000.0):
        super().__init__()
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.theta = theta

        freqs = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
        t = torch.arange(max_seq_len)
        freqs = torch.outer(t, freqs)
        # Store cos and sin directly instead of complex numbers for broader dtype support
        self.register_buffer("cos_cached", freqs.cos())
        self.register_buffer("sin_cached", freqs.sin())

    def forward(self, x: torch.Tensor, offset: int = 0) -> torch.Tensor:
        """Apply rotary embeddings to input tensor.

        Args:
            x: (batch, seq_len, num_heads, head_dim)
            offset: position offset for caching
        """
        seq_len = x.shape[1]
        cos_half = self.cos_cached[offset : offset + seq_len]  # (seq, head_dim//2)
        sin_half = self.sin_cached[offset : offset + seq_len]

        # Repeat to full head_dim: [cos, cos] and [sin, sin]
        cos = torch.cat([cos_half, cos_half], dim=-1).unsqueeze(0).unsqueeze(2)
        sin = torch.cat([sin_half, sin_half], dim=-1).unsqueeze(0).unsqueeze(2)

        # Split into halves and rotate: [-x2, x1]
        x1 = x[..., : self.head_dim // 2]
        x2 = x[..., self.head_dim // 2 :]
        rotated = torch.cat([-x2, x1], dim=-1)
        return (x.float() * cos + rotated.float() * sin).type_as(x)


class SwiGLU(nn.Module):
    """SwiGLU feed-forward network (Qwen3/Llama-style)."""

    def __init__(self, hidden_size: int, intermediate_size: int, bias: bool = False):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=bias)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=bias)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))
