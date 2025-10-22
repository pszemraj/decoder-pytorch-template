"""ROSA residual blocks.

Neurosymbolic blocks that combine deterministic recurrence pointers from
the online suffix automaton with learned neural mixing and feedforward layers.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .rosa_automaton import batch_pointers
from .rosa_relaxations import SoftPointerMixer, naive_topk_candidates


class STEIndexEmbedding(nn.Module):
    """Index embedding with automatic clamping (straight-through gradient)."""

    def __init__(self, num_embeddings: int, dim: int):
        """Initialize embedding table.

        Args:
            num_embeddings: Number of embeddings (pointer vocabulary size)
            dim: Embedding dimension
        """
        super().__init__()
        self.emb = nn.Embedding(num_embeddings, dim)

    def forward(self, idx: torch.LongTensor) -> torch.Tensor:
        """Embed indices with automatic clamping.

        Args:
            idx: Indices to embed

        Returns:
            Embeddings
        """
        return self.emb(idx.clamp(min=0, max=self.emb.num_embeddings - 1))


class ROSAResidual(nn.Module):
    """Single ROSA residual block.

    Replaces attention with deterministic recurrence pointers + learned mixing.
    """

    def __init__(
        self,
        dim: int,
        rosa_state_cap: int = 65536,
        drop: float = 0.1,
        ff_mult: int = 4,
        k_candidates: int = 1,
        temperature: float = 1.0,
        norm_eps: float = 1e-5,
    ):
        """Initialize ROSA residual block.

        Args:
            dim: Model dimension
            rosa_state_cap: Maximum pointer state capacity
            drop: Dropout probability
            ff_mult: Feedforward dimension multiplier
            k_candidates: Number of pointer candidates (>1 enables soft mixing)
            temperature: Temperature for soft pointer mixing
            norm_eps: Layer norm epsilon
        """
        super().__init__()
        self.dim = dim
        self.rosa_state_cap = int(rosa_state_cap)
        self.k_candidates = int(k_candidates)
        self.temperature = float(temperature)

        # Pointer embedding
        self.ptr_emb = STEIndexEmbedding(self.rosa_state_cap, dim)

        # Normalization layers
        self.norm_in = nn.LayerNorm(dim, eps=norm_eps)
        self.norm_out = nn.LayerNorm(dim, eps=norm_eps)

        # Gating for pointer mixing
        self.gate = nn.Parameter(torch.zeros(1))

        # Feedforward network (SwiGLU-style)
        self.ff = nn.Sequential(
            nn.Linear(dim, ff_mult * dim),
            nn.SiLU(),
            nn.Linear(ff_mult * dim, dim),
            nn.Dropout(drop),
        )

        # Optional soft pointer mixer
        self.soft_mixer = (
            SoftPointerMixer(dim, self.k_candidates)
            if self.k_candidates > 1
            else None
        )

    def forward(self, x: torch.Tensor, input_ids: torch.LongTensor) -> torch.Tensor:
        """Apply ROSA residual block.

        Args:
            x: Token stream of shape (batch, seq_len, dim)
            input_ids: Original token IDs of shape (batch, seq_len)

        Returns:
            Updated token stream of shape (batch, seq_len, dim)
        """
        # Get base pointers from online suffix automaton (no gradient)
        base_ptr = batch_pointers(input_ids, cap=self.rosa_state_cap).to(x.device)

        if self.k_candidates > 1:
            # Soft pointer mixing: generate candidates and learn mixture
            cands = naive_topk_candidates(base_ptr, self.k_candidates)  # (B, T, K)
            B, T, K = cands.shape
            flat = cands.reshape(B * T * K)
            emb_flat = self.ptr_emb(flat)
            cand_vec = emb_flat.view(B, T, K, self.dim)
            ptr_vec = self.soft_mixer(
                x, cand_vec, temperature=self.temperature, hard=not self.training
            )
        else:
            # Hard pointer (STE gradient)
            ptr_vec = self.ptr_emb(base_ptr)

        # Gated mixing of token stream and pointer embedding
        g = torch.sigmoid(self.gate)
        h = self.norm_in(x + g * ptr_vec)

        # Feedforward with residual
        y = x + self.ff(h)
        y = self.norm_out(y)

        return y


class MultiROSAResidual(nn.Module):
    """Multi-channel ROSA residual block.

    Splits model dimension into multiple channels, each with its own
    ROSA recurrence stream, then projects back.
    """

    def __init__(
        self,
        dim: int,
        channels: int = 2,
        rosa_state_cap: int = 65536,
        drop: float = 0.1,
        ff_mult: int = 4,
        norm_eps: float = 1e-5,
    ):
        """Initialize multi-channel ROSA block.

        Args:
            dim: Model dimension
            channels: Number of parallel ROSA channels
            rosa_state_cap: Maximum pointer state capacity
            drop: Dropout probability
            ff_mult: Feedforward dimension multiplier
            norm_eps: Layer norm epsilon
        """
        super().__init__()
        assert dim % channels == 0, "dim must be divisible by channels"
        subdim = dim // channels

        self.blocks = nn.ModuleList(
            [
                ROSAResidual(
                    subdim,
                    rosa_state_cap=rosa_state_cap,
                    drop=drop,
                    ff_mult=ff_mult,
                    norm_eps=norm_eps,
                )
                for _ in range(channels)
            ]
        )
        self.proj = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim, eps=norm_eps)

    def forward(self, x: torch.Tensor, input_ids: torch.LongTensor) -> torch.Tensor:
        """Apply multi-channel ROSA block.

        Args:
            x: Token stream of shape (batch, seq_len, dim)
            input_ids: Original token IDs of shape (batch, seq_len)

        Returns:
            Updated token stream of shape (batch, seq_len, dim)
        """
        parts = torch.chunk(x, len(self.blocks), dim=-1)
        outs = [blk(p, input_ids) for blk, p in zip(self.blocks, parts)]
        y = torch.cat(outs, dim=-1)
        return self.norm(x + self.proj(y))
