"""ROSA: Rapid Online Suffix Automaton language model.

Neurosymbolic language model that replaces attention with deterministic
recurrence pointers from an online suffix automaton, combined with learned
neural mixing and feedforward layers.

References:
    See ROSA 1-pager for full details on architecture and motivation.
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import einsum

from .rosa_blocks import MultiROSAResidual, ROSAResidual
from .utils import gumbel_sample, min_p_filter


class ROSA(nn.Module):
    """ROSA: Rapid Online Suffix Automaton language model.

    Replaces transformer attention with neurosymbolic recurrence pointers
    for linear-time, lossless information propagation.
    """

    def __init__(
        self,
        num_tokens: int,
        dim: int,
        depth: int,
        rosa_state_cap: int = 65536,
        drop: float = 0.1,
        ff_mult: int = 4,
        k_candidates: int = 1,
        temperature: float = 1.0,
        channels: int = 1,
        norm_eps: float = 1e-5,
        max_seq_len: int = 2048,
        tied_embedding: bool = True,
    ):
        """Initialize ROSA model.

        Args:
            num_tokens: Vocabulary size
            dim: Model dimension
            depth: Number of ROSA layers
            rosa_state_cap: Maximum pointer state capacity
            drop: Dropout probability
            ff_mult: Feedforward dimension multiplier
            k_candidates: Number of pointer candidates (>1 enables soft mixing)
            temperature: Temperature for soft pointer mixing
            channels: Number of parallel ROSA channels per layer (>1 enables multi-channel)
            norm_eps: Layer norm epsilon
            max_seq_len: Maximum sequence length (for compatibility)
            tied_embedding: Tie input/output embeddings
        """
        super().__init__()

        self.num_tokens = num_tokens
        self.dim = dim
        self.depth = depth
        self.tied_embedding = tied_embedding
        self.max_seq_len = max_seq_len

        # Token embeddings
        self.token_embed = nn.Embedding(num_tokens, dim)

        # Positional embeddings (simple learned absolute positions)
        self.pos_embed = nn.Embedding(max_seq_len, dim)

        # Dropout after embedding
        self.drop = nn.Dropout(drop)

        # ROSA blocks
        blocks = []
        for _ in range(depth):
            if channels > 1:
                blocks.append(
                    MultiROSAResidual(
                        dim,
                        channels=channels,
                        rosa_state_cap=rosa_state_cap,
                        drop=drop,
                        ff_mult=ff_mult,
                        norm_eps=norm_eps,
                    )
                )
            else:
                blocks.append(
                    ROSAResidual(
                        dim,
                        rosa_state_cap=rosa_state_cap,
                        drop=drop,
                        ff_mult=ff_mult,
                        k_candidates=k_candidates,
                        temperature=temperature,
                        norm_eps=norm_eps,
                    )
                )
        self.layers = nn.ModuleList(blocks)

        # Final norm
        self.norm = nn.LayerNorm(dim, eps=norm_eps)

        # Output projection (or tied with embedding)
        if not tied_embedding:
            self.to_logits = nn.Linear(dim, num_tokens, bias=False)
        else:
            self.to_logits = None

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize model weights using standard initialization."""
        # Initialize embeddings
        nn.init.normal_(self.token_embed.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.pos_embed.weight, mean=0.0, std=0.02)

        # Initialize output projection if not tied
        if self.to_logits is not None:
            nn.init.normal_(self.to_logits.weight, mean=0.0, std=0.02)

    @property
    def vocab_size(self) -> int:
        """Return vocabulary size."""
        return self.num_tokens

    @property
    def model_dim(self) -> int:
        """Return model dimension."""
        return self.dim

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_loss: bool = False,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input token IDs of shape (batch, seq_len)
            mask: Optional attention mask (not used in ROSA, for API compatibility)
            return_loss: If True, compute cross-entropy loss

        Returns:
            If return_loss=False: logits of shape (batch, seq_len, vocab_size)
            If return_loss=True: scalar loss tensor
        """
        if return_loss:
            x, labels = x[:, :-1], x[:, 1:]

        batch, seq_len = x.shape

        # Keep original token IDs for ROSA blocks
        input_ids = x.clone()

        # Embed tokens
        x = self.token_embed(x)

        # Add positional embeddings
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch, seq_len)
        x = x + self.pos_embed(positions)
        x = self.drop(x)

        # Apply ROSA blocks
        for layer in self.layers:
            x = layer(x, input_ids)

        # Final norm
        x = self.norm(x)

        # Compute logits
        if self.to_logits is not None:
            logits = self.to_logits(x)
        else:
            # Tied embeddings
            logits = einsum(x, self.token_embed.weight, "b n d, v d -> b n v")

        if not return_loss:
            return logits

        # Compute loss
        loss = F.cross_entropy(
            logits.reshape(-1, self.num_tokens),
            labels.reshape(-1),
            ignore_index=-1,
        )

        return loss

    @torch.no_grad()
    def generate(
        self,
        prompt: torch.Tensor,
        max_length: int,
        temperature: float = 1.0,
        filter_thres: float = 0.9,
        min_p: float = 0.1,
    ) -> torch.Tensor:
        """Generate text autoregressively.

        Args:
            prompt: Input prompt of shape (batch, seq_len)
            max_length: Number of tokens to generate
            temperature: Sampling temperature
            filter_thres: Not used (for compatibility)
            min_p: Min-p decoding threshold

        Returns:
            Generated sequence
        """
        out = prompt.clone()

        for _ in range(max_length):
            # Truncate to max_seq_len if needed
            if out.size(1) > self.max_seq_len:
                out = out[:, -self.max_seq_len :]

            logits = self.forward(out)
            logits = logits[:, -1]  # Get last position

            # Apply min-p filtering
            logits = min_p_filter(logits, min_p=min_p)

            # Sample
            sample = gumbel_sample(logits, temperature=temperature, dim=-1)
            out = torch.cat((out, sample), dim=-1)

        return out[:, prompt.shape[-1] :]
