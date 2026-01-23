"""Differential Attention V2 transformer implementation.

Implements DiffAttn V2 following the HuggingFace blog spec:
- Core mechanism: doubled query heads with shared KV, differential subtraction
- attn1 - sigmoid(λ) * attn2 where adjacent Q heads share identical K/V

References:
- Differential Transformer: https://arxiv.org/abs/2410.05258
- HuggingFace blog: https://huggingface.co/blog/differential-transformer
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import einsum, rearrange
from rotary_embedding_torch import RotaryEmbedding

from .llama import RMSNorm, SwiGLU
from .utils import gumbel_sample, min_p_filter


def exists(v):
    """Check if value exists (is not None)."""
    return v is not None


class DiffAttentionV2(nn.Module):
    """Differential Attention V2 with doubled Q heads and shared KV.

    Key insight: 2h query heads, h KV heads. Adjacent Q heads (0::2, 1::2)
    share identical K/V via repeat_interleave. Output is attn1 - sigmoid(λ) * attn2.
    """

    def __init__(
        self,
        dim: int,
        heads: int = 8,
        dim_head: int = 64,
        causal: bool = True,
        max_seq_len: int = 2048,
        rope_theta: float = 10000.0,
        flash_attn: bool = True,
    ):
        """Initialize differential attention.

        Args:
            dim: Model dimension
            heads: Number of output heads (internal uses 2*heads for Q)
            dim_head: Dimension per head
            causal: Whether to use causal masking
            max_seq_len: Maximum sequence length for RoPE
            rope_theta: RoPE theta parameter
            flash_attn: Use flash attention if available
        """
        super().__init__()
        self.heads = heads  # Output heads
        self.q_heads = heads * 2  # Doubled Q heads for differential
        self.dim_head = dim_head
        self.causal = causal
        self.flash_attn = flash_attn
        if self.flash_attn and not self._flash_attn_available():
            print(
                "Warning: Flash attention requested but not available, using standard attention."
            )
            self.flash_attn = False
        self.max_seq_len = max_seq_len

        q_inner_dim = self.q_heads * dim_head  # Doubled
        kv_inner_dim = heads * dim_head  # Same KV budget

        # Q has doubled heads, K/V have standard heads
        self.wq = nn.Linear(dim, q_inner_dim, bias=False)
        self.wk = nn.Linear(dim, kv_inner_dim, bias=False)
        self.wv = nn.Linear(dim, kv_inner_dim, bias=False)
        self.wo = nn.Linear(heads * dim_head, dim, bias=False)  # Output from h heads

        # Per-token, per-head lambda projection
        # Projects from model dim to number of output heads
        self.lam_proj = nn.Linear(dim, heads, bias=False)

        # RoPE
        self.rotary_emb = RotaryEmbedding(
            dim=dim_head,
            theta=rope_theta,
        )

        # Register causal mask buffer
        self.register_buffer("causal_mask", None, persistent=False)

    def _flash_attn_available(self) -> bool:
        """Return True if flash attention kernels are available."""
        if torch.cuda.is_available():
            return True
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return True
        dynamo = getattr(torch, "_dynamo", None)
        if dynamo is not None:
            try:
                return bool(dynamo.is_compiling())
            except Exception:
                return False
        return False

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Apply differential attention.

        Args:
            x: Input tensor of shape (batch, seq_len, dim)
            mask: Optional attention mask

        Returns:
            Output tensor of shape (batch, seq_len, dim)
        """
        batch, seq_len, _ = x.shape

        # Compute per-token lambda values: (batch, seq_len, heads)
        lam = torch.sigmoid(self.lam_proj(x))
        # Expand for attention: (batch, heads, seq_len, 1) for broadcasting
        lam = rearrange(lam, "b n h -> b h n 1")

        # Project to Q, K, V
        q = self.wq(x)  # (batch, seq_len, 2*heads * dim_head)
        k = self.wk(x)  # (batch, seq_len, heads * dim_head)
        v = self.wv(x)  # (batch, seq_len, heads * dim_head)

        # Split heads - Q has 2*heads, K/V have heads
        q = rearrange(q, "b n (h d) -> b h n d", h=self.q_heads)
        k = rearrange(k, "b n (h d) -> b h n d", h=self.heads)
        v = rearrange(v, "b n (h d) -> b h n d", h=self.heads)

        # Duplicate K/V so adjacent Q heads share the same K/V
        # (batch, heads, seq, dim) -> (batch, 2*heads, seq, dim)
        k = k.repeat_interleave(2, dim=1)
        v = v.repeat_interleave(2, dim=1)

        # Apply RoPE
        q = self.rotary_emb.rotate_queries_or_keys(q)
        k = self.rotary_emb.rotate_queries_or_keys(k)

        # Compute attention
        if self.flash_attn:
            # Use PyTorch's scaled_dot_product_attention
            attn_out = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=mask,
                is_causal=self.causal and mask is None,
            )
        else:
            # Manual attention
            scale = self.dim_head**-0.5
            attn = einsum(q, k, "b h i d, b h j d -> b h i j") * scale

            if self.causal:
                if self.causal_mask is None or self.causal_mask.shape[-1] < seq_len:
                    mask_size = min(seq_len, self.max_seq_len)
                    self.causal_mask = torch.ones(
                        (mask_size, mask_size), device=x.device, dtype=torch.bool
                    ).triu(1)

                causal_mask = self.causal_mask[:seq_len, :seq_len]
                attn = attn.masked_fill(causal_mask, float("-inf"))

            if exists(mask):
                attn = attn.masked_fill(~mask, float("-inf"))

            attn = F.softmax(attn, dim=-1)
            attn_out = einsum(attn, v, "b h i j, b h j d -> b h i d")

        # Differential attention: split into odd/even heads and subtract
        # attn_out shape: (batch, 2*heads, seq_len, dim_head)
        attn1 = attn_out[:, 0::2]  # Even heads: (batch, heads, seq_len, dim_head)
        attn2 = attn_out[:, 1::2]  # Odd heads: (batch, heads, seq_len, dim_head)

        # Differential: attn1 - sigmoid(lambda) * attn2
        out = attn1 - lam * attn2

        # Merge heads
        out = rearrange(out, "b h n d -> b n (h d)")

        # Output projection
        return self.wo(out)


class DiffTransformerBlock(nn.Module):
    """Transformer block with Differential Attention V2 and pre-normalization."""

    def __init__(
        self,
        dim: int,
        heads: int = 8,
        dim_head: int = 64,
        norm_eps: float = 1e-5,
        causal: bool = True,
        max_seq_len: int = 2048,
        rope_theta: float = 10000.0,
        multiple_of: int = 256,
        ffn_dim_multiplier: Optional[float] = None,
        flash_attn: bool = True,
    ):
        """Initialize transformer block.

        Args:
            dim: Model dimension
            heads: Number of attention heads
            dim_head: Dimension per head
            norm_eps: Epsilon for RMSNorm
            causal: Causal masking
            max_seq_len: Maximum sequence length
            rope_theta: RoPE theta
            multiple_of: FFN hidden dimension multiple
            ffn_dim_multiplier: FFN dimension multiplier
            flash_attn: Use flash attention
        """
        super().__init__()

        # Pre-norm for attention
        self.attn_norm = RMSNorm(dim, eps=norm_eps)
        self.attn = DiffAttentionV2(
            dim=dim,
            heads=heads,
            dim_head=dim_head,
            causal=causal,
            max_seq_len=max_seq_len,
            rope_theta=rope_theta,
            flash_attn=flash_attn,
        )

        # Pre-norm for feedforward
        self.ff_norm = RMSNorm(dim, eps=norm_eps)
        self.ff = SwiGLU(
            dim=dim,
            multiple_of=multiple_of,
            ffn_dim_multiplier=ffn_dim_multiplier,
        )

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Apply transformer block.

        Args:
            x: Input tensor
            mask: Optional attention mask

        Returns:
            Output tensor
        """
        # Attention with residual
        x = x + self.attn(self.attn_norm(x), mask=mask)

        # Feedforward with residual
        x = x + self.ff(self.ff_norm(x))

        return x


class DiffLlamaV2(nn.Module):
    """Llama-style causal language model with Differential Attention V2."""

    def __init__(
        self,
        num_tokens: int,
        dim: int,
        depth: int,
        heads: int = 8,
        dim_head: int = 64,
        norm_eps: float = 1e-5,
        max_seq_len: int = 2048,
        rope_theta: float = 10000.0,
        multiple_of: int = 256,
        ffn_dim_multiplier: Optional[float] = None,
        tied_embedding: bool = True,
        flash_attn: bool = True,
    ):
        """Initialize DiffLlamaV2 model.

        Args:
            num_tokens: Vocabulary size
            dim: Model dimension
            depth: Number of transformer layers
            heads: Number of attention heads
            dim_head: Dimension per head
            norm_eps: Epsilon for RMSNorm
            max_seq_len: Maximum sequence length
            rope_theta: RoPE theta parameter
            multiple_of: Make FFN hidden dim multiple of this
            ffn_dim_multiplier: FFN dimension multiplier
            tied_embedding: Tie input/output embeddings
            flash_attn: Use flash attention
        """
        super().__init__()

        self.num_tokens = num_tokens
        self.dim = dim
        self.depth = depth
        self.tied_embedding = tied_embedding

        # Token embeddings
        self.token_embed = nn.Embedding(num_tokens, dim)

        # Transformer blocks with DiffAttention
        self.layers = nn.ModuleList(
            [
                DiffTransformerBlock(
                    dim=dim,
                    heads=heads,
                    dim_head=dim_head,
                    norm_eps=norm_eps,
                    causal=True,
                    max_seq_len=max_seq_len,
                    rope_theta=rope_theta,
                    multiple_of=multiple_of,
                    ffn_dim_multiplier=ffn_dim_multiplier,
                    flash_attn=flash_attn,
                )
                for _ in range(depth)
            ]
        )

        # Final norm
        self.norm = RMSNorm(dim, eps=norm_eps)

        # Output projection (or tied with embedding)
        if not tied_embedding:
            self.to_logits = nn.Linear(dim, num_tokens, bias=False)
        else:
            self.to_logits = None

        # Initialize weights properly
        self._init_weights()

    def _init_weights(self):
        """Initialize model weights using standard transformer initialization."""
        # Initialize embeddings with smaller std
        nn.init.normal_(self.token_embed.weight, mean=0.0, std=0.02)

        # Initialize output projection if not tied
        if self.to_logits is not None:
            nn.init.normal_(self.to_logits.weight, mean=0.0, std=0.02)

        # Initialize lambda projections to zero -> sigmoid(0) = 0.5 at init
        for layer in self.layers:
            nn.init.zeros_(layer.attn.lam_proj.weight)

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
            mask: Optional attention mask
            return_loss: If True, compute cross-entropy loss

        Returns:
            If return_loss=False: logits of shape (batch, seq_len, vocab_size)
            If return_loss=True: scalar loss tensor
        """
        if return_loss:
            x, labels = x[:, :-1], x[:, 1:]

        # Embed tokens
        x = self.token_embed(x)

        # Apply transformer blocks
        for layer in self.layers:
            x = layer(x, mask=mask)

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
            rearrange(logits, "b n v -> b v n"),
            labels,
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
            logits = self.forward(out)
            logits = logits[:, -1]  # Get last position

            # Apply min-p filtering
            logits = min_p_filter(logits, min_p=min_p)

            # Sample
            sample = gumbel_sample(logits, temperature=temperature, dim=-1)
            out = torch.cat((out, sample), dim=-1)

        return out[:, prompt.shape[-1] :]
