"""Relaxations for differentiable pointer selection in ROSA.

Provides soft/stochastic alternatives to hard pointer selection to enable
gradient flow during training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def naive_topk_candidates(ptr_idx: torch.LongTensor, k: int) -> torch.LongTensor:
    """Generate K candidate pointers by offsetting from base pointer.

    Simple heuristic: take pointer and K-1 previous states.

    Args:
        ptr_idx: Base pointer indices of shape (batch, seq_len)
        k: Number of candidates to generate

    Returns:
        Candidate indices of shape (batch, seq_len, k)
    """
    B, T = ptr_idx.shape
    K = int(k)
    cands = []
    for i in range(K):
        cands.append(torch.clamp(ptr_idx - i, min=0))
    return torch.stack(cands, dim=-1)  # (B, T, K)


class SoftPointerMixer(nn.Module):
    """Soft pointer mixing using learned attention over K candidates.

    During training, computes weighted average of candidate embeddings.
    During eval, can optionally use straight-through estimator for hard selection.
    """

    def __init__(self, dim: int, k: int):
        """Initialize soft pointer mixer.

        Args:
            dim: Model dimension
            k: Number of candidate pointers
        """
        super().__init__()
        self.k = int(k)
        self.logit_proj = nn.Linear(dim, self.k)

    def forward(
        self,
        token_stream: torch.Tensor,
        cand_embeds: torch.Tensor,
        temperature: float = 1.0,
        hard: bool = False,
    ) -> torch.Tensor:
        """Mix candidate embeddings using learned attention.

        Args:
            token_stream: Token embeddings of shape (batch, seq_len, dim)
            cand_embeds: Candidate pointer embeddings of shape (batch, seq_len, k, dim)
            temperature: Softmax temperature for mixing
            hard: If True, use straight-through estimator (hard selection in forward,
                  soft gradient in backward)

        Returns:
            Mixed pointer embedding of shape (batch, seq_len, dim)
        """
        B, T, D = token_stream.shape
        logits = self.logit_proj(token_stream)  # (B, T, K)
        probs = F.softmax(logits / max(1e-6, temperature), dim=-1)

        # Soft mixture
        mixed = torch.einsum("btk,btkd->btd", probs, cand_embeds)

        if hard:
            # Straight-through estimator: argmax forward, soft backward
            idx = torch.argmax(probs, dim=-1)
            hard_vec = cand_embeds[
                torch.arange(B).unsqueeze(1), torch.arange(T).unsqueeze(0), idx
            ]
            mixed = mixed + (hard_vec - mixed).detach()

        return mixed
