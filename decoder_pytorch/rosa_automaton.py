"""Online Suffix Automaton for ROSA.

Pure Python implementation of online suffix automaton that produces
deterministic recurrence pointers for each token in a sequence.
"""

from typing import Dict, List

import torch


class OnlineSuffixAutomaton:
    """Online suffix automaton for substring indexing and recurrence detection.

    The automaton maintains all suffixes and substrings seen so far in an online
    manner, allowing O(1) amortized updates and deterministic pointer extraction.
    """

    __slots__ = ("lengths", "links", "nexts", "last")

    def __init__(self):
        """Initialize empty automaton with root state."""
        self.lengths: List[int] = [0]  # Length of longest string ending at state
        self.links: List[int] = [-1]  # Suffix link to longest proper suffix
        self.nexts: List[Dict[int, int]] = [dict()]  # Transitions: char -> state
        self.last: int = 0  # Current state (last added)

    def push(self, c: int) -> int:
        """Add character to automaton and return new state ID.

        Args:
            c: Character (token ID) to add

        Returns:
            State ID of the newly created state
        """
        cur = len(self.lengths)
        self.lengths.append(self.lengths[self.last] + 1)
        self.links.append(0)
        self.nexts.append({})

        p = self.last
        while p >= 0 and c not in self.nexts[p]:
            self.nexts[p][c] = cur
            p = self.links[p]

        if p == -1:
            self.links[cur] = 0
        else:
            q = self.nexts[p][c]
            if self.lengths[p] + 1 == self.lengths[q]:
                self.links[cur] = q
            else:
                # Clone state q
                clone = len(self.lengths)
                self.lengths.append(self.lengths[p] + 1)
                self.links.append(self.links[q])
                self.nexts.append(self.nexts[q].copy())

                # Update transitions
                while p >= 0 and self.nexts[p].get(c, None) == q:
                    self.nexts[p][c] = clone
                    p = self.links[p]

                self.links[q] = self.links[cur] = clone

        self.last = cur
        return cur

    def best_pointer(self, state_id: int) -> int:
        """Get best recurrence pointer for a state.

        Uses suffix link as a simple heuristic for the most informative
        prior recurrence point.

        Args:
            state_id: Current state ID

        Returns:
            Pointer to prior state (suffix link, or self if root)
        """
        lnk = self.links[state_id]
        return state_id if lnk < 0 else lnk


def pointers_for_sequence(seq: List[int]) -> List[int]:
    """Compute pointer sequence for a token sequence.

    Args:
        seq: List of token IDs

    Returns:
        List of pointer indices (one per token)
    """
    osa = OnlineSuffixAutomaton()
    out: List[int] = []
    for t in seq:
        s = osa.push(int(t))
        out.append(osa.best_pointer(s))
    return out


@torch.no_grad()
def batch_pointers(input_ids: torch.LongTensor, cap: int) -> torch.LongTensor:
    """Compute pointer sequences for a batch of token sequences.

    Args:
        input_ids: Token IDs of shape (batch, seq_len)
        cap: Maximum pointer value (for clamping to embedding table size)

    Returns:
        Pointer indices of shape (batch, seq_len)
    """
    B, T = input_ids.shape
    rows = []
    for b in range(B):
        seq = input_ids[b].tolist()
        ptrs = pointers_for_sequence(seq)
        rows.append(torch.tensor(ptrs, dtype=torch.long))
    ptr = torch.stack(rows, dim=0)
    if cap is not None and cap > 0:
        ptr.clamp_(min=0, max=cap - 1)
    return ptr
