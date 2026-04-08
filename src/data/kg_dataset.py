"""
KG triple dataset for RotatE training (Stage 1).

Implements self-adversarial negative sampling: for each positive triple
we generate `negative_sample_size` corrupted triples (half head, half tail).
"""
from __future__ import annotations

import random
from typing import Dict, Optional

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class KGTripleDataset(Dataset):
    """
    Dataset that yields positive triples and pre-sampled negatives.

    Args:
        triples:             (N, 3) tensor of (head, relation, tail) IDs
        num_entities:        Total entity count for negative sampling
        negative_sample_size: # negatives per positive (split head/tail)
        mode:                "train" (with negatives) or "eval" (positives only)
        true_head:          dict mapping (rel, tail) → set of true heads (for filtered eval)
        true_tail:          dict mapping (head, rel) → set of true tails
    """

    def __init__(
        self,
        triples: torch.Tensor,
        num_entities: int,
        negative_sample_size: int = 128,
        mode: str = "train",
        true_head: Optional[Dict] = None,
        true_tail: Optional[Dict] = None,
        seed: int = 42,
    ):
        assert mode in ("train", "eval")
        self.triples = triples                      # (N, 3)
        self.num_entities = num_entities
        self.neg_size = negative_sample_size
        self.mode = mode
        self.true_head = true_head or {}
        self.true_tail = true_tail or {}
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.triples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        h, r, t = self.triples[idx].tolist()

        if self.mode == "eval":
            return {"head": torch.tensor(h), "relation": torch.tensor(r), "tail": torch.tensor(t)}

        # ── Negative sampling (half corrupt head, half corrupt tail) ──────────
        neg_per_side = self.neg_size // 2

        neg_heads = self._sample_negatives(
            pos=h, r=r, t=t, side="head", n=neg_per_side
        )
        neg_tails = self._sample_negatives(
            pos=t, r=r, t=h, side="tail", n=neg_per_side
        )

        return {
            "head": torch.tensor(h, dtype=torch.long),
            "relation": torch.tensor(r, dtype=torch.long),
            "tail": torch.tensor(t, dtype=torch.long),
            "negative_head": torch.tensor(neg_heads, dtype=torch.long),  # (neg_per_side,)
            "negative_tail": torch.tensor(neg_tails, dtype=torch.long),
        }

    def _sample_negatives(
        self, pos: int, r: int, t: int, side: str, n: int
    ) -> np.ndarray:
        """Sample `n` negatives for the given side (head or tail)."""
        samples: list[int] = []
        true_set = (
            self.true_head.get((r, t), set()) if side == "head"
            else self.true_tail.get((pos, r), set())
        )
        while len(samples) < n:
            candidates = self.rng.integers(0, self.num_entities, size=n * 2)
            for c in candidates.tolist():
                if c not in true_set and c != pos:
                    samples.append(c)
                    if len(samples) == n:
                        break
        return np.array(samples[:n], dtype=np.int64)


def build_true_dicts(triples: torch.Tensor) -> tuple[Dict, Dict]:
    """
    Pre-compute (rel, tail) → {true heads} and (head, rel) → {true tails}
    for filtered evaluation (removes false negatives from rank computation).
    """
    true_head: Dict = {}
    true_tail: Dict = {}
    for h, r, t in triples.tolist():
        h, r, t = int(h), int(r), int(t)
        true_head.setdefault((r, t), set()).add(h)
        true_tail.setdefault((h, r), set()).add(t)
    return true_head, true_tail


def build_dataloaders(
    train_triples: torch.Tensor,
    val_triples: torch.Tensor,
    test_triples: torch.Tensor,
    num_entities: int,
    negative_sample_size: int,
    batch_size: int,
    num_workers: int = 4,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Build train/val/test DataLoaders for RotatE training."""
    # Build true dicts from ALL triples for filtered evaluation
    all_triples = torch.cat([train_triples, val_triples, test_triples], dim=0)
    true_head, true_tail = build_true_dicts(all_triples)

    train_ds = KGTripleDataset(
        train_triples, num_entities, negative_sample_size,
        mode="train", true_head=true_head, true_tail=true_tail
    )
    val_ds = KGTripleDataset(val_triples, num_entities, mode="eval")
    test_ds = KGTripleDataset(test_triples, num_entities, mode="eval")

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size * 4, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size * 4, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    return train_loader, val_loader, test_loader
