"""
RotatE: Knowledge Graph Embedding by Relational Rotation in Complex Space.

Reference: Sun et al. "RotatE: Knowledge Graph Embedding by Relational
           Rotation in Complex Space." ICLR 2019.

Key idea:
    - Entities: h, t ∈ ℂ^(d/2)  (represented as real vectors of size d)
    - Relations: r ∈ [0, 2π)^(d/2)  (phase angles)
    - Score: d(h ∘ r, t) = ||h ∘ r - t||   (lower = more probable)
    - ∘ denotes element-wise complex multiplication

Properties modelled:
    - Symmetry:     interacts_with  (θ = π → rotation by π)
    - Antisymmetry: regulates       (θ ≠ π)
    - Inversion:    causes / caused_by  (θ → -θ)
    - Composition:  multi-hop (compose rotations)
"""
from __future__ import annotations

import logging
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class RotatE(nn.Module):
    """
    RotatE knowledge graph embedding model.

    Args:
        num_entities:  Total number of entities in the KG
        num_relations: Total number of relation types
        embedding_dim: Embedding dimension d (must be even)
        margin:        Gamma — boundary between positive and negative scores
        epsilon:       Initialisation scale
    """

    def __init__(
        self,
        num_entities: int,
        num_relations: int,
        embedding_dim: int = 256,
        margin: float = 9.0,
        epsilon: float = 2.0,
    ):
        super().__init__()
        assert embedding_dim % 2 == 0, "embedding_dim must be even"

        self.num_entities = num_entities
        self.num_relations = num_relations
        self.embedding_dim = embedding_dim
        self.margin = margin
        self.epsilon = epsilon

        # Entity embeddings stored as real vectors of dim d
        # (real and imaginary parts interleaved: [re_0, im_0, re_1, im_1, ...])
        self.entity_embedding = nn.Embedding(num_entities, embedding_dim)

        # Relation embeddings stored as phase angles (dim d/2)
        self.relation_embedding = nn.Embedding(num_relations, embedding_dim // 2)

        self._init_weights()

    def _init_weights(self) -> None:
        half = self.embedding_dim // 2
        nn.init.uniform_(
            self.entity_embedding.weight,
            -self.epsilon / self.embedding_dim,
            self.epsilon / self.embedding_dim,
        )
        nn.init.uniform_(
            self.relation_embedding.weight,
            -self.epsilon / half,
            self.epsilon / half,
        )

    # ── Scoring ───────────────────────────────────────────────────────────────

    def score_triples(
        self,
        head: torch.Tensor,
        relation: torch.Tensor,
        tail: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute RotatE score for a batch of triples.

        Args:
            head:     (B,) entity IDs
            relation: (B,) relation IDs
            tail:     (B,) entity IDs

        Returns:
            scores: (B,)  — lower is better (more probable triple)
        """
        h = self.entity_embedding(head)      # (B, d)
        r = self.relation_embedding(relation)  # (B, d/2)
        t = self.entity_embedding(tail)      # (B, d)

        re_h, im_h = torch.chunk(h, 2, dim=-1)   # (B, d/2)
        re_t, im_t = torch.chunk(t, 2, dim=-1)

        # Relation as rotation: e^{iθ}
        phase = r / (self.margin / torch.pi)
        re_r = torch.cos(phase)   # (B, d/2)
        im_r = torch.sin(phase)

        # Complex multiply: h ∘ r = (re_h + i·im_h)(re_r + i·im_r)
        re_hr = re_h * re_r - im_h * im_r
        im_hr = re_h * im_r + im_h * re_r

        # Distance to tail
        re_d = re_hr - re_t
        im_d = im_hr - im_t
        score = torch.sqrt(re_d ** 2 + im_d ** 2 + 1e-8).sum(dim=-1)   # (B,)
        return score

    def score_all_tails(
        self,
        head: torch.Tensor,
        relation: torch.Tensor,
        entity_chunk_size: int = 256,
    ) -> torch.Tensor:
        """
        Score (head, relation) against all entities (for link prediction eval).

        Args:
            head:     (B,) entity IDs
            relation: (B,) relation IDs
            entity_chunk_size: entities scored per chunk — bounds memory to
                O(B * entity_chunk_size) instead of O(B * num_entities), which
                otherwise blows up for large KGs with large eval batches.

        Returns:
            scores: (B, num_entities)
        """
        h = self.entity_embedding(head)      # (B, d)
        r = self.relation_embedding(relation)  # (B, d/2)
        all_t = self.entity_embedding.weight   # (E, d)
        num_entities = all_t.shape[0]

        re_h, im_h = torch.chunk(h, 2, dim=-1)     # (B, d/2)

        phase = r / (self.margin / torch.pi)
        re_r = torch.cos(phase)   # (B, d/2)
        im_r = torch.sin(phase)

        re_hr = re_h * re_r - im_h * im_r   # (B, d/2)
        im_hr = re_h * im_r + im_h * re_r

        scores = torch.empty(head.shape[0], num_entities, device=head.device, dtype=re_hr.dtype)
        for start in range(0, num_entities, entity_chunk_size):
            end = min(start + entity_chunk_size, num_entities)
            re_at, im_at = torch.chunk(all_t[start:end], 2, dim=-1)  # (chunk, d/2)

            # Broadcast: (B, 1, d/2) - (1, chunk, d/2)
            re_d = re_hr.unsqueeze(1) - re_at.unsqueeze(0)
            im_d = im_hr.unsqueeze(1) - im_at.unsqueeze(0)
            scores[:, start:end] = torch.sqrt(re_d ** 2 + im_d ** 2 + 1e-8).sum(dim=-1)
        return scores

    def score_all_heads(
        self,
        relation: torch.Tensor,
        tail: torch.Tensor,
        entity_chunk_size: int = 256,
    ) -> torch.Tensor:
        """
        Score all entities as head for (?, relation, tail).

        Args:
            entity_chunk_size: see `score_all_tails`.

        Returns:
            scores: (B, num_entities)
        """
        r = self.relation_embedding(relation)  # (B, d/2)
        t = self.entity_embedding(tail)        # (B, d)
        all_h = self.entity_embedding.weight   # (E, d)
        num_entities = all_h.shape[0]

        re_t, im_t = torch.chunk(t, 2, dim=-1)

        phase = r / (self.margin / torch.pi)
        re_r = torch.cos(phase)
        im_r = torch.sin(phase)

        # Inverse rotation: h ≈ t ∘ r^{-1}  (conjugate)
        re_t_inv = re_t * re_r + im_t * im_r   # (B, d/2)
        im_t_inv = im_t * re_r - re_t * im_r

        scores = torch.empty(tail.shape[0], num_entities, device=tail.device, dtype=re_t_inv.dtype)
        for start in range(0, num_entities, entity_chunk_size):
            end = min(start + entity_chunk_size, num_entities)
            re_ah, im_ah = torch.chunk(all_h[start:end], 2, dim=-1)  # (chunk, d/2)

            re_d = re_ah.unsqueeze(0) - re_t_inv.unsqueeze(1)
            im_d = im_ah.unsqueeze(0) - im_t_inv.unsqueeze(1)
            scores[:, start:end] = torch.sqrt(re_d ** 2 + im_d ** 2 + 1e-8).sum(dim=-1)
        return scores

    # ── Loss ─────────────────────────────────────────────────────────────────

    def loss(
        self,
        batch: Dict[str, torch.Tensor],
        adversarial_temperature: float = 1.0,
        regularization: float = 0.01,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Self-adversarial negative sampling loss (Sun et al. 2019, Eq. 3).

        ℒ = -log σ(γ - d(h, r, t))
            - Σᵢ p(h'ᵢ|h,r,t) · log σ(d(h'ᵢ, r, t) - γ)

        where p(h') = softmax(-α · d(h', r, t))  (harder negatives → higher weight)
        """
        h, r, t = batch["head"], batch["relation"], batch["tail"]
        neg_h = batch["negative_head"]   # (B, K)
        neg_t = batch["negative_tail"]   # (B, K)
        B, K = neg_h.shape

        # ── Positive score ──────────────────────────────────────────────────
        pos_score = self.score_triples(h, r, t)   # (B,)

        # ── Negative scores (corrupt head) ──────────────────────────────────
        h_exp = h.unsqueeze(1).expand(B, K).reshape(B * K)
        r_exp = r.unsqueeze(1).expand(B, K).reshape(B * K)
        t_exp = t.unsqueeze(1).expand(B, K).reshape(B * K)

        neg_h_flat = neg_h.reshape(B * K)
        neg_t_flat = neg_t.reshape(B * K)

        neg_score_h = self.score_triples(neg_h_flat, r_exp, t_exp).view(B, K)
        neg_score_t = self.score_triples(h_exp, r_exp, neg_t_flat).view(B, K)
        neg_score = torch.cat([neg_score_h, neg_score_t], dim=1)   # (B, 2K)

        # ── Self-adversarial weights ─────────────────────────────────────────
        weights = F.softmax(-adversarial_temperature * neg_score, dim=1).detach()

        # ── Margin-based sigmoid loss ────────────────────────────────────────
        gamma = self.margin
        pos_loss = F.logsigmoid(gamma - pos_score).mean()
        neg_loss = (weights * F.logsigmoid(neg_score - gamma)).sum(dim=1).mean()
        main_loss = -(pos_loss + neg_loss)

        # ── Regularisation ───────────────────────────────────────────────────
        reg = (
            self.entity_embedding.weight.norm(p=2, dim=1).mean()
            + self.relation_embedding.weight.norm(p=2, dim=1).mean()
        )
        total_loss = main_loss + regularization * reg

        return total_loss, {
            "loss": total_loss.item(),
            "main_loss": main_loss.item(),
            "pos_loss": pos_loss.item(),
            "neg_loss": neg_loss.item(),
            "reg": reg.item(),
        }

    # ── Utilities ─────────────────────────────────────────────────────────────

    def get_entity_embeddings(self) -> torch.Tensor:
        """Return a detached CPU copy of entity embeddings (num_entities, d)."""
        return self.entity_embedding.weight.data.detach().cpu()

    def get_relation_embeddings(self) -> torch.Tensor:
        """Return a detached CPU copy of relation embeddings (num_relations, d/2)."""
        return self.relation_embedding.weight.data.detach().cpu()

    def nearest_entities(
        self,
        query_ids: torch.Tensor,
        k: int = 10,
        id2entity: Optional[Dict[int, str]] = None,
    ) -> Dict[int, list]:
        """Find k nearest entities in embedding space (by cosine similarity)."""
        embs = F.normalize(self.entity_embedding.weight, dim=-1)   # (E, d)
        q_embs = embs[query_ids]   # (Q, d)
        sims = q_embs @ embs.T    # (Q, E)
        topk = torch.topk(sims, k=k + 1, dim=-1)   # include self
        results = {}
        for i, qid in enumerate(query_ids.tolist()):
            neighbors = []
            for j, score in zip(topk.indices[i].tolist(), topk.values[i].tolist()):
                if j != qid:
                    name = id2entity[j] if id2entity else str(j)
                    neighbors.append((name, round(score, 4)))
            results[qid] = neighbors[:k]
        return results
