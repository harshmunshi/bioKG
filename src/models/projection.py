"""
Stage 2: KG → LM Projection Layer.

Learns to map RotatE entity embeddings (256-dim) into the LLM token-embedding
space (4096-dim for Llama-3-8B) using contrastive (InfoNCE) alignment.

The intuition: after training, the RotatE embedding of "Thbd" and the LLM
token embedding of the token "Thbd" should be close in LM space.
"""
from __future__ import annotations

import logging
import math
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class KGProjectionLayer(nn.Module):
    """
    Two-layer MLP with LayerNorm that projects KG embeddings into LM space.

    Architecture:
        Linear(kg_dim → hidden_dim)
        LayerNorm + GELU + Dropout
        Linear(hidden_dim → lm_dim)

    Args:
        kg_dim:     Input dimension (RotatE embedding size = 256)
        hidden_dim: Intermediate dimension (1024)
        lm_dim:     Output dimension (LLM hidden size = 4096)
        dropout:    Dropout rate
    """

    def __init__(
        self,
        kg_dim: int = 256,
        hidden_dim: int = 1024,
        lm_dim: int = 4096,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.kg_dim = kg_dim
        self.lm_dim = lm_dim

        self.net = nn.Sequential(
            nn.Linear(kg_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, lm_dim),
        )
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, kg_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Args:
            kg_embeddings: (B, kg_dim) or (kg_dim,) RotatE embeddings

        Returns:
            projected: (B, lm_dim) embeddings in LM space
        """
        return self.net(kg_embeddings)


class ProjectionAlignmentLoss(nn.Module):
    """
    InfoNCE contrastive loss for aligning KG embeddings with LM embeddings.

    Given a batch of (kg_embedding, lm_embedding) pairs for the same entity,
    maximises similarity of matching pairs and minimises similarity of
    non-matching pairs (cross-modal contrastive objective).

    Args:
        temperature: Softmax temperature τ (lower = sharper distribution)
    """

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature
        self.log_scale = nn.Parameter(torch.ones([]) * math.log(1 / temperature))

    def forward(
        self,
        kg_projected: torch.Tensor,    # (B, lm_dim) — projected KG embeddings
        lm_embeddings: torch.Tensor,   # (B, lm_dim) — LM token embeddings for same entity
    ) -> Dict[str, torch.Tensor]:
        """
        Symmetric InfoNCE loss (both KG→LM and LM→KG directions).

        Returns dict with 'loss', 'accuracy'.
        """
        B = kg_projected.shape[0]

        # L2-normalise both sides
        kg_norm = F.normalize(kg_projected, dim=-1)    # (B, d)
        lm_norm = F.normalize(lm_embeddings, dim=-1)  # (B, d)

        # Scaled dot-product similarity matrix
        scale = self.log_scale.exp().clamp(max=100)
        logits = scale * (kg_norm @ lm_norm.T)         # (B, B)

        # Diagonal = positive pairs
        labels = torch.arange(B, device=logits.device)

        # Symmetric loss
        loss_kg2lm = F.cross_entropy(logits, labels)
        loss_lm2kg = F.cross_entropy(logits.T, labels)
        loss = (loss_kg2lm + loss_lm2kg) / 2

        # Accuracy (top-1 retrieval)
        acc = (logits.argmax(dim=-1) == labels).float().mean()

        return {"loss": loss, "accuracy": acc}


def build_projection_batch(
    entity_ids: torch.Tensor,          # (B,) entity IDs to use
    kg_embeddings: torch.Tensor,       # (E, kg_dim) full entity embedding table
    base_llm,                          # frozen LLM model
    tokenizer,                         # LLM tokenizer
    id2entity: Dict[int, str],         # entity ID → name string
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Build a batch of (kg_embedding, lm_embedding) pairs for projection training.

    For each entity in the batch:
        kg_emb  = kg_embeddings[entity_id]
        lm_emb  = mean of LLM input embeddings for the entity's name tokens
    """
    entity_ids_list = entity_ids.tolist()
    entity_names = [id2entity[eid] for eid in entity_ids_list]

    # KG embeddings
    kg_batch = kg_embeddings[entity_ids].to(device)   # (B, kg_dim)

    # LM embeddings (frozen — no grad)
    with torch.no_grad():
        enc = tokenizer(
            entity_names,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=16,
        )
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)

        token_embs = base_llm.get_input_embeddings()(input_ids)   # (B, L, d)
        # Mean-pool over non-padding tokens
        mask = attention_mask.unsqueeze(-1).float()
        lm_batch = (token_embs * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        # (B, lm_dim)

    return kg_batch, lm_batch
