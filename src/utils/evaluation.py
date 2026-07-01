"""
Evaluation utilities for BioKG-LoRA.

Covers:
    - Link prediction (MRR, Hits@K) for RotatE (Stage 1)
    - Embedding clustering quality (Stage 1 validation)
    - Projection alignment (nearest-neighbour retrieval) for Stage 2
    - ROUGE-L, perplexity for language model (Stage 3)
    - Entity mention accuracy (Stage 3)
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


# ── Stage 1: Link Prediction Evaluation ──────────────────────────────────────

@torch.no_grad()
def evaluate_link_prediction(
    model,
    data_loader: DataLoader,
    num_entities: int,
    k_list: List[int] = [1, 3, 10],
    device: Optional[torch.device] = None,
    filter_true: Optional[Dict] = None,
) -> Dict[str, float]:
    """
    Evaluate link prediction with filtered MRR and Hits@K.

    For each test triple (h, r, t):
        1. Predict tail: score all entities as candidate tail
        2. Predict head: score all entities as candidate head
        3. Rank the true answer after removing other true answers (filtered)

    Args:
        filter_true: dict mapping (h, r) → set of true tails (for filtered eval)

    Returns:
        dict with keys: mrr, hits@1, hits@3, hits@10
    """
    if device is None:
        device = next(model.parameters()).device

    model.eval()
    ranks: List[float] = []

    for batch in data_loader:
        h = batch["head"].to(device)
        r = batch["relation"].to(device)
        t = batch["tail"].to(device)

        # ── Predict tail ──────────────────────────────────────────────────────
        scores = model.score_all_tails(h, r)   # (B, E)  lower = better

        for i in range(len(h)):
            true_tail = t[i].item()
            row = scores[i]  # (E,)

            # Filtered: mask out other true tails
            if filter_true is not None:
                key = (h[i].item(), r[i].item())
                true_set = filter_true.get(key, set())
                for tt in true_set:
                    if tt != true_tail:
                        row[tt] = float("inf")  # push them out of ranking

            rank = (row < row[true_tail]).sum().item() + 1
            ranks.append(rank)

        # ── Predict head ──────────────────────────────────────────────────────
        scores = model.score_all_heads(r, t)   # (B, E)

        for i in range(len(h)):
            true_head = h[i].item()
            row = scores[i]

            if filter_true is not None:
                key_inv = (t[i].item(), r[i].item())  # (tail, rel) → true heads
                true_set = filter_true.get(key_inv, set())
                for th in true_set:
                    if th != true_head:
                        row[th] = float("inf")

            rank = (row < row[true_head]).sum().item() + 1
            ranks.append(rank)

    ranks_arr = np.array(ranks, dtype=np.float32)
    mrr = float((1.0 / ranks_arr).mean())

    metrics = {"mrr": mrr}
    for k in k_list:
        metrics[f"hits@{k}"] = float((ranks_arr <= k).mean())

    logger.info("Link prediction: MRR=%.4f  " + "  ".join(f"Hits@{k}=%.4f" for k in k_list),
                mrr, *[metrics[f"hits@{k}"] for k in k_list])
    return metrics


# ── Stage 1: Embedding Clustering Quality ────────────────────────────────────

def evaluate_embedding_clustering(
    entity_embeddings: torch.Tensor,   # (E, d)
    entity2id: Dict[str, int],
    test_groups: Optional[Dict[str, List[str]]] = None,
) -> Dict[str, Dict]:
    """
    Test whether biologically related entities cluster together.

    For each group:
        intra_similarity: avg cosine sim within group
        random_similarity: avg cosine sim for random same-size group
        lift: intra / random  (>1 means clustering better than random)
    """
    if test_groups is None:
        test_groups = {
            "BMP_family":    ["Bmp2", "Bmp4", "Bmp7", "Bmp8a"],
            "Coagulation":   ["Thbd", "Proc", "F2", "F10"],
            "Kidney_markers": ["Pax2", "Wt1", "Six2", "Sall1"],
            "Liver_markers": ["Alb", "Hnf4a", "Cyp3a11", "Apoa1"],
        }

    results: Dict[str, Dict] = {}
    all_ids = list(entity2id.values())
    rng = np.random.default_rng(42)

    for group_name, members in test_groups.items():
        valid = [m for m in members if m in entity2id]
        if len(valid) < 2:
            continue
        ids = [entity2id[m] for m in valid]
        grp_emb = F.normalize(entity_embeddings[ids], dim=-1)    # (G, d)

        # Intra-group pairwise cosine similarity
        sims = grp_emb @ grp_emb.T   # (G, G)
        mask = ~torch.eye(len(ids), dtype=torch.bool)
        intra_sim = sims[mask].mean().item()

        # Random baseline
        rand_ids = rng.choice(all_ids, size=len(valid), replace=False).tolist()
        rand_emb = F.normalize(entity_embeddings[rand_ids], dim=-1)
        rand_sims = rand_emb @ rand_emb.T
        rand_sim = rand_sims[mask].mean().item()

        lift = intra_sim / (rand_sim + 1e-8)
        results[group_name] = {
            "members_found": len(valid),
            "intra_similarity": round(intra_sim, 4),
            "random_similarity": round(rand_sim, 4),
            "lift": round(lift, 3),
        }
        logger.info("%s: intra=%.4f  random=%.4f  lift=%.3f",
                    group_name, intra_sim, rand_sim, lift)

    return results


# ── Stage 2: Projection Alignment ────────────────────────────────────────────

@torch.no_grad()
def evaluate_projection_alignment(
    projection,
    kg_embs: torch.Tensor,      # (N, kg_dim)
    lm_embs: torch.Tensor,      # (N, lm_dim)
    device: torch.device,
    batch_size: int = 512,
) -> Dict[str, float]:
    """
    Nearest-neighbour retrieval accuracy for the projection layer.

    For each entity, projects its KG embedding and ranks the corresponding LM
    embedding among all N LM embeddings by cosine similarity.  A perfect
    projection would always rank the true target first (Recall@1 = 1.0).

    Returns:
        projection_mrr, projection_recall@1/5/10
    """
    projection.eval()
    N = kg_embs.shape[0]
    lm_norm = F.normalize(lm_embs.to(device).float(), dim=-1)   # (N, lm_dim)

    ranks: List[float] = []
    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        proj = F.normalize(
            projection(kg_embs[start:end].to(device)).float(), dim=-1
        )                                                          # (B, lm_dim)
        sims = proj @ lm_norm.T                                   # (B, N)
        for i in range(end - start):
            true_idx = start + i
            rank = int((sims[i] > sims[i, true_idx]).sum().item()) + 1
            ranks.append(rank)

    ranks_arr = np.array(ranks, dtype=np.float32)
    metrics = {
        "projection_mrr":       float((1.0 / ranks_arr).mean()),
        "projection_recall@1":  float((ranks_arr <= 1).mean()),
        "projection_recall@5":  float((ranks_arr <= 5).mean()),
        "projection_recall@10": float((ranks_arr <= 10).mean()),
    }
    logger.info(
        "Projection alignment: MRR=%.4f  R@1=%.4f  R@5=%.4f  R@10=%.4f",
        metrics["projection_mrr"], metrics["projection_recall@1"],
        metrics["projection_recall@5"], metrics["projection_recall@10"],
    )
    return metrics


# ── Stage 3: Language Model Evaluation ───────────────────────────────────────

@torch.no_grad()
def evaluate_perplexity(
    model,
    data_loader: DataLoader,
    device: Optional[torch.device] = None,
) -> float:
    """Compute average per-token perplexity on a data split."""
    if device is None:
        device = next(model.parameters()).device
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    for batch in data_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        entity_spans = batch.get("entity_spans")

        out = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            entity_spans=entity_spans,
        )
        # loss is mean over non-ignored tokens
        n_tokens = (labels != -100).sum().item()
        total_loss += out.loss.item() * n_tokens
        total_tokens += n_tokens

    ppl = float(np.exp(total_loss / max(total_tokens, 1)))
    logger.info("Perplexity: %.2f", ppl)
    return ppl


def evaluate_rouge(
    predictions: List[str],
    references: List[str],
    rouge_types: List[str] = ("rouge1", "rouge2", "rougeL"),
) -> Dict[str, float]:
    """Compute ROUGE scores."""
    try:
        from rouge_score import rouge_scorer as rouge_mod
    except ImportError:
        logger.warning("rouge_score not installed — skipping ROUGE. pip install rouge-score")
        return {}

    scorer = rouge_mod.RougeScorer(rouge_types, use_stemmer=True)
    agg: Dict[str, List[float]] = {rt: [] for rt in rouge_types}

    for pred, ref in zip(predictions, references):
        scores = scorer.score(ref, pred)
        for rt in rouge_types:
            agg[rt].append(scores[rt].fmeasure)

    return {rt: float(np.mean(vals)) for rt, vals in agg.items()}


def evaluate_entity_mention_accuracy(
    generated_texts: List[str],
    reference_entities: List[List[str]],
    entity_linker,
) -> Dict[str, float]:
    """
    Compute entity mention precision, recall, and F1.

    For each generated text, detect entities using the linker and compare
    against the reference entity list for that sample.
    """
    precisions, recalls, f1s = [], [], []

    for gen, refs in zip(generated_texts, reference_entities):
        detected = {e for e, _ in entity_linker.recognize(gen)}
        ref_set = set(refs)

        if not detected and not ref_set:
            precisions.append(1.0)
            recalls.append(1.0)
            f1s.append(1.0)
            continue

        tp = len(detected & ref_set)
        prec = tp / len(detected) if detected else 0.0
        rec = tp / len(ref_set) if ref_set else 0.0
        f1 = 2 * prec * rec / (prec + rec + 1e-8)

        precisions.append(prec)
        recalls.append(rec)
        f1s.append(f1)

    return {
        "entity_precision": float(np.mean(precisions)),
        "entity_recall": float(np.mean(recalls)),
        "entity_f1": float(np.mean(f1s)),
    }


# ── Combined evaluation report ───────────────────────────────────────────────

def print_metrics(metrics: Dict[str, float], title: str = "Evaluation Results") -> None:
    line = "=" * 50
    print(f"\n{line}")
    print(f"  {title}")
    print(line)
    for k, v in metrics.items():
        print(f"  {k:<25} {v:.4f}")
    print(line)
