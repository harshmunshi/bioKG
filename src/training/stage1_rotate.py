"""
Stage 1: RotatE Embedding Training

Trains the RotatE model on KG triples from Stage 0.
Saves:
    checkpoints/rotate/best.pt        ← full checkpoint (model + optimizer)
    data/kg/entity_embeddings.pt      ← (E, 256) entity embeddings
    data/kg/relation_embeddings.pt    ← (R, 128) relation embeddings
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.optim as optim

logger = logging.getLogger(__name__)


def run(cfg: Dict, resume: bool = False) -> None:
    """
    Entry point for Stage 1.

    Args:
        cfg:    full config dict
        resume: if True, resume from latest checkpoint in checkpoints/rotate/
    """
    paths = cfg["paths"]
    rot_cfg = cfg["rotate"]
    hw_cfg = cfg["hardware"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Stage 1 (RotatE) using device: %s", device)

    # ── Load KG ──────────────────────────────────────────────────────────────
    kg_dir = Path(paths["data_root"]) / "kg"
    train_triples = torch.load(str(kg_dir / "triples_train.pt"))
    val_triples = torch.load(str(kg_dir / "triples_val.pt"))
    test_triples = torch.load(str(kg_dir / "triples_test.pt"))

    with open(str(kg_dir / "entity2id.json"), encoding="utf-8") as f:
        entity2id = json.load(f)
    with open(str(kg_dir / "relation2id.json"), encoding="utf-8") as f:
        relation2id = json.load(f)

    num_entities = len(entity2id)
    num_relations = len(relation2id)
    logger.info("KG: %d entities, %d relations, %d train triples",
                num_entities, num_relations, len(train_triples))

    # ── DataLoaders ──────────────────────────────────────────────────────────
    from src.data.kg_dataset import build_dataloaders, build_true_dicts

    train_loader, val_loader, test_loader = build_dataloaders(
        train_triples, val_triples, test_triples,
        num_entities=num_entities,
        negative_sample_size=rot_cfg["negative_sample_size"],
        batch_size=rot_cfg["batch_size"],
        num_workers=hw_cfg.get("num_workers", 0),
    )

    # Build filtered eval dicts
    all_triples = torch.cat([train_triples, val_triples, test_triples])
    true_head, true_tail = build_true_dicts(all_triples)
    # Merge into single dict keyed by (h, r) → true tails & (r, t) → true heads
    filter_dict = {**{k: v for k, v in true_tail.items()},
                   **{k: v for k, v in true_head.items()}}

    # ── Model ────────────────────────────────────────────────────────────────
    from src.models.rotate import RotatE

    model = RotatE(
        num_entities=num_entities,
        num_relations=num_relations,
        embedding_dim=rot_cfg["embedding_dim"],
        margin=rot_cfg["margin"],
        epsilon=rot_cfg["epsilon"],
    ).to(device)

    # ── Optimizer ────────────────────────────────────────────────────────────
    optimizer = optim.Adam(
        model.parameters(),
        lr=rot_cfg["learning_rate"],
        weight_decay=rot_cfg.get("weight_decay", 0.0),
    )

    scheduler = _build_scheduler(optimizer, rot_cfg)

    # ── Resume ────────────────────────────────────────────────────────────────
    ckpt_dir = Path(paths["checkpoints"]["rotate"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    start_epoch = 0
    best_mrr = 0.0

    if resume:
        start_epoch, best_mrr = _resume(model, optimizer, ckpt_dir, device)

    # ── Training loop ────────────────────────────────────────────────────────
    from src.utils.evaluation import evaluate_link_prediction, print_metrics

    logger.info("Starting RotatE training for %d epochs", rot_cfg["max_epochs"])

    for epoch in range(start_epoch, rot_cfg["max_epochs"]):
        model.train()
        epoch_loss = 0.0
        t0 = time.time()

        for batch in train_loader:
            batch = _to_device(batch, device)
            loss, metrics = model.loss(
                batch,
                adversarial_temperature=rot_cfg.get("adversarial_temperature", 1.0),
                regularization=rot_cfg.get("regularization", 0.01),
            )
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += metrics["loss"]

        avg_loss = epoch_loss / len(train_loader)
        elapsed = time.time() - t0

        logger.info("Epoch %d/%d | loss=%.4f | %.1fs",
                    epoch + 1, rot_cfg["max_epochs"], avg_loss, elapsed)

        # ── Validation ───────────────────────────────────────────────────────
        if (epoch + 1) % rot_cfg["eval_every"] == 0:
            val_metrics = evaluate_link_prediction(
                model, val_loader, num_entities,
                k_list=rot_cfg.get("eval_k_list", [1, 3, 10]),
                device=device,
            )
            print_metrics(val_metrics, title=f"Epoch {epoch+1} Validation")

            if scheduler is not None:
                _step_scheduler(scheduler, rot_cfg["lr_scheduler"], val_metrics.get("mrr", 0))

            mrr = val_metrics.get("mrr", 0)
            if mrr > best_mrr:
                best_mrr = mrr
                _save_checkpoint(model, optimizer, epoch, mrr, ckpt_dir / "best.pt")
                logger.info("  ✓ New best MRR=%.4f — checkpoint saved", mrr)

        # ── Periodic save ────────────────────────────────────────────────────
        if (epoch + 1) % rot_cfg.get("save_every", 50) == 0:
            _save_checkpoint(model, optimizer, epoch, best_mrr,
                             ckpt_dir / f"epoch_{epoch+1:04d}.pt")

    # ── Final evaluation ──────────────────────────────────────────────────────
    logger.info("Loading best checkpoint for final test evaluation…")
    best_ckpt = torch.load(str(ckpt_dir / "best.pt"), map_location=device)
    model.load_state_dict(best_ckpt["model"])
    test_metrics = evaluate_link_prediction(
        model, test_loader, num_entities,
        k_list=rot_cfg.get("eval_k_list", [1, 3, 10]),
        device=device,
    )
    print_metrics(test_metrics, title="TEST SET RESULTS")

    # ── Extract and save embeddings ───────────────────────────────────────────
    _save_embeddings(model, kg_dir)
    logger.info("Stage 1 complete. Best MRR: %.4f", best_mrr)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _build_scheduler(optimizer, rot_cfg: Dict):
    sched_type = rot_cfg.get("lr_scheduler", "plateau")
    if sched_type == "plateau":
        return optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max",
            factor=rot_cfg.get("lr_factor", 0.5),
            patience=rot_cfg.get("lr_patience", 5),
        )
    elif sched_type == "cosine":
        return optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=rot_cfg["max_epochs"]
        )
    return None


def _step_scheduler(scheduler, sched_type: str, metric: float) -> None:
    if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
        scheduler.step(metric)
    else:
        scheduler.step()


def _to_device(batch: Dict, device: torch.device) -> Dict:
    return {k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()}


def _save_checkpoint(model, optimizer, epoch: int, mrr: float, path: Path) -> None:
    torch.save({
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "mrr": mrr,
    }, str(path))


def _resume(model, optimizer, ckpt_dir: Path, device: torch.device) -> tuple[int, float]:
    ckpt_path = ckpt_dir / "best.pt"
    if not ckpt_path.exists():
        logger.info("No checkpoint found at %s — starting from scratch", ckpt_path)
        return 0, 0.0
    ckpt = torch.load(str(ckpt_path), map_location=device)
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    start = ckpt["epoch"] + 1
    mrr = ckpt.get("mrr", 0.0)
    logger.info("Resumed from epoch %d (best MRR=%.4f)", start, mrr)
    return start, mrr


def _save_embeddings(model, kg_dir: Path) -> None:
    entity_embs = model.get_entity_embeddings()   # (E, d)
    relation_embs = model.get_relation_embeddings()  # (R, d/2)
    torch.save(entity_embs, str(kg_dir / "entity_embeddings.pt"))
    torch.save(relation_embs, str(kg_dir / "relation_embeddings.pt"))
    logger.info("Entity embeddings saved: %s", entity_embs.shape)
    logger.info("Relation embeddings saved: %s", relation_embs.shape)

    # Optional: clustering validation
    try:
        import json
        with open(str(kg_dir / "entity2id.json"), encoding="utf-8") as f:
            entity2id = json.load(f)
        from src.utils.evaluation import evaluate_embedding_clustering
        cluster_metrics = evaluate_embedding_clustering(entity_embs, entity2id)
        for group, metrics in cluster_metrics.items():
            logger.info("Clustering %s: lift=%.3f", group, metrics["lift"])
    except Exception as e:
        logger.warning("Clustering eval skipped: %s", e)
