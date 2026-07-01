"""
Stage 2: KG → LM Projection Layer Training

Trains the projection network that maps RotatE embeddings (256-dim) into
LLM token-embedding space (4096-dim) using InfoNCE contrastive loss.

Input:  data/kg/entity_embeddings.pt  (from Stage 1)
Output: checkpoints/projection/best.pt
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Dict, Tuple

import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, TensorDataset

logger = logging.getLogger(__name__)


def _make_writer(log_dir: Path):
    try:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(log_dir=str(log_dir))
        logger.info("TensorBoard writer → %s", log_dir)
        return writer
    except Exception as e:
        logger.warning("TensorBoard unavailable (%s) — skipping", e)
        return None


def run(cfg: Dict, resume: bool = False) -> None:
    """
    Entry point for Stage 2.

    Args:
        cfg:    full config dict
        resume: resume from checkpoint if available
    """
    from src.utils.model_profile import resolve_model_profile
    model_profile = resolve_model_profile(cfg)

    paths = cfg["paths"]
    proj_cfg = cfg["projection"]
    hw_cfg = cfg["hardware"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Stage 2 (Projection) using device: %s", device)

    log_dir = Path(paths["logs"])
    log_dir.mkdir(parents=True, exist_ok=True)
    writer = _make_writer(log_dir / "stage2")

    # ── Load KG embeddings (frozen, from Stage 1) ─────────────────────────────
    kg_dir = Path(paths["data_root"]) / "kg"
    entity_embs = torch.load(str(kg_dir / "entity_embeddings.pt"))   # (E, kg_dim)

    with open(str(kg_dir / "entity2id.json"), encoding="utf-8") as f:
        entity2id: Dict[str, int] = json.load(f)

    id2entity = {v: k for k, v in entity2id.items()}
    num_entities = entity_embs.shape[0]
    logger.info("Loaded %d entity embeddings (dim=%d)", num_entities, entity_embs.shape[1])

    # ── Load base LLM (frozen) ────────────────────────────────────────────────
    base_model_name = cfg["lora"]["base_model"]
    logger.info("Loading base LLM for embedding extraction: %s", base_model_name)
    base_llm, tokenizer = _load_frozen_llm(base_model_name, cfg["lora"])

    # ── Pre-compute all LM embeddings (do once, not every batch) ─────────────
    logger.info("Pre-computing LM embeddings for all %d entities…", num_entities)
    lm_embs = _precompute_lm_embeddings(
        base_llm, tokenizer, id2entity, num_entities, device,
        batch_size=512,
    )  # (E, lm_dim) on CPU
    logger.info("LM embeddings pre-computed: %s", lm_embs.shape)

    # ── Train / val split ────────────────────────────────────────────────────
    val_fraction = proj_cfg.get("val_fraction", 0.1)
    all_ids = torch.arange(num_entities)
    dataset = TensorDataset(all_ids, entity_embs, lm_embs)

    rng = torch.Generator().manual_seed(42)
    n_val = max(1, int(num_entities * val_fraction))
    n_train = num_entities - n_val
    perm = torch.randperm(num_entities, generator=rng)
    train_idx, val_idx = perm[:n_train], perm[n_train:]
    logger.info("Split: %d train / %d val entities", n_train, n_val)

    n_workers = hw_cfg.get("num_workers", 0)
    loader = DataLoader(
        Subset(dataset, train_idx.tolist()),
        batch_size=proj_cfg["batch_size"],
        shuffle=True,
        num_workers=n_workers,
        pin_memory=(n_workers > 0),
        drop_last=True,
    )
    val_loader = DataLoader(
        Subset(dataset, val_idx.tolist()),
        batch_size=proj_cfg["batch_size"],
        shuffle=False,
        num_workers=n_workers,
        pin_memory=(n_workers > 0),
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    from src.models.projection import KGProjectionLayer, ProjectionAlignmentLoss

    projection = KGProjectionLayer(
        kg_dim=proj_cfg["kg_dim"],
        hidden_dim=proj_cfg["hidden_dim"],
        lm_dim=model_profile["lm_dim"],
        dropout=proj_cfg["dropout"],
    ).to(device)

    loss_fn = ProjectionAlignmentLoss(temperature=proj_cfg["temperature"]).to(device)

    # ── Optimizer ─────────────────────────────────────────────────────────────
    params = list(projection.parameters()) + list(loss_fn.parameters())
    optimizer = optim.AdamW(
        params,
        lr=proj_cfg["learning_rate"],
        weight_decay=proj_cfg["weight_decay"],
    )
    max_steps = len(loader) * proj_cfg["max_epochs"]
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=proj_cfg["learning_rate"],
        total_steps=max_steps,
        pct_start=proj_cfg.get("warmup_steps", 50) / max_steps,
    )

    # ── Checkpoint dir ────────────────────────────────────────────────────────
    ckpt_dir = Path(paths["checkpoints"]["projection"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    start_epoch = 0
    best_loss = float("inf")

    if resume and (ckpt_dir / "best.pt").exists():
        start_epoch, best_loss = _resume(projection, optimizer, ckpt_dir, device)

    # ── Training loop ─────────────────────────────────────────────────────────
    logger.info("Starting projection training for %d epochs", proj_cfg["max_epochs"])
    from src.utils.evaluation import evaluate_projection_alignment, print_metrics

    history = []

    for epoch in range(start_epoch, proj_cfg["max_epochs"]):
        projection.train()
        epoch_loss = 0.0
        epoch_acc = 0.0
        t0 = time.time()

        for batch_ids, kg_batch, lm_batch in loader:
            kg_batch = kg_batch.to(device)
            lm_batch = lm_batch.to(device)

            kg_projected = projection(kg_batch)
            out = loss_fn(kg_projected, lm_batch)
            loss = out["loss"]

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()
            epoch_acc += out["accuracy"].item()

        avg_loss = epoch_loss / len(loader)
        avg_acc = epoch_acc / len(loader)
        elapsed = time.time() - t0

        # ── Validation ───────────────────────────────────────────────────────
        val_loss, val_acc = _evaluate_val(projection, loss_fn, val_loader, device)

        # Nearest-neighbour retrieval on val entities
        val_ids_list = val_idx.tolist()
        align_metrics = evaluate_projection_alignment(
            projection,
            kg_embs=entity_embs[val_ids_list],
            lm_embs=lm_embs[val_ids_list],
            device=device,
        )
        print_metrics(
            {"val_loss": val_loss, "val_acc": val_acc, **align_metrics},
            title=f"Epoch {epoch+1} Validation",
        )

        logger.info(
            "Epoch %d/%d | train_loss=%.4f | train_acc=%.4f | val_loss=%.4f | %.1fs",
            epoch + 1, proj_cfg["max_epochs"], avg_loss, avg_acc, val_loss, elapsed,
        )

        if writer:
            writer.add_scalar("train/loss", avg_loss, epoch + 1)
            writer.add_scalar("train/accuracy", avg_acc, epoch + 1)
            writer.add_scalar("train/lr", optimizer.param_groups[0]["lr"], epoch + 1)
            writer.add_scalar("val/loss", val_loss, epoch + 1)
            writer.add_scalar("val/accuracy", val_acc, epoch + 1)
            for k, v in align_metrics.items():
                writer.add_scalar(f"val/{k}", v, epoch + 1)

        history.append({
            "epoch": epoch + 1,
            "train_loss": avg_loss, "train_acc": avg_acc,
            "val_loss": val_loss, "val_acc": val_acc,
            **align_metrics,
        })

        if val_loss < best_loss:
            best_loss = val_loss
            torch.save({
                "epoch": epoch,
                "projection": projection.state_dict(),
                "loss_fn": loss_fn.state_dict(),
                "optimizer": optimizer.state_dict(),
                "best_loss": best_loss,
            }, str(ckpt_dir / "best.pt"))
            torch.save(projection.state_dict(), str(ckpt_dir / "projection_weights.pt"))
            logger.info("  ✓ New best val_loss=%.4f — checkpoint saved", best_loss)

    if writer:
        writer.close()

    logger.info("Stage 2 complete. Best val loss: %.4f", best_loss)

    # ── Save metrics JSON ─────────────────────────────────────────────────────
    metrics_path = log_dir / "stage2_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump({"history": history}, f, indent=2)
    logger.info("Metrics saved to %s", metrics_path)

    # Copy projection weights to standard location
    import shutil
    src = str(ckpt_dir / "projection_weights.pt")
    dst = str(Path(paths["checkpoints"]["projection"]) / "projection_weights.pt")
    if src != dst:
        shutil.copy(src, dst)
    logger.info("Projection weights saved to: %s", dst)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _load_frozen_llm(model_name: str, lora_cfg: Dict):
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    import torch

    bnb = None
    if lora_cfg.get("load_in_4bit"):
        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb,
        device_map="auto",
        trust_remote_code=True,
    )
    for p in model.parameters():
        p.requires_grad = False
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


@torch.no_grad()
def _precompute_lm_embeddings(
    base_llm,
    tokenizer,
    id2entity: Dict[int, str],
    num_entities: int,
    device: torch.device,
    batch_size: int = 512,
) -> torch.Tensor:
    """Pre-compute LM embeddings for all entities by mean-pooling name tokens."""
    lm_dim = base_llm.get_input_embeddings().weight.shape[1]
    all_lm_embs = torch.zeros(num_entities, lm_dim)

    for start in range(0, num_entities, batch_size):
        end = min(start + batch_size, num_entities)
        names = [id2entity[i] for i in range(start, end)]
        enc = tokenizer(
            names, return_tensors="pt", padding=True,
            truncation=True, max_length=32,
        )
        input_ids = enc["input_ids"].to(device)
        mask = enc["attention_mask"].to(device)

        token_embs = base_llm.get_input_embeddings()(input_ids)   # (B, L, d)
        mask_f = mask.unsqueeze(-1).float()
        pooled = (token_embs * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp(min=1)
        all_lm_embs[start:end] = pooled.cpu().float()

        if (start // batch_size) % 10 == 0:
            logger.info("  LM embedding pre-compute: %d/%d", end, num_entities)

    return all_lm_embs


@torch.no_grad()
def _evaluate_val(projection, loss_fn, val_loader, device) -> Tuple[float, float]:
    projection.eval()
    total_loss, total_acc, n = 0.0, 0.0, 0
    for _, kg_batch, lm_batch in val_loader:
        kg_batch = kg_batch.to(device)
        lm_batch = lm_batch.to(device)
        out = loss_fn(projection(kg_batch), lm_batch)
        total_loss += out["loss"].item()
        total_acc += out["accuracy"].item()
        n += 1
    return total_loss / max(n, 1), total_acc / max(n, 1)


def _resume(projection, optimizer, ckpt_dir: Path, device: torch.device) -> tuple[int, float]:
    ckpt = torch.load(str(ckpt_dir / "best.pt"), map_location=device)
    projection.load_state_dict(ckpt["projection"])
    optimizer.load_state_dict(ckpt["optimizer"])
    epoch = ckpt.get("epoch", 0) + 1
    best_loss = ckpt.get("best_loss", float("inf"))
    logger.info("Resumed from epoch %d (best loss=%.4f)", epoch, best_loss)
    return epoch, best_loss
