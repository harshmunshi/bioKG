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
from typing import Dict

import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

logger = logging.getLogger(__name__)


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

    # ── Build DataLoader ─────────────────────────────────────────────────────
    all_ids = torch.arange(num_entities)
    dataset = TensorDataset(all_ids, entity_embs, lm_embs)
    n_workers = hw_cfg.get("num_workers", 0)
    loader = DataLoader(
        dataset,
        batch_size=proj_cfg["batch_size"],
        shuffle=True,
        num_workers=n_workers,
        pin_memory=(n_workers > 0),
        drop_last=True,
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

    for epoch in range(start_epoch, proj_cfg["max_epochs"]):
        projection.train()
        epoch_loss = 0.0
        epoch_acc = 0.0
        t0 = time.time()

        for batch_ids, kg_batch, lm_batch in loader:
            kg_batch = kg_batch.to(device)
            lm_batch = lm_batch.to(device)

            kg_projected = projection(kg_batch)   # (B, lm_dim)
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

        logger.info(
            "Epoch %d/%d | loss=%.4f | acc=%.4f | %.1fs",
            epoch + 1, proj_cfg["max_epochs"], avg_loss, avg_acc, elapsed,
        )

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                "epoch": epoch,
                "projection": projection.state_dict(),
                "loss_fn": loss_fn.state_dict(),
                "optimizer": optimizer.state_dict(),
                "best_loss": best_loss,
            }, str(ckpt_dir / "best.pt"))
            # Also save just the projection weights for easy loading in Stage 3
            torch.save(projection.state_dict(), str(ckpt_dir / "projection_weights.pt"))
            logger.info("  ✓ New best loss=%.4f — checkpoint saved", best_loss)

    logger.info("Stage 2 complete. Best contrastive loss: %.4f", best_loss)

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


def _resume(projection, optimizer, ckpt_dir: Path, device: torch.device) -> tuple[int, float]:
    ckpt = torch.load(str(ckpt_dir / "best.pt"), map_location=device)
    projection.load_state_dict(ckpt["projection"])
    optimizer.load_state_dict(ckpt["optimizer"])
    epoch = ckpt.get("epoch", 0) + 1
    best_loss = ckpt.get("best_loss", float("inf"))
    logger.info("Resumed from epoch %d (best loss=%.4f)", epoch, best_loss)
    return epoch, best_loss
