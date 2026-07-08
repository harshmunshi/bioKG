"""
Stage 3: BioKG-LoRA Fine-tuning

Fine-tunes the LLM with LoRA adapters on biological QA pairs,
with KG embeddings injected at entity token positions.

Input:  QA dataset from Stage 0 (data/qa/)
        Projection weights from Stage 2 (checkpoints/projection/)
Output: LoRA adapters + projection (checkpoints/lora/)
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.optim as optim

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
    Entry point for Stage 3.

    Args:
        cfg:    full config dict
        resume: resume from checkpoint if available
    """
    paths = cfg["paths"]
    lora_cfg = cfg["lora"]
    hw_cfg = cfg["hardware"]
    eval_cfg = cfg["evaluation"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Stage 3 (LoRA fine-tuning) using device: %s", device)

    log_dir = Path(paths["logs"])
    log_dir.mkdir(parents=True, exist_ok=True)
    writer = _make_writer(log_dir / "stage3")

    # ── Tokeniser ────────────────────────────────────────────────────────────
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        lora_cfg["base_model"], trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # ── Entity linker ─────────────────────────────────────────────────────────
    kg_dir = Path(paths["data_root"]) / "kg"
    entity_linker = None
    entity2id_path = str(kg_dir / "entity2id.json")
    if Path(entity2id_path).exists():
        from src.utils.entity_linker import EntityLinker
        entity_linker = EntityLinker.from_file(
            entity2id_path,
            threshold=lora_cfg.get("entity_linking_threshold", 0.85),
        )

    # ── DataLoaders ───────────────────────────────────────────────────────────
    from src.data.qa_dataset import build_qa_dataloaders

    qa_dir = Path(paths["data_root"]) / "qa"
    train_loader, val_loader = build_qa_dataloaders(
        train_path=str(qa_dir / "train.jsonl"),
        val_path=str(qa_dir / "val.jsonl"),
        tokenizer=tokenizer,
        max_length=lora_cfg["max_seq_length"],
        batch_size=lora_cfg["batch_size"],
        num_workers=hw_cfg.get("num_workers", 0),
        entity_linker=entity_linker,
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    proj_ckpt = str(Path(paths["checkpoints"]["projection"]) / "projection_weights.pt")
    kg_emb_path = str(kg_dir / "entity_embeddings.pt")

    from src.models.biokg_lora import BioKGLoRA
    from src.utils.model_profile import resolve_model_profile

    model_profile = resolve_model_profile(cfg)

    model = BioKGLoRA(
        base_model_name=lora_cfg["base_model"],
        kg_embedding_path=kg_emb_path if Path(kg_emb_path).exists() else None,
        entity2id_path=entity2id_path if Path(entity2id_path).exists() else None,
        projection_ckpt=proj_ckpt if Path(proj_ckpt).exists() else None,
        kg_dim=cfg["projection"]["kg_dim"],
        lm_dim=model_profile["lm_dim"],
        kg_weight=lora_cfg.get("kg_weight", 0.3),
        lora_rank=lora_cfg["lora_rank"],
        lora_alpha=lora_cfg["lora_alpha"],
        lora_dropout=lora_cfg["lora_dropout"],
        quantization=lora_cfg.get("quantization"),
        target_modules=model_profile["target_modules"],
    )

    # ── Optimizer ─────────────────────────────────────────────────────────────
    trainable_params = model.trainable_parameters()
    n_trainable = sum(p.numel() for p in trainable_params)
    n_total = sum(p.numel() for p in model.parameters())
    logger.info(
        "Trainable params: %d / %d (%.2f%%)",
        n_trainable, n_total, 100 * n_trainable / (n_total + 1),
    )

    optimizer = optim.AdamW(
        trainable_params,
        lr=lora_cfg["learning_rate"],
        weight_decay=lora_cfg["weight_decay"],
    )

    total_steps = lora_cfg["max_steps"]
    warmup_steps = lora_cfg["warmup_steps"]
    scheduler = _cosine_with_warmup(optimizer, warmup_steps, total_steps)

    # ── Gradient accumulation ─────────────────────────────────────────────────
    accum_steps = lora_cfg.get("gradient_accumulation_steps", 8)
    max_grad_norm = lora_cfg.get("gradient_clipping", 1.0)

    # ── Checkpoint dir ────────────────────────────────────────────────────────
    ckpt_dir = Path(paths["checkpoints"]["lora"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    global_step = 0
    best_val_loss = float("inf")

    if resume and (ckpt_dir / "latest").exists():
        global_step, best_val_loss = _resume(model, optimizer, ckpt_dir, device)

    # ── Training loop ─────────────────────────────────────────────────────────
    logger.info("Starting LoRA fine-tuning (max_steps=%d)", total_steps)
    from src.utils.evaluation import evaluate_perplexity

    model.train()
    optimizer.zero_grad()

    train_iter = _infinite_loader(train_loader)
    log_every   = lora_cfg.get("logging_steps", 10)
    eval_every  = lora_cfg.get("eval_steps", 500)
    save_every  = lora_cfg.get("save_steps", 500)
    rouge_every = lora_cfg.get("rouge_eval_steps", eval_every * 5)

    running_loss = 0.0
    t0 = time.time()
    step_metrics: List[Dict] = []

    while global_step < total_steps:
        batch = next(train_iter)
        batch = _to_device(batch, device)

        out = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
            entity_spans=batch.get("entity_spans"),
        )
        loss = out.loss / accum_steps
        loss.backward()
        running_loss += out.loss.item()

        if (global_step + 1) % accum_steps == 0:
            torch.nn.utils.clip_grad_norm_(trainable_params, max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        global_step += 1

        if global_step % log_every == 0:
            elapsed = time.time() - t0
            avg_loss = running_loss / log_every
            lr = scheduler.get_last_lr()[0]
            logger.info(
                "Step %d/%d | loss=%.4f | lr=%.2e | %.1fs",
                global_step, total_steps, avg_loss, lr, elapsed,
            )
            if writer:
                writer.add_scalar("train/loss", avg_loss, global_step)
                writer.add_scalar("train/lr", lr, global_step)
            running_loss = 0.0
            t0 = time.time()

        if global_step % eval_every == 0:
            val_loss = _evaluate(model, val_loader, device)
            val_ppl  = evaluate_perplexity(model, val_loader, device)
            logger.info("  Val loss: %.4f | Val perplexity: %.2f", val_loss, val_ppl)
            if writer:
                writer.add_scalar("val/loss", val_loss, global_step)
                writer.add_scalar("val/perplexity", val_ppl, global_step)
            step_metrics.append({
                "step": global_step,
                "val_loss": val_loss, "val_perplexity": val_ppl,
            })
            model.train()

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                model.save_adapters(str(ckpt_dir / "best"))
                logger.info("  ✓ New best val loss=%.4f — saved", best_val_loss)

        if global_step % rouge_every == 0:
            rouge = _quick_rouge_eval(model, tokenizer, qa_dir, eval_cfg,
                                      entity_linker, device, max_samples=200)
            if rouge and writer:
                for k, v in rouge.items():
                    writer.add_scalar(f"val/{k}", v, global_step)
            if rouge:
                logger.info("  Periodic ROUGE @ step %d: %s", global_step,
                            {k: f"{v:.4f}" for k, v in rouge.items()})
            model.train()

        if global_step % save_every == 0:
            model.save_adapters(str(ckpt_dir / f"step_{global_step:06d}"))
            _save_training_state(optimizer, scheduler, global_step, best_val_loss,
                                 ckpt_dir / "latest")

    # ── Final save ────────────────────────────────────────────────────────────
    model.save_adapters(str(ckpt_dir / "final"))
    logger.info("Stage 3 complete. Best val loss: %.4f", best_val_loss)

    if writer:
        writer.close()

    # ── Final evaluation with ROUGE on full test set ──────────────────────────
    final_metrics = _run_generation_eval(
        model, tokenizer, qa_dir, eval_cfg, entity_linker, device
    )

    # ── Save metrics JSON ─────────────────────────────────────────────────────
    all_metrics = {"step_metrics": step_metrics, "final": final_metrics or {}}
    metrics_path = log_dir / "stage3_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(all_metrics, f, indent=2)
    logger.info("Metrics saved to %s", metrics_path)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _cosine_with_warmup(optimizer, warmup_steps: int, total_steps: int):
    import math

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return max(0.0, 0.5 * (1 + math.cos(math.pi * progress)))

    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def _to_device(batch: Dict, device: torch.device) -> Dict:
    return {k: (v.to(device) if isinstance(v, torch.Tensor) else v)
            for k, v in batch.items()}


def _infinite_loader(loader):
    while True:
        yield from loader


@torch.no_grad()
def _evaluate(model, val_loader, device: torch.device) -> float:
    model.eval()
    total_loss, n = 0.0, 0
    for batch in val_loader:
        batch = _to_device(batch, device)
        out = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
            entity_spans=batch.get("entity_spans"),
        )
        total_loss += out.loss.item()
        n += 1
    return total_loss / max(n, 1)


def _save_training_state(optimizer, scheduler, step: int, best_loss: float, path: Path) -> None:
    torch.save({
        "step": step,
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "best_val_loss": best_loss,
    }, str(path))


def _resume(model, optimizer, ckpt_dir: Path, device: torch.device) -> tuple[int, float]:
    from peft import PeftModel
    latest_path = ckpt_dir / "latest"
    state = torch.load(str(latest_path), map_location=device)
    optimizer.load_state_dict(state["optimizer"])
    step = state.get("step", 0)
    best_loss = state.get("best_val_loss", float("inf"))
    logger.info("Resumed from step %d (best val loss=%.4f)", step, best_loss)
    return step, best_loss


def _generate_predictions(model, tokenizer, samples, eval_cfg, entity_linker, device):
    """Run generation for a list of QA samples, return (predictions, references, ref_entities)."""
    predictions, references, ref_entities = [], [], []
    gen_cfg = eval_cfg.get("generation", {})
    model.eval()

    for sample in samples:
        q = sample["question"]
        prompt = f"### Biological Reasoning Question\n{q}\n\n### Answer\n"
        enc = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
        input_ids = enc["input_ids"].to(device)
        attn = enc["attention_mask"].to(device)

        entity_spans = None
        if entity_linker:
            raw_spans = entity_linker.find_token_spans(prompt, tokenizer)
            entity_spans = [raw_spans]

        out_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attn,
            entity_spans=entity_spans,
            max_new_tokens=gen_cfg.get("max_new_tokens", 256),
            temperature=gen_cfg.get("temperature", 0.7),
            top_p=gen_cfg.get("top_p", 0.9),
            do_sample=gen_cfg.get("do_sample", True),
            repetition_penalty=gen_cfg.get("repetition_penalty", 1.3),
            no_repeat_ngram_size=gen_cfg.get("no_repeat_ngram_size", 3),
            eos_token_id=tokenizer.eos_token_id,
        )
        gen_text = tokenizer.decode(out_ids[0][input_ids.shape[1]:], skip_special_tokens=True)
        predictions.append(gen_text)
        references.append(sample["answer"])
        ref_entities.append(sample.get("entities", []))

    return predictions, references, ref_entities


@torch.no_grad()
def _quick_rouge_eval(model, tokenizer, qa_dir: Path, eval_cfg: Dict,
                       entity_linker, device: torch.device,
                       max_samples: int = 200) -> Optional[Dict]:
    """Lightweight mid-training ROUGE check on a subset of the validation set."""
    from src.utils.evaluation import evaluate_rouge
    val_path = qa_dir / "val.jsonl"
    if not val_path.exists():
        return None
    samples = []
    with open(val_path, encoding="utf-8") as f:
        for line in f:
            samples.append(json.loads(line.strip()))
    samples = samples[:max_samples]
    preds, refs, _ = _generate_predictions(
        model, tokenizer, samples, eval_cfg, entity_linker, device
    )
    return evaluate_rouge(preds, refs, eval_cfg.get("rouge_types", ["rougeL"]))


@torch.no_grad()
def _run_generation_eval(model, tokenizer, qa_dir: Path, eval_cfg: Dict,
                          entity_linker, device: torch.device) -> Optional[Dict]:
    """Generate answers for the full test set and compute ROUGE + entity metrics."""
    from src.utils.evaluation import evaluate_rouge, evaluate_entity_mention_accuracy, print_metrics

    test_path = qa_dir / "test.jsonl"
    if not test_path.exists():
        return None

    samples = []
    with open(test_path, encoding="utf-8") as f:
        for line in f:
            samples.append(json.loads(line.strip()))
    logger.info("Running final generation eval on %d test samples…", len(samples))

    predictions, references, ref_entities = _generate_predictions(
        model, tokenizer, samples, eval_cfg, entity_linker, device
    )

    rouge_metrics = evaluate_rouge(predictions, references, eval_cfg.get("rouge_types", ["rougeL"]))
    print_metrics(rouge_metrics, title="Test ROUGE Scores")
    result = dict(rouge_metrics)

    if entity_linker:
        entity_metrics = evaluate_entity_mention_accuracy(predictions, ref_entities, entity_linker)
        print_metrics(entity_metrics, title="Entity Mention Accuracy")
        result.update(entity_metrics)

    return result
