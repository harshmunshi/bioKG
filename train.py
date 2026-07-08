#!/usr/bin/env python3
"""
BioKG-LoRA: Main training entry point.

Usage:
    # Run all stages sequentially from scratch
    python train.py --from-scratch

    # Run a specific stage only
    python train.py --stage 0    # KG construction + QA generation
    python train.py --stage 1    # RotatE embedding training
    python train.py --stage 2    # Projection layer training
    python train.py --stage 3    # LoRA fine-tuning

    # Resume from checkpoint (stages 1-3)
    python train.py --stage 1 --resume

    # Override any config value from the command line
    python train.py --stage 3 rotate.max_epochs=100 lora.learning_rate=1e-4

    # Use a different config file
    python train.py --config config/config_debug.yaml --stage 1

    # Run stages 0 and 1 only (e.g. for pre-processing on a CPU node)
    python train.py --stages 0,1

    # Select specific GPU (useful on multi-GPU servers)
    CUDA_VISIBLE_DEVICES=2 python train.py --stage 1
"""
from __future__ import annotations

import argparse
import logging
import os
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import yaml

# ── Project root — all relative paths are resolved against this ───────────────
# Works regardless of which directory the user calls `python train.py` from.
PROJECT_ROOT = Path(__file__).resolve().parent


# ── Logging setup ─────────────────────────────────────────────────────────────

def setup_logging(log_dir: str, level: str = "INFO") -> None:
    log_dir_path = _abs(log_dir)
    log_dir_path.mkdir(parents=True, exist_ok=True)

    # Verify the log directory is actually writable before setting up the handler
    log_file = log_dir_path / "biokg_lora.log"
    try:
        log_file.touch(exist_ok=True)
    except OSError as e:
        # Fall back to stdout-only logging rather than crashing at startup
        logging.basicConfig(
            level=getattr(logging, level.upper(), logging.INFO),
            format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            handlers=[logging.StreamHandler(sys.stdout)],
        )
        logging.warning("Cannot write to log file %s: %s — logging to stdout only", log_file, e)
        return

    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(str(log_file), encoding="utf-8"),
        ],
    )

logger = logging.getLogger("biokg_lora")


# ── Path resolution ───────────────────────────────────────────────────────────

def _abs(p: str) -> Path:
    """Resolve a path relative to PROJECT_ROOT if it is not already absolute."""
    path = Path(p)
    return path if path.is_absolute() else PROJECT_ROOT / path


def resolve_cfg_paths(cfg: Dict) -> Dict:
    """
    Walk the paths section of the config and make every value absolute,
    anchored to PROJECT_ROOT. This means train.py can be called from any
    working directory on the server.
    """
    paths = cfg["paths"]

    paths["data_root"] = str(_abs(paths["data_root"]))
    paths["logs"] = str(_abs(paths["logs"]))

    raw = paths["raw_data"]
    for key in raw:
        raw[key] = str(_abs(raw[key]))

    kg = paths["kg"]
    for key in kg:
        kg[key] = str(_abs(kg[key]))

    qa = paths["qa"]
    for key in qa:
        qa[key] = str(_abs(qa[key]))

    ckpts = paths["checkpoints"]
    for key in ckpts:
        ckpts[key] = str(_abs(ckpts[key]))

    return cfg


# ── Config ────────────────────────────────────────────────────────────────────

def load_config(config_path: str) -> Dict:
    config_path = str(_abs(config_path))
    if not Path(config_path).exists():
        print(
            f"ERROR: Config file not found: {config_path}\n"
            f"  Run from the project root: cd {PROJECT_ROOT} && python train.py ...",
            file=sys.stderr,
        )
        sys.exit(1)
    with open(config_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return resolve_cfg_paths(cfg)


def apply_overrides(cfg: Dict, overrides: List[str]) -> Dict:
    """Apply dot-notation overrides from CLI, e.g. rotate.max_epochs=50."""
    for override in overrides:
        key, _, val = override.partition("=")
        key = key.lstrip("-")
        parts = key.split(".")
        d = cfg
        for part in parts[:-1]:
            d = d.setdefault(part, {})
        d[parts[-1]] = _coerce(val)
    return cfg


def _coerce(val: str):
    if val.lower() == "true":
        return True
    if val.lower() == "false":
        return False
    try:
        return int(val)
    except ValueError:
        pass
    try:
        return float(val)
    except ValueError:
        pass
    return val


# ── Environment setup ─────────────────────────────────────────────────────────

def setup_environment(cfg: Dict) -> None:
    """
    Apply environment variables and validate the runtime environment
    before any stage starts. Catches the most common server-specific
    failures early with clear error messages.
    """
    hw = cfg.get("hardware", {})

    # 1. HuggingFace cache — configurable so it doesn't fill the user's home dir
    hf_cache = cfg.get("paths", {}).get("hf_cache")
    if hf_cache:
        hf_cache = str(_abs(hf_cache))
        os.environ.setdefault("HF_HOME", hf_cache)
        os.environ.setdefault("TRANSFORMERS_CACHE", str(Path(hf_cache) / "transformers"))
    # If HF_HOME is already set in the shell environment, respect it
    if "HF_HOME" in os.environ:
        logger.info("HuggingFace cache: %s", os.environ["HF_HOME"])

    # 2. GPU selection — set CUDA_VISIBLE_DEVICES from config if not already set in shell
    gpu_id = hw.get("gpu_id")
    if gpu_id is not None and "CUDA_VISIBLE_DEVICES" not in os.environ:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        logger.info("Pinned to GPU %s (from config hardware.gpu_id)", gpu_id)
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        logger.info("CUDA_VISIBLE_DEVICES=%s", os.environ["CUDA_VISIBLE_DEVICES"])

    # 3. DataLoader multiprocessing — use spawn to avoid CUDA fork issues on Linux
    if hw.get("num_workers", 0) > 0:
        try:
            import torch.multiprocessing as mp
            if mp.get_start_method(allow_none=True) is None:
                mp.set_start_method("spawn")
        except RuntimeError:
            pass  # Already set

    # 4. GPU capability validation
    if torch.cuda.is_available():
        _validate_gpu(cfg)

    # 5. Write-access check for all output directories
    _check_write_access(cfg)


def _validate_gpu(cfg: Dict) -> None:
    """Warn if GPU doesn't support the requested quantization / dtype."""
    if torch.cuda.device_count() == 0:
        logger.warning("No GPU detected — training will be very slow")
        return

    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        cc = props.major * 10 + props.minor  # compute capability × 10
        name = props.name

        # bfloat16 and 4-bit quantization requires Ampere (sm_80) or newer
        uses_4bit = cfg.get("lora", {}).get("load_in_4bit", False)
        uses_bf16 = cfg.get("hardware", {}).get("bf16", False)

        if (uses_4bit or uses_bf16) and cc < 80:
            logger.warning(
                "GPU %d (%s, compute capability %d.%d) does not support bfloat16 / 4-bit "
                "quantization (requires Ampere sm_80+). "
                "Set hardware.bf16=false and lora.quantization=8bit in config.yaml.",
                i, name, props.major, props.minor,
            )
        logger.info("GPU %d: %s  (sm_%d%d, %d MiB)",
                    i, name, props.major, props.minor,
                    props.total_memory // (1024 ** 2))


def _check_write_access(cfg: Dict) -> None:
    """Test write access to all output directories before training starts."""
    paths = cfg["paths"]
    dirs_to_check = [
        paths["logs"],
        paths["data_root"],
    ] + list(paths["checkpoints"].values())

    failed = []
    for d in dirs_to_check:
        p = Path(d)
        try:
            p.mkdir(parents=True, exist_ok=True)
            test_file = p / ".write_test"
            test_file.touch()
            test_file.unlink()
        except OSError as e:
            failed.append(f"  {d}: {e}")

    if failed:
        logger.error(
            "Write permission check failed for the following directories:\n%s\n"
            "Fix permissions or change paths in config.yaml before running.",
            "\n".join(failed),
        )
        sys.exit(1)


# ── Reproducibility ───────────────────────────────────────────────────────────

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ── Stage runners ─────────────────────────────────────────────────────────────

STAGE_NAMES = {
    0: "Knowledge Graph Construction + QA Generation",
    1: "RotatE Embedding Training",
    2: "KG → LM Projection Layer Training",
    3: "BioKG-LoRA Fine-tuning",
}


def run_stage(stage: int, cfg: Dict, resume: bool = False) -> None:
    logger.info("=" * 60)
    logger.info("STAGE %d: %s", stage, STAGE_NAMES[stage])
    logger.info("=" * 60)

    if stage == 0:
        from src.training.stage0_kg_construction import run
        run(cfg)
    elif stage == 1:
        from src.training.stage1_rotate import run
        _check_inputs(stage, cfg)
        run(cfg, resume=resume)
    elif stage == 2:
        from src.training.stage2_projection import run
        _check_inputs(stage, cfg)
        run(cfg, resume=resume)
    elif stage == 3:
        from src.training.stage3_lora import run
        _check_inputs(stage, cfg)
        run(cfg, resume=resume)
    else:
        raise ValueError(f"Unknown stage: {stage}. Valid stages: 0, 1, 2, 3")

    logger.info("Stage %d complete.", stage)


def _check_inputs(stage: int, cfg: Dict) -> None:
    """Verify that required outputs from previous stages exist."""
    kg_dir = Path(cfg["paths"]["data_root"]) / "kg"
    qa_dir = Path(cfg["paths"]["data_root"]) / "qa"

    required: Dict[int, List[tuple]] = {
        1: [(kg_dir / "triples_train.pt",        "Run Stage 0 first (--stage 0)")],
        2: [(kg_dir / "entity_embeddings.pt",     "Run Stage 1 first (--stage 1)")],
        3: [(qa_dir / "train.jsonl",              "Run Stage 0 first (--stage 0)")],
    }

    missing = [(p, h) for p, h in required.get(stage, []) if not p.exists()]
    if missing:
        for path, hint in missing:
            logger.error("Required file missing: %s  →  %s", path, hint)
        sys.exit(1)


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="BioKG-LoRA training pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    group = parser.add_mutually_exclusive_group()
    group.add_argument("--from-scratch", action="store_true",
                       help="Run all 4 stages (0 → 1 → 2 → 3)")
    group.add_argument("--stage", type=int, choices=[0, 1, 2, 3],
                       help="Run a single stage")
    group.add_argument("--stages", type=str,
                       help="Comma-separated stages, e.g. '0,1'")

    parser.add_argument("--resume", action="store_true",
                        help="Resume from latest checkpoint (stages 1-3)")
    parser.add_argument("--config", type=str, default="config/config.yaml",
                        help="Path to YAML config (default: config/config.yaml)")
    parser.add_argument("--log-level", type=str, default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    parser.add_argument("--eval-only", action="store_true",
                        help="Evaluate test set (requires Stage 3 checkpoint)")
    parser.add_argument("--predict", type=str, default=None,
                        help="Answer a single question and exit")
    parser.add_argument("overrides", nargs="*",
                        help="Config overrides: key=value, e.g. rotate.max_epochs=200")

    return parser.parse_args()


# ── Eval / inference ──────────────────────────────────────────────────────────

def run_eval(cfg: Dict) -> None:
    from src.training.stage3_lora import _run_generation_eval
    from src.models.biokg_lora import BioKGLoRA
    from src.utils.model_profile import resolve_model_profile
    from transformers import AutoTokenizer
    from src.utils.entity_linker import EntityLinker

    paths = cfg["paths"]
    lora_cfg = cfg["lora"]
    model_profile = resolve_model_profile(cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_dir = Path(paths["checkpoints"]["lora"])
    kg_dir = Path(paths["data_root"]) / "kg"
    entity2id_path = str(kg_dir / "entity2id.json")

    tokenizer = AutoTokenizer.from_pretrained(lora_cfg["base_model"], trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    entity_linker = EntityLinker.from_file(entity2id_path) if Path(entity2id_path).exists() else None
    model = BioKGLoRA.load_for_inference(
        adapter_dir=str(ckpt_dir / "best"),
        base_model_name=lora_cfg["base_model"],
        kg_embedding_path=str(kg_dir / "entity_embeddings.pt"),
        entity2id_path=entity2id_path,
        kg_dim=cfg["projection"]["kg_dim"],
        lm_dim=model_profile["lm_dim"],
        kg_weight=lora_cfg.get("kg_weight", 0.3),
        lora_rank=lora_cfg["lora_rank"],
        lora_alpha=lora_cfg["lora_alpha"],
        lora_dropout=lora_cfg["lora_dropout"],
        quantization=lora_cfg.get("quantization"),
        target_modules=model_profile["target_modules"],
    )
    model.eval()
    _run_generation_eval(model, tokenizer,
                         qa_dir=Path(paths["data_root"]) / "qa",
                         eval_cfg=cfg["evaluation"],
                         entity_linker=entity_linker,
                         device=device)


def run_predict(question: str, cfg: Dict) -> None:
    from src.models.biokg_lora import BioKGLoRA
    from src.utils.model_profile import resolve_model_profile
    from transformers import AutoTokenizer
    from src.utils.entity_linker import EntityLinker

    paths = cfg["paths"]
    lora_cfg = cfg["lora"]
    model_profile = resolve_model_profile(cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    kg_dir = Path(paths["data_root"]) / "kg"
    ckpt_dir = Path(paths["checkpoints"]["lora"])
    entity2id_path = str(kg_dir / "entity2id.json")

    tokenizer = AutoTokenizer.from_pretrained(lora_cfg["base_model"], trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    entity_linker = EntityLinker.from_file(entity2id_path) if Path(entity2id_path).exists() else None
    model = BioKGLoRA.load_for_inference(
        adapter_dir=str(ckpt_dir / "best"),
        base_model_name=lora_cfg["base_model"],
        kg_embedding_path=str(kg_dir / "entity_embeddings.pt"),
        entity2id_path=entity2id_path,
        kg_dim=cfg["projection"]["kg_dim"],
        lm_dim=model_profile["lm_dim"],
        kg_weight=lora_cfg.get("kg_weight", 0.3),
        lora_rank=lora_cfg["lora_rank"],
        lora_alpha=lora_cfg["lora_alpha"],
        lora_dropout=lora_cfg["lora_dropout"],
        quantization=lora_cfg.get("quantization"),
        target_modules=model_profile["target_modules"],
    )
    model.eval()

    prompt = f"### Biological Reasoning Question\n{question}\n\n### Answer\n"
    enc = tokenizer(prompt, return_tensors="pt").to(device)
    entity_spans = [entity_linker.find_token_spans(prompt, tokenizer)] if entity_linker else None
    gen_cfg = cfg["evaluation"].get("generation", {})
    out_ids = model.generate(
        input_ids=enc["input_ids"],
        attention_mask=enc["attention_mask"],
        entity_spans=entity_spans,
        max_new_tokens=gen_cfg.get("max_new_tokens", 512),
        temperature=gen_cfg.get("temperature", 0.7),
        top_p=gen_cfg.get("top_p", 0.9),
        do_sample=gen_cfg.get("do_sample", True),
        repetition_penalty=gen_cfg.get("repetition_penalty", 1.3),
        no_repeat_ngram_size=gen_cfg.get("no_repeat_ngram_size", 3),
        eos_token_id=tokenizer.eos_token_id,
    )
    answer = tokenizer.decode(out_ids[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)
    print(f"\n{'='*60}\nQuestion: {question}\n{'-'*60}\nAnswer:\n{answer}\n{'='*60}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    # Must add project root to sys.path so `src.*` imports work regardless of CWD
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

    args = parse_args()
    cfg = load_config(args.config)
    if args.overrides:
        cfg = apply_overrides(cfg, args.overrides)

    setup_logging(cfg["paths"]["logs"], level=args.log_level)
    logger.info("BioKG-LoRA | project root: %s", PROJECT_ROOT)
    logger.info("Config: %s", _abs(args.config))

    seed = cfg["hardware"].get("seed", 42)
    set_seed(seed)
    logger.info("Random seed: %d", seed)

    setup_environment(cfg)

    if args.predict:
        run_predict(args.predict, cfg)
        return
    if args.eval_only:
        run_eval(cfg)
        return

    if args.from_scratch:
        stages = [0, 1, 2, 3]
    elif args.stage is not None:
        stages = [args.stage]
    elif args.stages:
        stages = [int(s.strip()) for s in args.stages.split(",")]
    else:
        logger.error("Specify --from-scratch, --stage N, --stages N,M, --eval-only, or --predict TEXT")
        sys.exit(1)

    for stage in stages:
        run_stage(stage, cfg, resume=args.resume)

    logger.info("All requested stages complete.")


if __name__ == "__main__":
    main()
