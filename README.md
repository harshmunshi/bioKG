# BioKG-LoRA

**Knowledge Graph Enhanced LLMs for Clinical Reasoning**

Injects RotatE embeddings from a biological knowledge graph into a small language model (Gemma-4-E2B-IT or Llama-3-8B) via LoRA, enabling biologically-grounded reasoning about gene–phenotype–clinical relationships.

```
Question: "What is the significance of elevated ALT in Thbd knockout?"

Base LLM:     "ALT elevation typically indicates liver damage..."  ❌ generic

BioKG-LoRA:   "ALT elevation in Thbd knockout is significant because
               THBD regulates coagulation in hepatic sinusoids.
               Loss of THBD causes microthrombi → ischemic liver
               injury → hepatocellular damage → ALT release."     ✓ grounded
```

---

## Pipeline Overview

```
Stage 0  Knowledge Graph + QA Generation   CPU · 1–2 hrs · no GPU
   ↓
Stage 1  RotatE Embedding Training          GPU · 2–3 days
   ↓
Stage 2  KG → LM Projection Layer          GPU · ~2 hrs
   ↓
Stage 3  BioKG-LoRA Fine-tuning            GPU · 4–6 hrs
```

---

## Requirements

- Docker + Docker Compose
- NVIDIA GPU with ≥24 GB VRAM (Stages 1–3)
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
- AWS credentials with write access to `s3://biokg-data-lora` (push only)

---

## Quick Start

### 1. Clone

```bash
git clone <repo-url>
cd bioKG
```

### 2. Configure environment

```bash
cp .env.example .env
# Edit .env — at minimum set HUGGING_FACE_HUB_TOKEN (required for Gemma / Llama)
```

### 3. Build the Docker image

```bash
docker compose build
```

### 4. Run

`dvc pull` runs automatically inside every container before training starts — no manual data download needed.

```bash
# Full pipeline from scratch
docker compose up pipeline

# Or stage by stage (recommended)
docker compose up kg          # Stage 0 — no GPU needed
docker compose up rotate      # Stage 1
docker compose up projection  # Stage 2
docker compose up lora        # Stage 3
```

On subsequent runs the pulled data lives in `./data_old` (a mounted volume), so DVC skips files already on disk.

---

## Detailed Usage

### Running stages individually

```bash
# Stage 0: Build knowledge graph + generate QA pairs (CPU only, ~32 GB RAM)
docker compose up kg

# Stage 1: Train RotatE embeddings (GPU, 2–3 days)
docker compose up rotate

# Resume Stage 1 from checkpoint
docker compose up rotate-resume

# Stage 2: Train KG → LM projection layer (GPU, ~2 hours)
docker compose up projection

# Stage 3: LoRA fine-tuning (GPU, 4–6 hours)
docker compose up lora

# Resume Stage 3 from checkpoint
docker compose up lora-resume
```

### Running without Docker

```bash
# Install Stage 0 deps only (CPU-safe, works on MacBook)
pip install -r requirements-stage0.txt

# Full deps (requires CUDA)
pip install -r requirements.txt

# Run any stage
python train.py --stage 0
python train.py --stage 1
python train.py --stage 1 --resume
python train.py --from-scratch

# Override config values inline
python train.py --stage 1 rotate.max_epochs=100 rotate.batch_size=512

# Evaluate
python train.py --eval-only

# Interactive prediction
python train.py --predict "What phenotypes does Thbd knockout cause?"
```

### Evaluation & monitoring

```bash
# TensorBoard (reads from logs/)
python -m tensorboard.main --logdir logs --port 6006
# → http://localhost:6006

# Evaluate best Stage 3 checkpoint on test set
docker compose up eval

# Interactive prediction via Docker
docker compose run predict --predict "Why is creatinine elevated in Fgfr2 knockout?"
```

---

## Data & Artifacts

All data and model artifacts are tracked with [DVC](https://dvc.org) and stored on S3.

| Path | Contents | DVC tracked |
|------|----------|-------------|
| `data_old/mgi/` | MGI gene + phenotype files | ✓ |
| `data_old/ontologies/` | GO OBO, MGI GAF annotations | ✓ |
| `data_old/string/` | STRING PPI network (mouse) | ✓ |
| `data/kg/` | Built KG, entity2id, triples, embeddings | ✓ |
| `data/qa/` | Generated QA pairs (train/val/test JSONL) | ✓ |
| `checkpoints/` | RotatE, projection, LoRA weights | ✓ |

### For collaborators (no AWS keys needed)

`dvc pull` runs automatically when you start any Docker Compose service.  To pull manually (e.g. outside Docker):

```bash
pip install "dvc[s3]"
dvc pull --remote s3remote-public
```

To skip the automatic pull on a subsequent container run (data already on disk):

```bash
SKIP_DVC_PULL=1 docker compose up lora
```

### For maintainers (pushing new artifacts)

Add your AWS credentials to `.env`:

```bash
AWS_ACCESS_KEY_ID=...
AWS_SECRET_ACCESS_KEY=...
AWS_DEFAULT_REGION=us-east-1
```

Then push from inside the container or locally:

```bash
dvc push
git add .
git commit -m "update dvc artifacts"
git push
```

---

## Project Structure

```
bioKG/
├── train.py                    # Main entry point — all stage control
├── config/config.yaml          # All hyperparameters + model_profiles
├── Dockerfile
├── docker-compose.yml
├── docker-entrypoint.sh        # Runs dvc pull then delegates to train.py
├── .env.example                # Copy to .env and fill in tokens/keys
├── requirements.txt            # Full deps (GPU server, includes dvc[s3])
├── requirements-stage0.txt     # Lightweight deps (Stage 0, MacBook-safe)
├── scripts/
│   ├── setup.sh                # New collaborator setup
│   └── download_data.sh        # Manual raw data download (if not using DVC)
└── src/
    ├── models/
    │   ├── rotate.py           # RotatE KG embedding model
    │   ├── projection.py       # KG → LM projection + InfoNCE loss
    │   └── biokg_lora.py       # Full BioKG-LoRA model
    ├── data/
    │   ├── kg_builder.py       # ETL: MGI / GO / STRING → unified KG
    │   ├── kg_dataset.py       # Triple dataset with negative sampling
    │   ├── qa_generator.py     # Auto QA generation from KG paths
    │   └── qa_dataset.py       # Tokenised QA dataset for LoRA
    ├── training/
    │   ├── stage0_kg_construction.py
    │   ├── stage1_rotate.py
    │   ├── stage2_projection.py
    │   └── stage3_lora.py
    └── utils/
        ├── entity_linker.py    # Trie-based biological entity linker
        ├── evaluation.py       # MRR, Hits@K, ROUGE, entity F1
        └── model_profile.py    # Resolves lm_dim / target_modules per model
```

---

## Configuration

All hyperparameters live in `config/config.yaml`. Override any value from the CLI:

```bash
python train.py --stage 1 rotate.max_epochs=200 rotate.batch_size=2048
python train.py --stage 3 lora.learning_rate=1e-4 lora.lora_rank=64
```

Key defaults:

| Stage | Key params |
|-------|-----------|
| RotatE | `embedding_dim=256`, `margin=9.0`, `batch_size=1024`, `max_epochs=500` |
| Projection | `kg_dim=256`, `lm_dim` auto-resolved from model profile, `temperature=0.07`, `max_epochs=10` |
| LoRA | `base_model=gemma-4-e2b-it`, `lora_rank=32`, `max_steps=5000`, `quantization=4bit` |

### Switching models

The active model is set in `config/config.yaml` under `lora.base_model`. Both models are pre-configured in `model_profiles` — switching is a one-line change:

```yaml
lora:
  base_model: "google/gemma-4-e2b-it"    # default
  # base_model: "meta-llama/Llama-3-8B"  # alternative
```

`lm_dim` and `target_modules` are resolved automatically from `model_profiles`. To add a new model, add an entry there and set `lora.base_model` to its HuggingFace ID.

You can also override the model at runtime without editing the config:

```bash
python train.py --stage 3 lora.base_model="meta-llama/Llama-3-8B"
```

---

## Expected Results

Results using Llama-3-8B as the base model (Gemma-4-E2B-IT numbers TBD):

| Model | Perplexity | ROUGE-L | Entity F1 |
|-------|------------|---------|-----------|
| Llama-3-8B (base) | 24.3 | 0.45 | 0.38 |
| **BioKG-LoRA (Llama-3-8B)** | **17.8** | **0.63** | **0.89** |

---

## HuggingFace Token

Both supported models are gated and require a HuggingFace token. Set it in `.env` before running Stages 2–3:

```bash
# .env
HUGGING_FACE_HUB_TOKEN=hf_...
```

Or inline:

```bash
HUGGING_FACE_HUB_TOKEN=hf_... docker compose up lora
```

Request access:
- Gemma-4-E2B-IT: https://huggingface.co/google/gemma-4-e2b-it
- Llama-3-8B: https://huggingface.co/meta-llama/Meta-Llama-3-8B
