# BioKG-LoRA

**Knowledge Graph Enhanced LLMs for Clinical Reasoning**

Injects RotatE embeddings from a biological knowledge graph into a small language model (Llama-3-8B) via LoRA, enabling biologically-grounded reasoning about gene–phenotype–clinical relationships.

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

### 2. Pull data & artifacts

No AWS credentials needed — the bucket is public-read.

```bash
# Install DVC
pip install "dvc[s3]"

# Pull all tracked data (raw files, KG, QA pairs, checkpoints)
dvc pull --remote s3remote-public
```

Or use the setup script (also installs Stage 0 Python deps):

```bash
bash scripts/setup.sh
```

### 3. Build the Docker image

```bash
docker compose build
```

### 4. Run

```bash
# Full pipeline from scratch
docker compose up pipeline

# Or stage by stage (recommended)
docker compose up kg          # Stage 0 — no GPU needed
docker compose up rotate      # Stage 1
docker compose up projection  # Stage 2
docker compose up lora        # Stage 3
```

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

```bash
dvc pull --remote s3remote-public
```

### For maintainers (pushing new artifacts)

```bash
# Configure AWS credentials
export AWS_ACCESS_KEY_ID=...
export AWS_SECRET_ACCESS_KEY=...
export AWS_DEFAULT_REGION=us-east-1

# After generating new artifacts, push
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
├── config/config.yaml          # All hyperparameters
├── Dockerfile
├── docker-compose.yml
├── requirements.txt            # Full deps (GPU server)
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
        └── evaluation.py       # MRR, Hits@K, ROUGE, entity F1
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
| Projection | `kg_dim=256`, `lm_dim=4096`, `temperature=0.07`, `max_epochs=10` |
| LoRA | `base_model=Llama-3-8B`, `lora_rank=32`, `max_steps=5000`, `quantization=4bit` |

---

## Expected Results

| Model | Perplexity | ROUGE-L | Entity F1 |
|-------|------------|---------|-----------|
| Llama-3-8B (base) | 24.3 | 0.45 | 0.38 |
| **BioKG-LoRA** | **17.8** | **0.63** | **0.89** |

---

## HuggingFace Token

Llama-3-8B is a gated model. Set your token before running Stages 2–3:

```bash
# Locally
export HUGGING_FACE_HUB_TOKEN=hf_...

# Docker
HUGGING_FACE_HUB_TOKEN=hf_... docker compose up lora
```

Request access at: https://huggingface.co/meta-llama/Meta-Llama-3-8B
