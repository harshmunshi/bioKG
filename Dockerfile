# ─────────────────────────────────────────────────────────────────────────────
# BioKG-LoRA Dockerfile
#
# Stages:
#   base      – CUDA + Python + system deps
#   deps      – pip dependencies (cached layer)
#   app       – application code
#
# Build:
#   docker build -t biokg-lora .
#   docker build --target deps -t biokg-lora:deps .   # pre-install deps only
#
# Run examples:
#   docker run --gpus all biokg-lora --from-scratch
#   docker run --gpus all biokg-lora --stage 1
#   docker run --gpus all biokg-lora --stage 3 --resume
#   docker run --gpus all biokg-lora --predict "What phenotypes does Thbd knockout cause?"
# ─────────────────────────────────────────────────────────────────────────────

# ── Base: CUDA 12.1 + Python 3.11 ─────────────────────────────────────────────
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04 AS base

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONFAULTHANDLER=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.11 \
        python3.11-dev \
        python3-pip \
        python3.11-venv \
        build-essential \
        git \
        wget \
        curl \
        ca-certificates \
        libgomp1 \
    && ln -sf /usr/bin/python3.11 /usr/bin/python \
    && ln -sf /usr/bin/python3.11 /usr/bin/python3 \
    && rm -rf /var/lib/apt/lists/*

# ── Dependencies ──────────────────────────────────────────────────────────────
FROM base AS deps

WORKDIR /opt/biokg

# Copy only requirements first for better layer caching
COPY requirements.txt .

# Install PyTorch with CUDA 12.1 support first (specific index URL needed)
RUN pip install --upgrade pip && \
    pip install torch==2.2.2 torchvision==0.17.2 \
        --index-url https://download.pytorch.org/whl/cu121

# Install torch-geometric (separate index)
RUN pip install torch-geometric==2.5.3 \
        torch-scatter torch-sparse torch-cluster torch-spline-conv \
        -f https://data.pyg.org/whl/torch-2.2.2+cu121.html

# Install remaining requirements
RUN pip install -r requirements.txt

# ── Application ───────────────────────────────────────────────────────────────
FROM deps AS app

WORKDIR /workspace/biokg

# Copy source code
COPY config/   config/
COPY src/      src/
COPY train.py  train.py

# Create data and checkpoint directories
RUN mkdir -p \
    data/raw/mgi \
    data/raw/go \
    data/raw/kegg \
    data/raw/string \
    data/raw/gtex \
    data/kg \
    data/qa \
    checkpoints/rotate \
    checkpoints/projection \
    checkpoints/lora \
    logs

# ── HuggingFace cache (mount a volume here for persistence) ──────────────────
ENV HF_HOME=/workspace/hf_cache
ENV TRANSFORMERS_CACHE=/workspace/hf_cache/transformers
RUN mkdir -p /workspace/hf_cache

# ── Healthcheck ───────────────────────────────────────────────────────────────
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import torch; assert torch.cuda.is_available(), 'No CUDA'" \
    || exit 1

# ── Entrypoint ────────────────────────────────────────────────────────────────
ENTRYPOINT ["python", "train.py"]
CMD ["--help"]
