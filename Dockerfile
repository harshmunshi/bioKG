# ─────────────────────────────────────────────────────────────────────────────
# BioKG-LoRA Dockerfile
#
# Stages:
#   base      – CUDA + Python + system deps
#   deps      – pip dependencies (cached layer, includes dvc[s3])
#   app       – application code + DVC tracking files
#
# On container start the entrypoint runs `dvc pull` automatically before
# delegating to train.py.  Set SKIP_DVC_PULL=1 to skip when data is already
# on a mounted volume.
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
#   docker run --gpus all -e SKIP_DVC_PULL=1 biokg-lora --stage 2   # skip dvc pull
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
        unzip \
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
COPY config/                config/
COPY src/                   src/
COPY train.py               train.py

# Copy DVC metadata so `dvc pull` inside the container knows what to fetch
COPY .dvc/config            .dvc/config
COPY .dvcignore             .dvcignore
COPY data_old/mgi.dvc       data_old/mgi.dvc
COPY data_old/ontologies.dvc data_old/ontologies.dvc
COPY data_old/string.dvc    data_old/string.dvc

# Copy and register the entrypoint script
COPY docker-entrypoint.sh   /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

# Create data and checkpoint directories (data_old sub-dirs are populated by dvc pull)
RUN mkdir -p \
    data_old/mgi \
    data_old/ontologies \
    data_old/string \
    data_old/kegg \
    data_old/gtex \
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

# Initialise a minimal git repo so DVC doesn't print autostage warnings
RUN git init -q && git config user.email "docker@biokg" && git config user.name "docker"

# ── HuggingFace cache (mount a volume here for persistence) ──────────────────
ENV HF_HOME=/workspace/hf_cache
ENV TRANSFORMERS_CACHE=/workspace/hf_cache/transformers
RUN mkdir -p /workspace/hf_cache

# ── Healthcheck ───────────────────────────────────────────────────────────────
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import torch; assert torch.cuda.is_available(), 'No CUDA'" \
    || exit 1

# ── Entrypoint ────────────────────────────────────────────────────────────────
# docker-entrypoint.sh runs `dvc pull` then execs `python train.py "$@"`
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["--help"]
