#!/usr/bin/env bash
# =============================================================================
# BioKG-LoRA: New collaborator setup
#
# Run once after cloning:
#   bash scripts/setup.sh
#
# What it does:
#   1. Installs Python dependencies (Stage 0 only, CPU-safe)
#   2. Pulls all tracked data & artifacts via DVC
# =============================================================================
set -euo pipefail

PYTHON="${PYTHON:-python3}"

echo "=== BioKG-LoRA setup ==="
echo ""

# 1. Dependencies
echo "[1/2] Installing Stage 0 dependencies..."
$PYTHON -m pip install -q -r requirements-stage0.txt
echo "  Done."

# 2. DVC pull (uses public S3 HTTPS — no credentials needed)
echo "[2/2] Pulling data & artifacts via DVC..."
if ! command -v dvc &>/dev/null; then
    $PYTHON -m pip install -q "dvc[s3]"
fi

dvc pull --remote s3remote-public
echo "  Done."

echo ""
echo "Setup complete. Run Stage 0:"
echo "  python train.py --stage 0"
