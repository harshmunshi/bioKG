#!/usr/bin/env bash
# =============================================================================
# BioKG-LoRA: Data & Artifact Setup Script
#
# New collaborator? Run this first:
#   bash scripts/download_data.sh
#
# What it does:
#   1. Downloads raw biological databases (MGI, GO, STRING)
#   2. Optionally downloads pre-built artifacts (KG, QA pairs, embeddings)
#      if you don't want to run Stage 0/1/2 yourself
#
# Options:
#   --raw-only       Download raw data only (then run Stage 0 yourself)
#   --artifacts-only Download pre-built KG + QA + embeddings only
#   --all            Download everything (default)
#   --source         Where to pull artifacts from: "gdrive" | "s3" | "hf"
#                    Default: gdrive
#
# Pre-built artifact sources (set whichever your team uses):
#   Google Drive:    set GDRIVE_FOLDER_ID below
#   AWS S3:          set S3_BUCKET below
#   HuggingFace Hub: set HF_REPO below
# =============================================================================

set -euo pipefail

# ── Configuration — edit these for your team ─────────────────────────────────
GDRIVE_FOLDER_ID=""          # e.g. "1BxiMVs0XRA5nFMdKvBdBZjgmUUqptlbs"
S3_BUCKET=""                 # e.g. "s3://my-lab-bucket/biokg-lora"
HF_REPO=""                   # e.g. "myorg/biokg-lora-artifacts"
# ─────────────────────────────────────────────────────────────────────────────

RAW_DIR="data_old"
KG_DIR="data/kg"
QA_DIR="data/qa"
CKPT_DIR="checkpoints"

DOWNLOAD_RAW=true
DOWNLOAD_ARTIFACTS=true
ARTIFACT_SOURCE="gdrive"

# ── Parse args ────────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --raw-only)       DOWNLOAD_ARTIFACTS=false; shift;;
        --artifacts-only) DOWNLOAD_RAW=false; shift;;
        --all)            shift;;
        --source)         ARTIFACT_SOURCE="$2"; shift 2;;
        *) echo "Unknown option: $1"; exit 1;;
    esac
done

mkdir -p "$RAW_DIR"/{mgi,ontologies,string,kegg,gtex} "$KG_DIR" "$QA_DIR" "$CKPT_DIR"

# =============================================================================
# PART 1: Raw biological databases
# =============================================================================

download_raw() {
    echo ""
    echo "=== Downloading raw biological databases ==="

    # ── MGI ───────────────────────────────────────────────────────────────────
    echo "[1/4] MGI (Mouse Genome Informatics)..."
    cd "$RAW_DIR/mgi"
    wget -nc -q --show-progress \
        "http://www.informatics.jax.org/downloads/reports/MRK_List2.rpt" || true
    wget -nc -q --show-progress \
        "http://www.informatics.jax.org/downloads/reports/MGI_PhenoGenoMP.rpt" || true
    cd -

    # ── GO ────────────────────────────────────────────────────────────────────
    echo "[2/4] Gene Ontology (OBO + MGI annotations)..."
    cd "$RAW_DIR/ontologies"
    wget -nc -q --show-progress \
        "http://purl.obolibrary.org/obo/go.obo" || true
    wget -nc -q --show-progress \
        "http://current.geneontology.org/annotations/mgi.gaf.gz" -O mgi_go.gaf.gz || true
    [ -f mgi_go.gaf.gz ] && gunzip -kf mgi_go.gaf.gz && echo "  Decompressed mgi_go.gaf" || true
    cd -

    # ── STRING ────────────────────────────────────────────────────────────────
    echo "[3/4] STRING protein interactions (mouse, taxid=10090, ~650 MB)..."
    cd "$RAW_DIR/string"
    BASE="https://stringdb-downloads.org/download"
    wget -nc -q --show-progress \
        "${BASE}/protein.links.v12.0/10090.protein.links.v12.0.txt.gz" || true
    wget -nc -q --show-progress \
        "${BASE}/protein.info.v12.0/10090.protein.info.v12.0.txt.gz" || true
    for f in *.gz; do [ -f "$f" ] && gunzip -kf "$f" && echo "  Decompressed $f" || true; done
    cd -

    # ── KEGG (optional) ───────────────────────────────────────────────────────
    echo "[4/4] KEGG — skipped (requires manual download or API key)"
    echo "      Place file at: $RAW_DIR/kegg/mmu_pathway_genes.txt"
    echo "      Format: tab-separated, columns: pathway_id  gene_symbol"

    echo ""
    echo "Raw data download complete."
    echo "Next: python train.py --stage 0"
}

# =============================================================================
# PART 2: Pre-built artifacts (skip Stage 0/1/2)
# =============================================================================

download_artifacts() {
    echo ""
    echo "=== Downloading pre-built artifacts (source: $ARTIFACT_SOURCE) ==="

    case "$ARTIFACT_SOURCE" in
        gdrive)  _download_gdrive;;
        s3)      _download_s3;;
        hf)      _download_hf;;
        *)       echo "Unknown source: $ARTIFACT_SOURCE. Use gdrive | s3 | hf"; exit 1;;
    esac
}

_download_gdrive() {
    if [ -z "$GDRIVE_FOLDER_ID" ]; then
        echo ""
        echo "  ⚠️  GDRIVE_FOLDER_ID is not set in this script."
        echo "  Ask a team member for the folder ID, then set it at the top of:"
        echo "  scripts/download_data.sh"
        return
    fi
    _require_cmd gdown "pip install gdown"
    echo "Downloading from Google Drive folder: $GDRIVE_FOLDER_ID"
    gdown --folder "https://drive.google.com/drive/folders/$GDRIVE_FOLDER_ID" \
          --output artifacts_tmp
    _unpack_artifacts artifacts_tmp
    rm -rf artifacts_tmp
}

_download_s3() {
    if [ -z "$S3_BUCKET" ]; then
        echo "  ⚠️  S3_BUCKET is not set. Edit scripts/download_data.sh"; return
    fi
    _require_cmd aws "pip install awscli"
    echo "Downloading from S3: $S3_BUCKET"
    aws s3 sync "$S3_BUCKET/kg"           "$KG_DIR/"
    aws s3 sync "$S3_BUCKET/qa"           "$QA_DIR/"
    aws s3 sync "$S3_BUCKET/checkpoints"  "$CKPT_DIR/"
}

_download_hf() {
    if [ -z "$HF_REPO" ]; then
        echo "  ⚠️  HF_REPO is not set. Edit scripts/download_data.sh"; return
    fi
    _require_cmd huggingface-cli "pip install huggingface_hub"
    echo "Downloading from HuggingFace: $HF_REPO"
    huggingface-cli download "$HF_REPO" \
        --repo-type dataset \
        --local-dir . \
        --include "data/kg/*" "data/qa/*" "checkpoints/rotate/best.pt" \
                  "checkpoints/projection/projection_weights.pt"
}

_unpack_artifacts() {
    local src="$1"
    [ -d "$src/kg" ]          && cp -r "$src/kg/"*          "$KG_DIR/"   && echo "  ✓ KG artifacts"
    [ -d "$src/qa" ]          && cp -r "$src/qa/"*          "$QA_DIR/"   && echo "  ✓ QA pairs"
    [ -d "$src/checkpoints" ] && cp -r "$src/checkpoints/"* "$CKPT_DIR/" && echo "  ✓ Checkpoints"
}

_require_cmd() {
    local cmd="$1" install="$2"
    if ! command -v "$cmd" &>/dev/null; then
        echo "  '$cmd' not found. Install with: $install"; exit 1
    fi
}

# =============================================================================
# PART 3: Upload helpers (for the person who built the artifacts)
# =============================================================================

upload_artifacts() {
    echo ""
    echo "=== Uploading artifacts ==="
    echo "Run one of:"
    echo ""
    echo "  # Google Drive (install gdown first: pip install gdown)"
    echo "  # Upload manually via browser or use gdrive CLI"
    echo ""
    echo "  # S3"
    echo "  aws s3 sync data/kg/      $S3_BUCKET/kg/"
    echo "  aws s3 sync data/qa/      $S3_BUCKET/qa/"
    echo "  aws s3 sync checkpoints/  $S3_BUCKET/checkpoints/"
    echo ""
    echo "  # HuggingFace Dataset"
    echo "  huggingface-cli upload $HF_REPO data/kg/     data/kg"
    echo "  huggingface-cli upload $HF_REPO data/qa/     data/qa"
    echo "  huggingface-cli upload $HF_REPO checkpoints/ checkpoints"
    echo ""
    echo "Then set the ID/bucket/repo at the top of scripts/download_data.sh"
    echo "and commit that change — collaborators run this script to get set up."
}

# =============================================================================
# Main
# =============================================================================

echo "BioKG-LoRA data setup"
echo "====================="

$DOWNLOAD_RAW      && download_raw
$DOWNLOAD_ARTIFACTS && download_artifacts

echo ""
echo "Done. Verify with:"
echo "  python train.py --stage 0   # rebuild KG from raw data"
echo "  python train.py --stage 1   # train RotatE (needs GPU)"
