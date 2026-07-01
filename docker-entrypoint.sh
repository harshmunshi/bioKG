#!/usr/bin/env bash
# docker-entrypoint.sh — runs dvc pull before delegating to train.py
set -euo pipefail

# ── DVC pull ──────────────────────────────────────────────────────────────────
# Uses the public-read S3 remote so no AWS credentials are required for pull.
# dvc pull is idempotent: already-present files are skipped after a fast MD5
# check, so re-running a container after data is downloaded is nearly free.
#
# To skip the pull entirely (e.g. data already on a mounted volume):
#   docker compose run -e SKIP_DVC_PULL=1 lora --stage 3

if [[ "${SKIP_DVC_PULL:-0}" != "1" ]]; then
    echo "[entrypoint] Running dvc pull (remote: s3remote-public)..."
    # Fall back gracefully — training may still work if data was pre-mounted.
    dvc pull --remote s3remote-public --allow-missing 2>&1 || \
        echo "[entrypoint] WARNING: dvc pull failed — proceeding with existing data on disk."
else
    echo "[entrypoint] SKIP_DVC_PULL=1 — skipping dvc pull."
fi

# ── Delegate to train.py ──────────────────────────────────────────────────────
exec python train.py "$@"
