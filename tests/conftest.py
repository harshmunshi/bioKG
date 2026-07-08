"""Shared fixtures for Stage 1 (RotatE) output tests."""
import json
from pathlib import Path

import pytest
import torch

KG_DIR = Path(__file__).resolve().parent.parent / "data" / "kg"


@pytest.fixture(scope="session")
def kg_dir() -> Path:
    if not (KG_DIR / "entity_embeddings.pt").exists():
        pytest.skip(f"Stage 1 outputs not found in {KG_DIR} — run `python train.py --stage 1` first")
    return KG_DIR


@pytest.fixture(scope="session")
def entity2id(kg_dir: Path) -> dict:
    with open(kg_dir / "entity2id.json", encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="session")
def id2entity(entity2id: dict) -> dict:
    return {v: k for k, v in entity2id.items()}


@pytest.fixture(scope="session")
def entity_embeddings(kg_dir: Path) -> torch.Tensor:
    return torch.load(str(kg_dir / "entity_embeddings.pt"), map_location="cpu")


@pytest.fixture(scope="session")
def relation_embeddings(kg_dir: Path) -> torch.Tensor:
    return torch.load(str(kg_dir / "relation_embeddings.pt"), map_location="cpu")
