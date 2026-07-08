"""
BioQA Dataset for LoRA fine-tuning (Stage 3).

Reads JSONL files produced by qa_generator.py, tokenises them with a
chat-template prompt format, and performs entity linking to extract spans
that will be augmented with KG embeddings during the forward pass.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import PreTrainedTokenizer

logger = logging.getLogger(__name__)

PROMPT_TEMPLATE = (
    "### Biological Reasoning Question\n"
    "{question}\n\n"
    "### Answer\n"
    "{answer}"
)


class BioQADataset(Dataset):
    """
    Tokenised QA dataset for causal language-model fine-tuning.

    Each sample is formatted as:
        ### Biological Reasoning Question
        <question>
        ### Answer
        <answer>

    The model is trained to predict the answer tokens only
    (question tokens get label = -100 to be ignored in loss).

    Args:
        path:         path to .jsonl file
        tokenizer:    HuggingFace tokenizer (with chat template support)
        max_length:   maximum token length (longer sequences are truncated)
        entity_linker: optional entity linker for KG span annotation
    """

    def __init__(
        self,
        path: str,
        tokenizer: PreTrainedTokenizer,
        max_length: int = 2048,
        entity_linker=None,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.entity_linker = entity_linker
        self.samples: List[Dict] = []

        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.samples.append(json.loads(line))

        logger.info("Loaded %d QA samples from %s", len(self.samples), path)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]
        question = sample["question"]
        answer = sample["answer"]

        # Append EOS so the model learns to stop generating after the answer —
        # without it, nothing in training ever signals "end here", and at
        # inference generation runs past the real answer into degenerate loops.
        full_text = PROMPT_TEMPLATE.format(question=question, answer=answer) + self.tokenizer.eos_token
        question_part = PROMPT_TEMPLATE.format(question=question, answer="").rstrip()

        # Tokenise full text
        encoding = self.tokenizer(
            full_text,
            max_length=self.max_length,
            truncation=True,
            padding=False,
            return_tensors="pt",
        )
        input_ids = encoding["input_ids"].squeeze(0)         # (L,)
        attention_mask = encoding["attention_mask"].squeeze(0)

        # Tokenise question only to find where the answer starts
        q_encoding = self.tokenizer(
            question_part,
            max_length=self.max_length,
            truncation=True,
            padding=False,
            return_tensors="pt",
        )
        q_len = q_encoding["input_ids"].shape[1]

        # Labels: ignore question tokens, predict answer tokens
        labels = input_ids.clone()
        labels[:q_len] = -100

        result: Dict[str, torch.Tensor] = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

        # Optionally annotate entity spans for KG augmentation
        if self.entity_linker is not None:
            entity_spans = self._get_entity_spans(full_text, input_ids)
            # Store as a list — collate_fn will handle it separately
            result["entity_spans"] = entity_spans  # type: ignore[assignment]

        return result

    def _get_entity_spans(
        self,
        text: str,
        input_ids: torch.Tensor,
    ) -> List[Tuple[int, int, str]]:
        """
        Find entity spans in the tokenised sequence.
        Returns list of (token_start, token_end, entity_name).
        """
        entities = self.entity_linker.recognize(text)
        spans: List[Tuple[int, int, str]] = []
        for entity_name, _ in entities:
            # Find token positions by encoding just the entity name
            ent_ids = self.tokenizer.encode(
                entity_name, add_special_tokens=False
            )
            seq = input_ids.tolist()
            for start in range(len(seq) - len(ent_ids) + 1):
                if seq[start: start + len(ent_ids)] == ent_ids:
                    spans.append((start, start + len(ent_ids), entity_name))
                    break
        return spans


def collate_fn(batch: List[Dict], pad_token_id: int = 0) -> Dict:
    """
    Custom collate that pads variable-length sequences and keeps
    entity_spans as a list-of-lists (not tensored).
    """
    # Pad input_ids, attention_mask, labels to max length in batch
    max_len = max(x["input_ids"].shape[0] for x in batch)

    input_ids_list, attn_list, label_list = [], [], []
    for x in batch:
        pad = max_len - x["input_ids"].shape[0]
        input_ids_list.append(
            torch.cat([x["input_ids"], torch.full((pad,), pad_token_id, dtype=torch.long)])
        )
        attn_list.append(
            torch.cat([x["attention_mask"], torch.zeros(pad, dtype=torch.long)])
        )
        label_list.append(
            torch.cat([x["labels"], torch.full((pad,), -100, dtype=torch.long)])
        )

    out = {
        "input_ids": torch.stack(input_ids_list),
        "attention_mask": torch.stack(attn_list),
        "labels": torch.stack(label_list),
    }

    if "entity_spans" in batch[0]:
        out["entity_spans"] = [x["entity_spans"] for x in batch]  # type: ignore[assignment]

    return out


def build_qa_dataloaders(
    train_path: str,
    val_path: str,
    tokenizer: PreTrainedTokenizer,
    max_length: int,
    batch_size: int,
    num_workers: int = 0,
    entity_linker=None,
) -> Tuple[DataLoader, DataLoader]:
    pad_id = tokenizer.pad_token_id or 0

    train_ds = BioQADataset(train_path, tokenizer, max_length, entity_linker)
    val_ds = BioQADataset(val_path, tokenizer, max_length, entity_linker)

    _collate = lambda b: collate_fn(b, pad_token_id=pad_id)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=(num_workers > 0), collate_fn=_collate,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=(num_workers > 0), collate_fn=_collate,
    )
    return train_loader, val_loader
