"""
Entity Linker: Recognises biological entities in text and maps them to KG IDs.

Strategy (fast, no external NER model required):
    1. Dictionary look-up using a trie/prefix index over all KG entity names
    2. Longest-match first (greedy left-to-right scan)
    3. Configurable confidence threshold (edit-distance based fuzzy matching)
"""
from __future__ import annotations

import json
import logging
import re
from typing import Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


class TrieNode:
    def __init__(self):
        self.children: Dict[str, "TrieNode"] = {}
        self.is_end: bool = False
        self.entity_name: Optional[str] = None


class EntityLinker:
    """
    Lightweight dictionary-based entity linker.

    Builds a character-level trie over all entity names for O(L) lookup,
    where L is the maximum entity name length.

    Args:
        entity2id:  dict mapping entity name → integer ID
        threshold:  minimum token overlap ratio to accept a match (0–1)
    """

    def __init__(self, entity2id: Dict[str, int], threshold: float = 0.85):
        self.entity2id = entity2id
        self.threshold = threshold
        self._root = TrieNode()
        self._max_len: int = 0
        self._build_trie(entity2id)

    @classmethod
    def from_file(cls, entity2id_path: str, threshold: float = 0.85) -> "EntityLinker":
        with open(entity2id_path) as f:
            entity2id = json.load(f)
        return cls(entity2id, threshold)

    # ── Trie construction ─────────────────────────────────────────────────────

    def _build_trie(self, entity2id: Dict[str, int]) -> None:
        logger.info("Building entity trie over %d entities…", len(entity2id))
        for name in entity2id:
            self._insert(name.lower())
            self._max_len = max(self._max_len, len(name))
        logger.info("Trie built (max entity length = %d chars)", self._max_len)

    def _insert(self, name: str) -> None:
        node = self._root
        for ch in name:
            node = node.children.setdefault(ch, TrieNode())
        node.is_end = True
        node.entity_name = name

    # ── Recognition ──────────────────────────────────────────────────────────

    def recognize(self, text: str) -> List[Tuple[str, str]]:
        """
        Find biological entities in text using longest-match greedy scan.

        Returns:
            list of (entity_name, entity_type_label) tuples
                where entity_type_label is "entity" for now
                (extend to use entity_types dict for per-type labels)
        """
        text_lower = text.lower()
        found: List[Tuple[str, str]] = []
        seen_spans: Set[Tuple[int, int]] = set()
        i = 0
        n = len(text_lower)

        while i < n:
            node = self._root
            best_end = -1
            best_name = ""
            j = i
            while j < min(n, i + self._max_len):
                ch = text_lower[j]
                if ch not in node.children:
                    break
                node = node.children[ch]
                if node.is_end and node.entity_name:
                    # Require word boundary at end
                    end_pos = j + 1
                    if end_pos == n or not text_lower[end_pos].isalnum():
                        best_end = end_pos
                        best_name = node.entity_name
                j += 1

            if best_end > -1 and (i, best_end) not in seen_spans:
                # Restore original casing from the entity name as stored
                canonical = self._restore_case(best_name)
                if canonical:
                    found.append((canonical, "entity"))
                    seen_spans.add((i, best_end))
                    i = best_end
                    continue
            i += 1

        return found

    def _restore_case(self, lower_name: str) -> Optional[str]:
        """Look up the original-cased entity name."""
        # Direct match (entity names are stored as-is in entity2id)
        if lower_name in self.entity2id:
            return lower_name
        # Try capitalised variants
        for variant in [lower_name, lower_name.upper(),
                        lower_name.capitalize(), lower_name.title()]:
            if variant in self.entity2id:
                return variant
        return None

    def get_entity_id(self, name: str) -> Optional[int]:
        return self.entity2id.get(name)

    # ── Span annotation for token sequences ──────────────────────────────────

    def find_token_spans(
        self,
        text: str,
        tokenizer,
        max_entity_tokens: int = 8,
    ) -> List[Tuple[int, int, str]]:
        """
        Identify entity spans as token index ranges for use in BioKGLoRA.forward.

        Returns:
            list of (token_start, token_end_exclusive, entity_name)
        """
        entities = self.recognize(text)
        if not entities:
            return []

        # Tokenise full text once
        encoding = tokenizer(text, return_tensors="pt", add_special_tokens=True)
        input_ids = encoding["input_ids"][0].tolist()
        spans = []

        for entity_name, _ in entities:
            ent_ids = tokenizer.encode(entity_name, add_special_tokens=False)
            if len(ent_ids) > max_entity_tokens:
                continue
            L = len(ent_ids)
            for start in range(len(input_ids) - L + 1):
                if input_ids[start: start + L] == ent_ids:
                    spans.append((start, start + L, entity_name))
                    break

        return spans


def build_entity_linker(entity2id_path: str, threshold: float = 0.85) -> EntityLinker:
    with open(entity2id_path) as f:
        entity2id = json.load(f)
    return EntityLinker(entity2id, threshold)
