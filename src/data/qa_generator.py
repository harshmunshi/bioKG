"""
Stage 0b: Automatic QA pair generation from the Knowledge Graph.

Generates four question types:
  1. Gene → Phenotype   : "What phenotypes does <gene> knockout cause?"
  2. Phenotype → Gene   : "Which genes cause phenotype <phenotype>?"
  3. Clinical Parameter : "What is the significance of elevated <param> in <gene> KO?"
  4. Multi-hop reasoning: "How does <gene> affect <tissue>?"
  5. Comparative        : "Compare the knockouts of <gene1> and <gene2>."

Each answer is constructed by traversing the KG along relevant paths.
The resulting JSONL files are used for LoRA fine-tuning (Stage 3).
"""
from __future__ import annotations

import json
import logging
import random
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import torch

logger = logging.getLogger(__name__)

# ── Clinical parameter keywords (subset of phenotype names) ──────────────────
CLINICAL_PARAMS = {
    "glucose", "ALT", "AST", "creatinine", "BUN", "albumin", "bilirubin",
    "cholesterol", "triglyceride", "hematocrit", "hemoglobin", "platelet",
    "white blood cell", "red blood cell", "neutrophil", "lymphocyte",
    "alkaline phosphatase", "ALP", "GGT", "LDH", "CK", "amylase", "lipase",
}


class QAGenerator:
    """
    Generates natural-language QA pairs by traversing the biological KG.

    Args:
        entity2id:   name → int
        relation2id: name → int
        triples:     (N, 3) tensor of (head, rel, tail)
        entity_types: id → type_id
    """

    def __init__(
        self,
        entity2id: Dict[str, int],
        relation2id: Dict[str, int],
        triples: torch.Tensor,
        entity_types: Dict[int, int],
        entity_names: Optional[Dict[int, str]] = None,
        seed: int = 42,
    ):
        self.entity2id = entity2id
        self.id2entity: Dict[int, str] = {v: k for k, v in entity2id.items()}
        self.relation2id = relation2id
        self.id2relation: Dict[int, str] = {v: k for k, v in relation2id.items()}
        self.entity_types = entity_types  # int id → type int
        self.entity_names = entity_names or {}  # int id → human-readable name (GO/MP terms)
        self.triples_tensor = triples
        self.rng = random.Random(seed)

        # Build adjacency: head_id → list[(rel_id, tail_id)]
        self._adj: Dict[int, List[Tuple[int, int]]] = {}
        for h, r, t in triples.tolist():
            self._adj.setdefault(int(h), []).append((int(r), int(t)))

        # Pre-index by relation
        self._rel_triples: Dict[int, List[Tuple[int, int]]] = {}
        for h, r, t in triples.tolist():
            self._rel_triples.setdefault(int(r), []).append((int(h), int(t)))

        # Type lookups
        self._GENE_TYPE = 0
        self._PATHWAY_TYPE = 1
        self._GO_TYPE = 2
        self._PHENO_TYPE = 3
        self._TISSUE_TYPE = 4
        self._PROTEIN_TYPE = 5

        self._causes_id = relation2id.get("causes", -1)
        self._expressed_id = relation2id.get("expressed_in", -1)
        self._part_of_id = relation2id.get("part_of", -1)
        self._participates_id = relation2id.get("participates_in", -1)
        self._interacts_id = relation2id.get("interacts_with", -1)
        self._has_function_id = relation2id.get("has_function", -1)

        # "participates_in" edges land on GO_TERM entities in this KG (KEGG
        # pathway data was never sourced, so the dedicated PATHWAY_TYPE is
        # always empty) — treat GO terms as the de facto pathway/process context.
        self._PATHWAY_LIKE_TYPES = {self._PATHWAY_TYPE, self._GO_TYPE}

    # ── Naming helpers ────────────────────────────────────────────────────────

    def _name(self, entity_id: int) -> str:
        """Human-readable name if one was resolved (GO/MP terms), else the raw code."""
        return self.entity_names.get(entity_id) or self.id2entity[entity_id]

    def _display(self, entity_id: int) -> str:
        """'CODE (name)' for ontology-coded entities, or just the symbol for genes."""
        code = self.id2entity[entity_id]
        name = self.entity_names.get(entity_id)
        return f"{code} ({name})" if name else code

    # ── Public API ─────────────────────────────────────────────────────────────

    def generate(
        self,
        n_train: int = 10000,
        n_val: int = 1000,
        n_test: int = 500,
        weights: Optional[Dict[str, float]] = None,
    ) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        """Generate train/val/test QA datasets."""
        weights = weights or {
            "gene_phenotype": 0.30,
            "phenotype_gene": 0.20,
            "clinical_parameter": 0.25,
            "multi_hop": 0.15,
            "comparative": 0.10,
        }
        total = n_train + n_val + n_test
        all_qa = self._generate_all(total, weights)
        self.rng.shuffle(all_qa)
        train = all_qa[:n_train]
        val = all_qa[n_train: n_train + n_val]
        test = all_qa[n_train + n_val: n_train + n_val + n_test]
        return train, val, test

    def save(self, qa_dir: str, train: List[Dict], val: List[Dict], test: List[Dict]) -> None:
        out = Path(qa_dir)
        out.mkdir(parents=True, exist_ok=True)
        for name, split in [("train", train), ("val", val), ("test", test)]:
            path = out / f"{name}.jsonl"
            with open(path, "w") as f:
                for item in split:
                    f.write(json.dumps(item) + "\n")
            logger.info("  Saved %d QA pairs to %s", len(split), path)

    # ── Internal generation ───────────────────────────────────────────────────

    def _generate_all(self, n: int, weights: Dict[str, float]) -> List[Dict]:
        generators = {
            "gene_phenotype": self._gene_phenotype_qa,
            "phenotype_gene": self._phenotype_gene_qa,
            "clinical_parameter": self._clinical_parameter_qa,
            "multi_hop": self._multi_hop_qa,
            "comparative": self._comparative_qa,
        }
        qa_list: List[Dict] = []
        keys = list(weights.keys())
        probs = [weights[k] for k in keys]
        # Normalise
        total_w = sum(probs)
        probs = [p / total_w for p in probs]

        attempts = 0
        max_attempts = n * 20
        while len(qa_list) < n and attempts < max_attempts:
            attempts += 1
            kind = self.rng.choices(keys, weights=probs, k=1)[0]
            qa = generators[kind]()
            if qa is not None:
                qa["type"] = kind
                qa_list.append(qa)

        if len(qa_list) < n:
            logger.warning("Only generated %d/%d QA pairs after %d attempts", len(qa_list), n, attempts)
        return qa_list

    _GENE_PHENOTYPE_QUESTIONS = [
        "What phenotypes are associated with {gene} knockout?",
        "What phenotypes does {gene} knockout cause?",
        "What happens when {gene} is knocked out?",
        "Which phenotypes result from loss of {gene} function?",
    ]

    def _gene_phenotype_qa(self) -> Optional[Dict]:
        """What phenotypes does <gene> knockout cause?"""
        genes = self._entities_of_type(self._GENE_TYPE)
        if not genes:
            return None
        gene_id = self.rng.choice(genes)
        gene_name = self.id2entity[gene_id]
        phenotypes = [
            t for r, t in self._adj.get(gene_id, [])
            if r == self._causes_id and self.entity_types.get(t) == self._PHENO_TYPE
        ]
        if not phenotypes:
            return None
        phenotypes = phenotypes[:5]
        pheno_names = [self.id2entity[p] for p in phenotypes]
        pheno_list = "\n".join(f"  - {self._display(p)}" for p in phenotypes)
        question = self.rng.choice(self._GENE_PHENOTYPE_QUESTIONS).format(gene=gene_name)
        answer = (
            f"{gene_name} knockout results in the following phenotypes:\n"
            f"{pheno_list}\n\n"
            f"These phenotypes arise because {gene_name} plays a critical role in "
            f"biological processes connected to these manifestations through the "
            f"gene-phenotype relationships encoded in the biological knowledge graph."
        )
        return {
            "question": question,
            "answer": answer,
            "entities": [gene_name] + pheno_names,
        }

    _PHENOTYPE_GENE_QUESTIONS = [
        "Which genes are associated with the phenotype: {pheno}?",
        "Which genes cause the phenotype {pheno}?",
        "What genes, when knocked out, produce the phenotype {pheno}?",
    ]

    def _phenotype_gene_qa(self) -> Optional[Dict]:
        """Which genes cause phenotype <X>?"""
        phenos = self._entities_of_type(self._PHENO_TYPE)
        if not phenos:
            return None
        pheno_id = self.rng.choice(phenos)
        pheno_display = self._display(pheno_id)
        caused_by_id = self.relation2id.get("caused_by", -1)
        genes = [
            self.id2entity[t]
            for r, t in self._adj.get(pheno_id, [])
            if r == caused_by_id and self.entity_types.get(t) == self._GENE_TYPE
        ]
        if not genes:
            return None
        genes = genes[:5]
        gene_list = ", ".join(genes)
        question = self.rng.choice(self._PHENOTYPE_GENE_QUESTIONS).format(pheno=pheno_display)
        answer = (
            f"The phenotype {pheno_display} is caused by knockout of the following genes: "
            f"{gene_list}. These genes share common biological roles or pathways that "
            f"converge on this phenotypic manifestation."
        )
        return {
            "question": question,
            "answer": answer,
            "entities": genes + [self.id2entity[pheno_id]],
        }

    _CLINICAL_PARAMETER_QUESTIONS = [
        "What is the significance of elevated {param} in a {gene} knockout mouse?",
        "What is the significance of '{param}' in {gene} knockout?",
        "Why does {gene} knockout affect {param}?",
    ]

    def _clinical_parameter_qa(self) -> Optional[Dict]:
        """What is the significance of elevated <param> in <gene> knockout?"""
        # Find phenotypes that sound like clinical parameters — match against the
        # resolved human-readable name (bare MP: codes never contain words like
        # "glucose", so matching the raw code here always misses).
        phenos = self._entities_of_type(self._PHENO_TYPE)
        clinical_phenos = [
            p for p in phenos
            if any(kw.lower() in self._name(p).lower() for kw in CLINICAL_PARAMS)
        ]
        if not clinical_phenos:
            return None
        pheno_id = self.rng.choice(clinical_phenos)
        pheno_display = self._display(pheno_id)
        param_name = self._name(pheno_id)
        caused_by_id = self.relation2id.get("caused_by", -1)
        genes = [
            self.id2entity[t]
            for r, t in self._adj.get(pheno_id, [])
            if r == caused_by_id and self.entity_types.get(t) == self._GENE_TYPE
        ]
        if not genes:
            return None
        gene_name = self.rng.choice(genes)
        gene_id = self.entity2id[gene_name]

        # Functional/pathway context — "participates_in" edges land on GO_TERM
        # entities here (no KEGG pathway data was sourced), so GO terms serve as
        # the de facto pathway context.
        pathways = [
            self.id2entity[t]
            for r, t in self._adj.get(gene_id, [])
            if r == self._participates_id and self.entity_types.get(t) in self._PATHWAY_LIKE_TYPES
        ]
        # Direct interaction partners give a mechanistic hop beyond the gene itself.
        partners = [
            self.id2entity[t]
            for r, t in self._adj.get(gene_id, [])
            if r == self._interacts_id and self.entity_types.get(t) == self._GENE_TYPE
        ]

        steps = [f"1. {gene_name} regulates biological processes"]
        if pathways:
            steps[0] += f" within {self._display(self.entity2id[pathways[0]])}"
        step_n = 2
        if partners:
            steps.append(f"{step_n}. {gene_name} interacts with {partners[0]}, extending this effect mechanistically")
            step_n += 1
        steps.append(f"{step_n}. Loss of {gene_name} function disrupts these processes")
        step_n += 1
        steps.append(f"{step_n}. This connects to phenotype {pheno_display}")

        related = ", ".join(self._display(self.entity2id[p]) for p in pathways[:3])
        question = self.rng.choice(self._CLINICAL_PARAMETER_QUESTIONS).format(
            param=param_name, gene=gene_name
        )
        answer = (
            f"Elevated {param_name} in {gene_name} knockout is significant because:\n"
            + "\n".join(steps)
            + (f"\nRelated pathways/processes: {related}" if related else "")
        )
        return {
            "question": question,
            "answer": answer,
            "entities": [gene_name, self.id2entity[pheno_id]] + pathways[:2] + partners[:1],
        }

    _MULTI_HOP_QUESTIONS = [
        "How does {gene} knockout lead to its observed phenotype?",
        "What is the mechanistic pathway from {gene} knockout to its phenotype?",
        "Explain the biological mechanism by which {gene} knockout causes phenotypic changes.",
    ]

    def _multi_hop_qa(self) -> Optional[Dict]:
        """How does <gene> knockout lead to <phenotype>? — traces a 2-3 hop mechanistic path.

        Routes through interacts_with (gene-gene) and participates_in
        (gene-GO term) rather than expressed_in/tissue: this KG has zero
        expressed_in/tissue edges (GTEx data was never sourced), so a
        tissue-expression-based path can never be found.
        """
        genes = self._entities_of_type(self._GENE_TYPE)
        if not genes:
            return None
        gene_id = self.rng.choice(genes)
        gene_name = self.id2entity[gene_id]

        partners = [
            t for r, t in self._adj.get(gene_id, [])
            if r == self._interacts_id and self.entity_types.get(t) == self._GENE_TYPE
        ]
        if not partners:
            return None
        partner_id = self.rng.choice(partners)
        partner_name = self.id2entity[partner_id]

        pathways = [
            t for r, t in self._adj.get(gene_id, [])
            if r == self._participates_id and self.entity_types.get(t) in self._PATHWAY_LIKE_TYPES
        ]

        phenos = [
            t for r, t in self._adj.get(gene_id, [])
            if r == self._causes_id and self.entity_types.get(t) == self._PHENO_TYPE
        ]
        if not phenos:
            return None
        pheno_id = self.rng.choice(phenos)
        pheno_display = self._display(pheno_id)

        steps = [f"1. {gene_name} interacts with {partner_name}"]
        step_n = 2
        if pathways:
            steps.append(f"{step_n}. This interaction occurs within {self._display(pathways[0])}")
            step_n += 1
        steps.append(f"{step_n}. Knockout of {gene_name} disrupts this interaction")
        step_n += 1
        steps.append(f"{step_n}. This leads to phenotype {pheno_display}")

        question = self.rng.choice(self._MULTI_HOP_QUESTIONS).format(gene=gene_name)
        answer = (
            f"{gene_name} knockout leads to {pheno_display} through the following mechanistic path:\n"
            + "\n".join(steps)
            + f"\n\nThis multi-step cascade explains the observed phenotype when {gene_name} function is lost."
        )
        return {
            "question": question,
            "answer": answer,
            "entities": [gene_name, partner_name, self.id2entity[pheno_id]]
            + [self.id2entity[p] for p in pathways[:1]],
        }

    def _comparative_qa(self) -> Optional[Dict]:
        """Compare the knockouts of <gene1> and <gene2>."""
        genes = self._entities_of_type(self._GENE_TYPE)
        if len(genes) < 2:
            return None
        g1_id, g2_id = self.rng.sample(genes, 2)
        g1, g2 = self.id2entity[g1_id], self.id2entity[g2_id]

        phenos1 = set(
            t for r, t in self._adj.get(g1_id, [])
            if r == self._causes_id and self.entity_types.get(t) == self._PHENO_TYPE
        )
        phenos2 = set(
            t for r, t in self._adj.get(g2_id, [])
            if r == self._causes_id and self.entity_types.get(t) == self._PHENO_TYPE
        )
        if not phenos1 or not phenos2:
            return None
        shared = phenos1 & phenos2
        unique1 = phenos1 - phenos2
        unique2 = phenos2 - phenos1

        question = f"Compare the knockout phenotypes of {g1} and {g2}."
        shared_display = [self._display(p) for p in list(shared)[:3]]
        shared_str = (
            "share the following phenotypes: " + ", ".join(shared_display)
            if shared else "do not share obvious phenotypes"
        )
        unique1_display = [self._display(p) for p in list(unique1)[:3]]
        unique2_display = [self._display(p) for p in list(unique2)[:3]]
        answer = (
            f"Knockout comparison of {g1} vs {g2}:\n\n"
            f"• Shared: Both {g1} and {g2} {shared_str}.\n"
            f"• {g1}-specific: {', '.join(unique1_display) or 'none identified'}.\n"
            f"• {g2}-specific: {', '.join(unique2_display) or 'none identified'}.\n\n"
            f"This comparison reveals both convergent and divergent biological roles "
            f"of these two genes."
        )
        return {
            "question": question,
            "answer": answer,
            "entities": [g1, g2]
            + [self.id2entity[p] for p in list(shared)[:2]]
            + [self.id2entity[p] for p in list(unique1)[:1]]
            + [self.id2entity[p] for p in list(unique2)[:1]],
        }

    def _entities_of_type(self, type_id: int) -> List[int]:
        return [eid for eid, tid in self.entity_types.items() if tid == type_id]
