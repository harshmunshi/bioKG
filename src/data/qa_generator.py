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
        seed: int = 42,
    ):
        self.entity2id = entity2id
        self.id2entity: Dict[int, str] = {v: k for k, v in entity2id.items()}
        self.relation2id = relation2id
        self.id2relation: Dict[int, str] = {v: k for k, v in relation2id.items()}
        self.entity_types = entity_types  # int id → type int
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
        pheno_names = [self.id2entity[p] for p in phenotypes[:5]]
        pheno_list = "\n".join(f"  - {p}" for p in pheno_names)
        question = f"What phenotypes are associated with {gene_name} knockout?"
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

    def _phenotype_gene_qa(self) -> Optional[Dict]:
        """Which genes cause phenotype <X>?"""
        phenos = self._entities_of_type(self._PHENO_TYPE)
        if not phenos:
            return None
        pheno_id = self.rng.choice(phenos)
        pheno_name = self.id2entity[pheno_id]
        caused_by_id = self.relation2id.get("caused_by", -1)
        genes = [
            self.id2entity[t]
            for r, t in self._adj.get(pheno_id, [])
            if r == caused_by_id and self.entity_types.get(t) == self._GENE_TYPE
        ]
        if not genes:
            return None
        gene_list = ", ".join(genes[:5])
        question = f"Which genes are associated with the phenotype: {pheno_name}?"
        answer = (
            f"The phenotype '{pheno_name}' is caused by knockout of the following genes: "
            f"{gene_list}. These genes share common biological roles or pathways that "
            f"converge on this phenotypic manifestation."
        )
        return {
            "question": question,
            "answer": answer,
            "entities": genes[:5] + [pheno_name],
        }

    def _clinical_parameter_qa(self) -> Optional[Dict]:
        """What is the significance of elevated <param> in <gene> knockout?"""
        # Find phenotypes that sound like clinical parameters
        phenos = self._entities_of_type(self._PHENO_TYPE)
        clinical_phenos = [
            p for p in phenos
            if any(kw.lower() in self.id2entity[p].lower() for kw in CLINICAL_PARAMS)
        ]
        if not clinical_phenos:
            return None
        pheno_id = self.rng.choice(clinical_phenos)
        pheno_name = self.id2entity[pheno_id]
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
        # Find pathways for the gene
        pathways = [
            self.id2entity[t]
            for r, t in self._adj.get(gene_id, [])
            if r == self._participates_id and self.entity_types.get(t) == self._PATHWAY_TYPE
        ]
        pathway_str = f" through its role in {pathways[0]}" if pathways else ""
        question = f"What is the significance of '{pheno_name}' in {gene_name} knockout?"
        answer = (
            f"The observation of '{pheno_name}' in {gene_name} knockout is significant "
            f"because {gene_name} regulates biological processes{pathway_str}. "
            f"Loss of {gene_name} function disrupts these processes, leading to the "
            f"clinical manifestation of {pheno_name}. This phenotype serves as a "
            f"measurable biomarker of the underlying molecular dysfunction."
        )
        return {
            "question": question,
            "answer": answer,
            "entities": [gene_name, pheno_name] + pathways[:2],
        }

    def _multi_hop_qa(self) -> Optional[Dict]:
        """How does <gene> affect <tissue>? — traces a 2–3 hop path."""
        genes = self._entities_of_type(self._GENE_TYPE)
        tissues = self._entities_of_type(self._TISSUE_TYPE)
        if not genes or not tissues:
            return None
        gene_id = self.rng.choice(genes)
        gene_name = self.id2entity[gene_id]

        # Find tissues connected via gene → expressed_in
        direct_tissues = [
            t for r, t in self._adj.get(gene_id, [])
            if r == self._expressed_id and self.entity_types.get(t) == self._TISSUE_TYPE
        ]
        if not direct_tissues:
            return None
        tissue_id = self.rng.choice(direct_tissues)
        tissue_name = self.id2entity[tissue_id]

        # Find phenotypes connected to gene
        phenos = [
            self.id2entity[t]
            for r, t in self._adj.get(gene_id, [])
            if r == self._causes_id and self.entity_types.get(t) == self._PHENO_TYPE
        ]
        # Find pathways
        pathways = [
            self.id2entity[t]
            for r, t in self._adj.get(gene_id, [])
            if r == self._participates_id
        ]

        steps = [f"1. {gene_name} is expressed in {tissue_name}"]
        if pathways:
            steps.append(f"2. {gene_name} participates in {pathways[0]}")
        if phenos:
            steps.append(f"3. Knockout of {gene_name} causes: {', '.join(phenos[:2])}")

        question = f"How does {gene_name} knockout affect {tissue_name}?"
        answer = (
            f"{gene_name} knockout affects {tissue_name} through the following pathway:\n"
            + "\n".join(steps)
            + f"\n\nThis multi-step cascade explains why {tissue_name} is impacted when "
            + f"{gene_name} function is lost."
        )
        return {
            "question": question,
            "answer": answer,
            "entities": [gene_name, tissue_name] + pathways[:1] + phenos[:2],
        }

    def _comparative_qa(self) -> Optional[Dict]:
        """Compare the knockouts of <gene1> and <gene2>."""
        genes = self._entities_of_type(self._GENE_TYPE)
        if len(genes) < 2:
            return None
        g1_id, g2_id = self.rng.sample(genes, 2)
        g1, g2 = self.id2entity[g1_id], self.id2entity[g2_id]

        phenos1 = set(
            self.id2entity[t]
            for r, t in self._adj.get(g1_id, [])
            if r == self._causes_id and self.entity_types.get(t) == self._PHENO_TYPE
        )
        phenos2 = set(
            self.id2entity[t]
            for r, t in self._adj.get(g2_id, [])
            if r == self._causes_id and self.entity_types.get(t) == self._PHENO_TYPE
        )
        if not phenos1 or not phenos2:
            return None
        shared = phenos1 & phenos2
        unique1 = phenos1 - phenos2
        unique2 = phenos2 - phenos1

        question = f"Compare the knockout phenotypes of {g1} and {g2}."
        shared_str = (
            "share the following phenotypes: " + ", ".join(list(shared)[:3])
            if shared else "do not share obvious phenotypes"
        )
        answer = (
            f"Knockout comparison of {g1} vs {g2}:\n\n"
            f"• Shared: Both {g1} and {g2} {shared_str}.\n"
            f"• {g1}-specific: {', '.join(list(unique1)[:3]) or 'none identified'}.\n"
            f"• {g2}-specific: {', '.join(list(unique2)[:3]) or 'none identified'}.\n\n"
            f"This comparison reveals both convergent and divergent biological roles "
            f"of these two genes."
        )
        return {
            "question": question,
            "answer": answer,
            "entities": [g1, g2] + list(shared)[:2] + list(unique1)[:1] + list(unique2)[:1],
        }

    def _entities_of_type(self, type_id: int) -> List[int]:
        return [eid for eid, tid in self.entity_types.items() if tid == type_id]
