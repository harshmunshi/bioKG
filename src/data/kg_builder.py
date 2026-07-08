"""
Stage 0: Knowledge Graph Construction

Builds a unified biological KG from:
  - MGI (Mouse Genome Informatics) — genes, gene-phenotype associations
  - GO  (Gene Ontology)            — molecular functions, processes, components
  - KEGG (Pathways)                — pathway memberships
  - STRING (Protein interactions)  — PPI network
  - MPO  (Mammalian Phenotype Ontology) — phenotype hierarchy
  - GTEx (Tissue Expression)       — gene-tissue expression

Output: entity2id.json, relation2id.json, entity_types.json, *.pt graph files.
"""
from __future__ import annotations

import json
import logging
import os
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import torch

logger = logging.getLogger(__name__)

# ── Canonical ontologies ──────────────────────────────────────────────────────

ENTITY_TYPES: Dict[str, int] = {
    "gene": 0,
    "pathway": 1,
    "go_term": 2,
    "phenotype": 3,
    "tissue": 4,
    "protein": 5,
}

RELATION_TYPES: Dict[str, int] = {
    "regulates": 0,
    "regulated_by": 1,
    "part_of": 2,
    "has_part": 3,
    "expressed_in": 4,
    "expresses": 5,
    "causes": 6,
    "caused_by": 7,
    "interacts_with": 8,
    "located_in": 9,
    "location_of": 10,
    "has_function": 11,
    "function_of": 12,
    "participates_in": 13,
    "associated_with": 14,
}


class KGBuilder:
    """
    Incrementally builds a biological knowledge graph.

    Usage:
        builder = KGBuilder()
        builder.load_mgi(path)
        builder.load_go(path)
        ...
        builder.save(output_dir)
    """

    def __init__(self, cfg: Optional[dict] = None):
        self.cfg = cfg or {}
        self.entity2id: Dict[str, int] = {}
        self.entity_type: Dict[int, int] = {}       # entity_id → type_id
        self.entity_names: Dict[int, str] = {}       # entity_id → human-readable name
        self.relation2id: Dict[str, int] = RELATION_TYPES.copy()
        self.triples: List[Tuple[int, int, int]] = []  # (head, rel, tail)
        self._triple_set: set = set()  # de-dup

    # ── Entity / triple management ────────────────────────────────────────────

    def _add_entity(self, name: str, entity_type: str) -> int:
        if name not in self.entity2id:
            eid = len(self.entity2id)
            self.entity2id[name] = eid
            self.entity_type[eid] = ENTITY_TYPES[entity_type]
        return self.entity2id[name]

    def _set_name(self, entity_key: str, display_name: str) -> None:
        """Attach a human-readable name to an already-registered entity.

        Ontology codes (MP:XXXXXXX, GO:XXXXXXX) carry no biological meaning on
        their own — without this, QA generation and answer formatting can only
        ever reference the bare code.
        """
        eid = self.entity2id.get(entity_key)
        if eid is not None and display_name:
            self.entity_names[eid] = display_name

    def _add_triple(self, head: str, relation: str, tail: str) -> None:
        if head not in self.entity2id or tail not in self.entity2id:
            return
        if relation not in self.relation2id:
            logger.warning("Unknown relation %s — skipped", relation)
            return
        h, r, t = self.entity2id[head], self.relation2id[relation], self.entity2id[tail]
        key = (h, r, t)
        if key not in self._triple_set:
            self._triple_set.add(key)
            self.triples.append(key)

    # ── Data loaders ──────────────────────────────────────────────────────────

    def load_mgi_genes(self, path: str) -> None:
        """
        Parse MGI_MRK_List2.rpt (tab-separated).
        Expected columns: 'Marker Symbol', 'MGI Accession ID', 'Marker Type'
        """
        logger.info("Loading MGI genes from %s", path)
        df = pd.read_csv(path, sep="\t", low_memory=False)
        col_sym = _find_col(df, ["Marker Symbol", "MarkerSymbol", "symbol"])
        col_type = _find_col(df, ["Marker Type", "MarkerType", "type"], required=False)
        added = 0
        for _, row in df.iterrows():
            sym = str(row[col_sym]).strip()
            if not sym or sym == "nan":
                continue
            mtype = str(row[col_type]).strip().lower() if col_type else "gene"
            if "gene" in mtype or mtype == "nan":
                self._add_entity(sym, "gene")
                added += 1
        logger.info("  Added %d gene entities", added)

    def load_mgi_phenotypes(self, path: str) -> None:
        """
        Parse MGI_PhenoGenoMP.rpt (no header row).

        Columns (0-indexed):
            0: Allelic Composition  e.g. Rb1<tm1Tyj>/Rb1<tm1Tyj>
            1: Allele Symbols       e.g. Rb1<tm1Tyj>   ← gene symbol is before '<'
            2: Genetic Background
            3: Mammalian Phenotype ID  e.g. MP:0000600
            4: PubMed ID
            5: MGI Marker Accession ID(s)  pipe-separated
            6: MGI Genotype Accession ID
        """
        logger.info("Loading gene-phenotype associations from %s", path)
        df = pd.read_csv(path, sep="\t", header=None, low_memory=False,
                         names=["allelic_comp", "allele_symbols", "background",
                                "mp_id", "pubmed_id", "mgi_ids", "genotype_id"])
        added = 0
        for _, row in df.iterrows():
            mp_id = str(row["mp_id"]).strip()
            if not mp_id.startswith("MP:"):
                continue

            # Extract gene symbols from allele symbols column
            # e.g. "Rb1<tm1Tyj>" → "Rb1",  "Brca1<tm2>, Trp53<tm1>" → ["Brca1", "Trp53"]
            allele_str = str(row["allele_symbols"])
            genes = []
            for allele in allele_str.split(","):
                allele = allele.strip()
                gene = allele.split("<")[0].strip()  # everything before '<'
                if gene and gene != "nan":
                    genes.append(gene)

            self._add_entity(mp_id, "phenotype")

            for gene in genes:
                if gene not in self.entity2id:
                    continue
                self._add_triple(gene, "causes", mp_id)
                self._add_triple(mp_id, "caused_by", gene)
                added += 1

        logger.info("  Added %d gene-phenotype triples", added)

    def load_go_annotations(self, path: str) -> None:
        """
        Parse GAF 2.x format (tab-separated, '!' comment lines).
        Columns: 1=DB_Object_Symbol, 4=GO_ID, 8=Aspect (F/P/C)
        """
        logger.info("Loading GO annotations from %s", path)
        rows = []
        with open(path) as fh:
            for line in fh:
                if line.startswith("!"):
                    continue
                parts = line.strip().split("\t")
                if len(parts) < 9:
                    continue
                rows.append(parts)
        added = 0
        for parts in rows:
            gene = parts[2].strip()
            go_id = parts[4].strip()
            aspect = parts[8].strip()
            if gene not in self.entity2id:
                continue
            self._add_entity(go_id, "go_term")
            rel_map = {"F": "has_function", "P": "participates_in", "C": "located_in"}
            inv_map = {"F": "function_of", "P": "participates_in", "C": "location_of"}
            rel = rel_map.get(aspect)
            inv = inv_map.get(aspect)
            if rel:
                self._add_triple(gene, rel, go_id)
                self._add_triple(go_id, inv, gene)
                added += 1
        logger.info("  Added %d GO annotation triples", added)

    def load_go_hierarchy(self, obo_path: str) -> None:
        """
        Parse go.obo and add is_a / part_of hierarchy edges.
        Requires: obonet (pip install obonet)
        """
        logger.info("Loading GO hierarchy from %s", obo_path)
        try:
            import obonet
        except ImportError:
            logger.warning("obonet not installed — skipping GO hierarchy. pip install obonet")
            return
        go_graph = obonet.read_obo(obo_path)
        added = 0
        for go_id, data in go_graph.nodes(data=True):
            if go_id not in self.entity2id:
                continue
            self._set_name(go_id, data.get("name"))
            for parent_id in data.get("is_a", []):
                # is_a entries look like "GO:XXXXXXX ! label"
                pid = parent_id.split(" ")[0].strip()
                if pid in self.entity2id:
                    self._add_triple(go_id, "part_of", pid)
                    self._add_triple(pid, "has_part", go_id)
                    added += 1
        logger.info("  Added %d GO hierarchy triples", added)

    def load_phenotype_names(self, obo_path: str) -> None:
        """
        Parse mpo.obo (Mammalian Phenotype Ontology) to attach a human-readable
        name to every MP: code already registered via load_mgi_phenotypes.
        Without this, phenotypes are only ever referenceable as bare codes
        (e.g. "MP:0005397") with no biological meaning attached.
        Requires: obonet (pip install obonet)
        """
        logger.info("Loading phenotype names from %s", obo_path)
        try:
            import obonet
        except ImportError:
            logger.warning("obonet not installed — skipping phenotype names. pip install obonet")
            return
        mp_graph = obonet.read_obo(obo_path)
        named = 0
        for mp_id, data in mp_graph.nodes(data=True):
            if mp_id not in self.entity2id:
                continue
            self._set_name(mp_id, data.get("name"))
            named += 1
        logger.info("  Named %d phenotype entities", named)

    def load_kegg_pathways(self, path: str) -> None:
        """
        Parse KEGG pathway-gene mapping file (tab-separated).
        Expected columns: 'pathway_id', 'gene_symbol'
        """
        logger.info("Loading KEGG pathways from %s", path)
        df = pd.read_csv(path, sep="\t", low_memory=False)
        col_pathway = _find_col(df, ["pathway_id", "PathwayID", "pathway"])
        col_gene = _find_col(df, ["gene_symbol", "GeneSymbol", "gene"])
        added = 0
        for pathway_id, grp in df.groupby(col_pathway):
            pathway_name = f"KEGG:{pathway_id}"
            self._add_entity(pathway_name, "pathway")
            for gene in grp[col_gene].dropna().unique():
                gene = str(gene).strip()
                if gene in self.entity2id:
                    self._add_triple(gene, "participates_in", pathway_name)
                    self._add_triple(pathway_name, "associated_with", gene)
                    added += 1
        logger.info("  Added %d KEGG pathway triples", added)

    def load_string_interactions(
        self,
        links_path: str,
        info_path: str,
        min_confidence: int = 400,
    ) -> None:
        """
        Parse STRING protein.links and protein.info files.
        Only include interactions above min_confidence (0-1000 scale).
        """
        logger.info("Loading STRING PPIs from %s", links_path)
        info = pd.read_csv(info_path, sep="\t", low_memory=False)
        col_prot = _find_col(info, ["protein_external_id", "#string_protein_id", "protein_id"])
        col_gene = _find_col(info, ["preferred_name", "gene_name"])
        protein2gene = dict(zip(info[col_prot], info[col_gene]))

        links = pd.read_csv(links_path, sep=" ", low_memory=False)
        col_p1 = _find_col(links, ["protein1"])
        col_p2 = _find_col(links, ["protein2"])
        col_score = _find_col(links, ["combined_score"])

        added = 0
        for _, row in links.iterrows():
            if int(row[col_score]) < min_confidence:
                continue
            g1 = protein2gene.get(row[col_p1])
            g2 = protein2gene.get(row[col_p2])
            if not g1 or not g2:
                continue
            g1, g2 = str(g1).strip(), str(g2).strip()
            if g1 in self.entity2id and g2 in self.entity2id:
                self._add_triple(g1, "interacts_with", g2)
                self._add_triple(g2, "interacts_with", g1)
                added += 2
        logger.info("  Added %d PPI triples", added)

    def load_gtex_expression(self, path: str, min_tpm: float = 1.0) -> None:
        """
        Parse GTEx gene_median_tpm.gct (tab-separated, 2 header lines).
        Columns: gene_id, Description, <tissue1>, <tissue2>, ...
        """
        logger.info("Loading GTEx expression from %s", path)
        df = pd.read_csv(path, sep="\t", skiprows=2, low_memory=False)
        col_gene = _find_col(df, ["Description", "gene_name", "Name"])
        tissue_cols = [c for c in df.columns if c not in {"Name", "Description", "gene_id"}]
        for tissue in tissue_cols:
            self._add_entity(tissue, "tissue")
        added = 0
        for _, row in df.iterrows():
            gene = str(row[col_gene]).strip()
            if gene not in self.entity2id:
                continue
            for tissue in tissue_cols:
                try:
                    tpm = float(row[tissue])
                except (ValueError, TypeError):
                    continue
                if tpm >= min_tpm:
                    self._add_triple(gene, "expressed_in", tissue)
                    self._add_triple(tissue, "expresses", gene)
                    added += 1
        logger.info("  Added %d expression triples", added)

    # ── Statistics ────────────────────────────────────────────────────────────

    def stats(self) -> Dict:
        type_counts: Dict[str, int] = defaultdict(int)
        for eid, tid in self.entity_type.items():
            type_name = [k for k, v in ENTITY_TYPES.items() if v == tid][0]
            type_counts[type_name] += 1
        rel_counts: Dict[str, int] = defaultdict(int)
        for h, r, t in self.triples:
            rel_name = [k for k, v in RELATION_TYPES.items() if v == r][0]
            rel_counts[rel_name] += 1
        return {
            "num_entities": len(self.entity2id),
            "num_triples": len(self.triples),
            "num_relations": len(self.relation2id),
            "entity_types": dict(type_counts),
            "relation_types": dict(rel_counts),
        }

    # ── Persistence ───────────────────────────────────────────────────────────

    def split_triples(
        self, val_ratio: float = 0.05, test_ratio: float = 0.05, seed: int = 42
    ) -> Tuple[List, List, List]:
        import random
        rng = random.Random(seed)
        triples = list(self.triples)
        rng.shuffle(triples)
        n = len(triples)
        n_test = int(n * test_ratio)
        n_val = int(n * val_ratio)
        test = triples[:n_test]
        val = triples[n_test: n_test + n_val]
        train = triples[n_test + n_val:]
        return train, val, test

    def save(self, output_dir: str, val_ratio: float = 0.05, test_ratio: float = 0.05) -> None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        # Save mappings as JSON
        with open(out / "entity2id.json", "w") as f:
            json.dump(self.entity2id, f)
        with open(out / "relation2id.json", "w") as f:
            json.dump(self.relation2id, f)
        with open(out / "entity_types.json", "w") as f:
            json.dump({str(k): v for k, v in self.entity_type.items()}, f)
        with open(out / "entity_names.json", "w") as f:
            json.dump({str(k): v for k, v in self.entity_names.items()}, f)

        # Save splits
        train, val, test = self.split_triples(val_ratio, test_ratio)
        for name, split in [("triples_train", train), ("triples_val", val), ("triples_test", test)]:
            t = torch.tensor(split, dtype=torch.long)
            torch.save(t, out / f"{name}.pt")

        s = self.stats()
        logger.info("KG saved to %s", output_dir)
        logger.info("  Entities : %d", s["num_entities"])
        logger.info("  Triples  : %d (train=%d val=%d test=%d)",
                    s["num_triples"], len(train), len(val), len(test))
        logger.info("  Entity types: %s", s["entity_types"])


# ── Helpers ───────────────────────────────────────────────────────────────────

def _find_col(df: pd.DataFrame, candidates: List[str], required: bool = True) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    if required:
        raise ValueError(f"None of {candidates} found in columns: {list(df.columns)}")
    return None


def load_kg_from_disk(kg_dir: str) -> Tuple[Dict, Dict, Dict, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Load a saved KG from disk. Returns (entity2id, relation2id, entity_types, train, val, test)."""
    kg_dir = Path(kg_dir)
    with open(kg_dir / "entity2id.json") as f:
        entity2id = json.load(f)
    with open(kg_dir / "relation2id.json") as f:
        relation2id = json.load(f)
    with open(kg_dir / "entity_types.json") as f:
        entity_types = {int(k): v for k, v in json.load(f).items()}
    train = torch.load(kg_dir / "triples_train.pt")
    val = torch.load(kg_dir / "triples_val.pt")
    test = torch.load(kg_dir / "triples_test.pt")
    return entity2id, relation2id, entity_types, train, val, test
