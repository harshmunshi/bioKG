"""
Stage 0: Knowledge Graph Construction

Orchestrates data loading from all sources and saves the unified KG to disk.
Expected raw data layout (configurable in config.yaml):

    data/raw/mgi/
        MRK_List2.rpt              ← all mouse markers (genes)
        MGI_PhenoGenoMP.rpt        ← gene-phenotype associations
    data/raw/go/
        mgi.gaf                    ← GO annotations (GAF 2.x)
        go.obo                     ← GO ontology hierarchy
    data/raw/kegg/
        mmu_pathway_genes.txt      ← KEGG mouse pathway-gene mapping
    data/raw/string/
        10090.protein.links.v12.0.txt
        10090.protein.info.v12.0.txt
    data/raw/gtex/
        gene_median_tpm.gct        ← GTEx median TPM by tissue

All paths default to the above but are overrideable via cfg["paths"].
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Dict

logger = logging.getLogger(__name__)


def run(cfg: Dict) -> None:
    """
    Entry point for Stage 0.

    Args:
        cfg: full config dict loaded from config.yaml
    """
    from src.data.kg_builder import KGBuilder

    paths = cfg["paths"]
    raw = paths["raw_data"]
    kg_out = str(Path(paths["data_root"]) / "kg")
    kg_cfg = cfg["kg"]
    hw_cfg = cfg["hardware"]

    builder = KGBuilder(cfg=kg_cfg)

    # ── 1. MGI Genes ──────────────────────────────────────────────────────────
    mgi_genes = Path(raw["mgi"]) / "MRK_List2.rpt"
    if mgi_genes.exists():
        builder.load_mgi_genes(str(mgi_genes))
    else:
        logger.warning("MGI gene file not found: %s — skipping", mgi_genes)

    # ── 2. Gene-Phenotype Associations ────────────────────────────────────────
    mgi_pheno = Path(raw["mgi"]) / "MGI_PhenoGenoMP.rpt"
    if mgi_pheno.exists():
        builder.load_mgi_phenotypes(str(mgi_pheno))
    else:
        logger.warning("MGI phenotype file not found: %s — skipping", mgi_pheno)

    # ── 3. GO Annotations ────────────────────────────────────────────────────
    go_gaf = Path(raw["go"]) / "mgi_go.gaf"
    if go_gaf.exists():
        builder.load_go_annotations(str(go_gaf))
    else:
        logger.warning("GO GAF file not found: %s — skipping", go_gaf)

    # ── 4. GO Hierarchy ──────────────────────────────────────────────────────
    go_obo = Path(raw["go"]) / "go.obo"
    if go_obo.exists():
        builder.load_go_hierarchy(str(go_obo))
    else:
        logger.warning("GO OBO file not found: %s — skipping", go_obo)

    # ── 5. KEGG Pathways ─────────────────────────────────────────────────────
    kegg_file = Path(raw["kegg"]) / "mmu_pathway_genes.txt"
    if kegg_file.exists():
        builder.load_kegg_pathways(str(kegg_file))
    else:
        logger.warning("KEGG file not found: %s — skipping", kegg_file)

    # ── 6. STRING PPI ─────────────────────────────────────────────────────────
    string_links = Path(raw["string"]) / "10090.protein.links.v12.0.txt"
    string_info = Path(raw["string"]) / "10090.protein.info.v12.0.txt"
    if string_links.exists() and string_info.exists():
        builder.load_string_interactions(
            str(string_links),
            str(string_info),
            min_confidence=kg_cfg.get("string_confidence_threshold", 400),
        )
    else:
        logger.warning("STRING files not found — skipping PPI edges")

    # ── 7. GTEx Expression ───────────────────────────────────────────────────
    gtex_file = Path(raw["gtex"]) / "gene_median_tpm.gct"
    if gtex_file.exists():
        builder.load_gtex_expression(
            str(gtex_file),
            min_tpm=kg_cfg.get("gtex_expression_threshold", 1.0),
        )
    else:
        logger.warning("GTEx file not found: %s — skipping", gtex_file)

    # ── Print stats ──────────────────────────────────────────────────────────
    stats = builder.stats()
    logger.info("=" * 60)
    logger.info("KG CONSTRUCTION COMPLETE")
    logger.info("  Entities  : %d", stats["num_entities"])
    logger.info("  Triples   : %d", stats["num_triples"])
    logger.info("  Relations : %d", stats["num_relations"])
    logger.info("  Entity types: %s", stats["entity_types"])
    logger.info("  Relation types: %s", stats["relation_types"])
    logger.info("=" * 60)

    if stats["num_entities"] == 0:
        logger.error(
            "No entities found! Check that raw data files are present in %s", raw
        )
        raise RuntimeError("KG construction produced 0 entities — aborting.")

    # ── Save ─────────────────────────────────────────────────────────────────
    builder.save(
        kg_out,
        val_ratio=kg_cfg.get("val_split", 0.05),
        test_ratio=kg_cfg.get("test_split", 0.05),
    )
    logger.info("Stage 0 complete. KG saved to: %s", kg_out)

    # ── Also generate QA dataset ──────────────────────────────────────────────
    _generate_qa(builder, cfg)


def _generate_qa(builder, cfg: Dict) -> None:
    """Generate QA pairs from the freshly built KG."""
    import json
    import torch
    from src.data.qa_generator import QAGenerator

    paths = cfg["paths"]
    ds_cfg = cfg["dataset"]
    kg_dir = Path(paths["data_root"]) / "kg"
    qa_dir = Path(paths["data_root"]) / "qa"

    # Load triples
    triples = torch.load(str(kg_dir / "triples_train.pt"))

    generator = QAGenerator(
        entity2id=builder.entity2id,
        relation2id=builder.relation2id,
        triples=triples,
        entity_types=builder.entity_type,
        seed=cfg["hardware"].get("seed", 42),
    )

    train, val, test = generator.generate(
        n_train=ds_cfg.get("train_size", 10000),
        n_val=ds_cfg.get("val_size", 1000),
        n_test=ds_cfg.get("test_size", 500),
        weights=ds_cfg.get("template_weights"),
    )
    generator.save(str(qa_dir), train, val, test)
    logger.info("QA generation complete: %d train, %d val, %d test", len(train), len(val), len(test))
