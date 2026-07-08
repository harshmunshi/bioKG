"""
Sanity + quality checks on Stage 1 (RotatE) outputs: entity_embeddings.pt /
relation_embeddings.pt in data/kg/.

Run with:
    pytest tests/test_stage1_outputs.py -v -s
"""
import numpy as np
import torch
import torch.nn.functional as F

K_NEAREST = 5


# ── TEST 1: Embedding integrity ──────────────────────────────────────────────

def test_embedding_shape_and_coverage(entity_embeddings, entity2id):
    print("\n" + "=" * 60)
    print("TEST 1: Embedding Shape & Coverage")
    print("=" * 60)

    num_entities = len(entity2id)
    assert entity_embeddings.shape[0] == num_entities, (
        f"Embedding table has {entity_embeddings.shape[0]} rows, "
        f"expected {num_entities} (one per entity2id entry)"
    )
    assert entity_embeddings.shape[1] % 2 == 0, "RotatE embedding_dim must be even"

    print(f"  Entities: {num_entities}")
    print(f"  Embedding dim: {entity_embeddings.shape[1]}")
    print("\n✓ Embedding table shape matches entity2id")


def test_no_nan_or_inf(entity_embeddings):
    print("\n" + "=" * 60)
    print("TEST 2: No NaN / Inf in Embeddings")
    print("=" * 60)

    n_nan = torch.isnan(entity_embeddings).sum().item()
    n_inf = torch.isinf(entity_embeddings).sum().item()
    print(f"  NaN count: {n_nan}")
    print(f"  Inf count: {n_inf}")

    assert n_nan == 0, f"Found {n_nan} NaN values in entity embeddings"
    assert n_inf == 0, f"Found {n_inf} Inf values in entity embeddings"

    print("\n✓ No NaN/Inf values found")


# ── TEST 3: Nearest neighbor quality ─────────────────────────────────────────

def test_nearest_neighbors(entity_embeddings, entity2id, id2entity):
    """
    Test k-nearest neighbors for biological coherence.
    """
    print("\n" + "=" * 60)
    print("TEST 3: Nearest Neighbor Quality")
    print("=" * 60)

    k = K_NEAREST

    # Test queries
    test_queries = [
        "Thbd",       # Coagulation gene
        "Bmp4",       # Developmental gene
        "Kidney",     # Tissue (may not exist in this KG — skipped gracefully)
        "MP:0003350",  # Phenotype (renal infarct)
    ]

    results = {}

    for query in test_queries:
        if query not in entity2id:
            print(f"\nQuery: {query}  (skipped — not in entity2id)")
            continue

        # Get query embedding
        query_emb = entity_embeddings[entity2id[query]]

        # Compute distances to all entities
        distances = torch.cdist(
            query_emb.unsqueeze(0),
            entity_embeddings,
        ).squeeze()

        # Get top-k nearest (excluding self)
        top_k_indices = torch.topk(
            distances,
            k=k + 1,
            largest=False,
        ).indices[1:]  # Skip first (self)

        neighbors = [id2entity[idx.item()] for idx in top_k_indices]
        neighbor_dists = [distances[idx].item() for idx in top_k_indices]

        print(f"\nQuery: {query}")
        print(f"  Top {k} nearest neighbors:")
        for i, (neighbor, dist) in enumerate(zip(neighbors, neighbor_dists), 1):
            print(f"    {i}. {neighbor} (dist: {dist:.3f})")

        results[query] = {
            "neighbors": neighbors,
            "distances": neighbor_dists,
        }

    assert results, "No test queries were found in entity2id — cannot validate neighbors"

    print(f"\n✓ Nearest neighbors computed for {len(results)}/{len(test_queries)} queries")
    print("  (Manual verification of biological coherence recommended)")


# ── TEST 4: Semantic similarity validation ───────────────────────────────────

def test_semantic_similarity(entity_embeddings, entity2id):
    """
    Validate embeddings using known biological similarity.
    """
    print("\n" + "=" * 60)
    print("TEST 4: Semantic Similarity Validation")
    print("=" * 60)

    # Positive pairs (should be similar)
    similar_pairs = [
        ("Thbd", "Proc", "Both in coagulation cascade"),
        ("Bmp2", "Bmp4", "Same gene family (BMPs)"),
        ("Bmp4", "Bmp7", "Same gene family"),
        ("Kidney", "Nephron", "Tissue-substructure hierarchy"),
        ("Liver", "Hepatocyte", "Tissue-cell type"),
    ]

    # Negative pairs (should be dissimilar)
    dissimilar_pairs = [
        ("Thbd", "Bmp4", "Unrelated genes"),
        ("Kidney", "Glucose", "Tissue vs metabolite"),
        ("Liver", "MP:0003350", "Tissue vs phenotype"),
    ]

    similar_scores = []
    dissimilar_scores = []

    for e1, e2, reason in similar_pairs:
        if e1 not in entity2id or e2 not in entity2id:
            print(f"  (skipped {e1} <-> {e2} — not in entity2id)")
            continue
        sim = F.cosine_similarity(
            entity_embeddings[entity2id[e1]].unsqueeze(0),
            entity_embeddings[entity2id[e2]].unsqueeze(0),
        ).item()
        similar_scores.append(sim)
        print(f"✓ Similar: {e1} <-> {e2} = {sim:.3f} ({reason})")

    print()

    for e1, e2, reason in dissimilar_pairs:
        if e1 not in entity2id or e2 not in entity2id:
            print(f"  (skipped {e1} <-> {e2} — not in entity2id)")
            continue
        sim = F.cosine_similarity(
            entity_embeddings[entity2id[e1]].unsqueeze(0),
            entity_embeddings[entity2id[e2]].unsqueeze(0),
        ).item()
        dissimilar_scores.append(sim)
        print(f"X Dissimilar: {e1} <-> {e2} = {sim:.3f} ({reason})")

    assert similar_scores, "No similar pairs found in entity2id — cannot validate"
    assert dissimilar_scores, "No dissimilar pairs found in entity2id — cannot validate"

    avg_similar = float(np.mean(similar_scores))
    avg_dissimilar = float(np.mean(dissimilar_scores))

    print("\nSummary:")
    print(f"  Avg similar pairs: {avg_similar:.3f}")
    print(f"  Avg dissimilar pairs: {avg_dissimilar:.3f}")
    print(f"  Separation: {avg_similar - avg_dissimilar:.3f}")

    # Should have clear separation
    assert avg_similar > avg_dissimilar + 0.1, (
        f"Insufficient semantic separation: similar={avg_similar:.3f}, dissimilar={avg_dissimilar:.3f}"
    )

    print(f"\n✓ Semantic similarity validated (separation: {avg_similar - avg_dissimilar:.3f})")
