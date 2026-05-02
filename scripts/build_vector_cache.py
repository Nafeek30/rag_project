"""
One-time script: samples ~10% of Pinecone vectors, reduces to 3D with PCA,
clusters with KMeans, and saves a small cache for the vector space visualization.

Run from the project root:
    conda activate rag-project
    python scripts/build_vector_cache.py
"""

import os
import sys
import numpy as np
from dotenv import load_dotenv

load_dotenv()

from pinecone import Pinecone
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

OUT = "web_app/vector_cache.npz"
SAMPLE_EVERY = 10   # keep 1 in every 10 vectors (~10%)
BATCH = 100
N_CLUSTERS = 6


def main():
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    index = pc.Index(os.getenv("PINECONE_INDEX_NAME", "rag-knowledge"))

    # ── 1. List all IDs and sample ────────────────────────────────
    print("Listing IDs from Pinecone...")
    all_ids = []
    for page in index.list():
        all_ids.extend(page)
    print(f"  Total vectors in index: {len(all_ids)}")

    sample_ids = all_ids[::SAMPLE_EVERY]
    print(f"  Sampling every {SAMPLE_EVERY}th → {len(sample_ids)} vectors")

    # ── 2. Fetch vectors + metadata in batches ────────────────────
    print("Fetching vectors...")
    vectors, texts, sources = [], [], []

    for i in range(0, len(sample_ids), BATCH):
        batch = sample_ids[i : i + BATCH]
        result = index.fetch(ids=batch)
        for rec in result.vectors.values():
            vectors.append(rec.values)
            texts.append(rec.metadata.get("text", "")[:200])
            sources.append(rec.metadata.get("source", "Research Paper"))
        done = min(i + BATCH, len(sample_ids))
        print(f"  {done}/{len(sample_ids)}", end="\r", flush=True)

    print(f"\n  Fetched {len(vectors)} vectors")
    vectors = np.array(vectors, dtype=np.float32)

    # ── 3. PCA 768 → 3 ───────────────────────────────────────────
    print("Running PCA (768 → 3 dims)...")
    pca = PCA(n_components=3, random_state=42)
    coords = pca.fit_transform(vectors).astype(np.float32)
    explained = pca.explained_variance_ratio_.sum() * 100
    print(f"  Variance explained: {explained:.1f}%")

    # ── 4. KMeans clustering for colors ──────────────────────────
    print(f"Clustering into {N_CLUSTERS} groups...")
    km = KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init=10)
    labels = km.fit_predict(coords).astype(np.int8)

    # ── 5. Save ───────────────────────────────────────────────────
    np.savez_compressed(
        OUT,
        coords=coords,
        texts=np.array(texts),
        sources=np.array(sources),
        labels=labels,
        pca_components=pca.components_.astype(np.float32),
        pca_mean=pca.mean_.astype(np.float32),
    )
    size_kb = os.path.getsize(OUT) / 1024
    print(f"\nSaved → {OUT}  ({size_kb:.0f} KB)")


if __name__ == "__main__":
    main()
