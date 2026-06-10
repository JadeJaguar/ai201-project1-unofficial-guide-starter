"""
Stretch feature 3 — Chunking Strategy Comparison.

Re-chunks the whole corpus at a few different sizes, embeds each version into a
throwaway in-memory index, and reports which chunking retrieves the right
source documents best on the 5 eval questions. This tests the claim from my
failure analysis: that chunking the Reddit listicles more finely would pull the
Q4 credit-history source into the top-5.

How "better" is judged without the LLM: each eval question has a known
grounding source file (see hybrid_compare.EVAL). For each config we check, per
question, whether a chunk from an expected source reaches the top-5 and how
close its distance is. More expected-source hits = better retrieval.

Runs fully local (MiniLM embeddings + an in-memory ChromaDB), makes ZERO Groq
API calls, and never touches the real ./chroma_db index.

Run:  python chunk_compare.py
"""

import chromadb

from embed import get_model, DEFAULT_K
from hybrid_compare import EVAL, best_rank
from ingest import (
    load_inventory, load_txt, load_pdf, extract_title, clean_text,
    make_source_name, split_text, DOCS_DIR, INVENTORY_CSV,
)

# (chunk_size, overlap) configs to compare. The middle one is the baseline that
# the shipped index uses; the others bracket it finer and coarser.
CONFIGS = [
    (400, 75),     # finer  — one fact/tip per chunk, focused embeddings
    (800, 150),    # baseline (current planning.md setting)
    (1500, 200),   # coarser — more context per chunk, blurrier embeddings
]

PROBE_DEPTH = 15


def load_clean_docs():
    """Load + clean every corpus document once (independent of chunk size)."""
    inventory = load_inventory(INVENTORY_CSV)
    docs = []
    for filename, meta in sorted(inventory.items()):
        path = DOCS_DIR / filename
        if not path.exists():
            continue
        raw = load_pdf(path) if path.suffix.lower() == ".pdf" else load_txt(path)
        title = extract_title(raw)
        docs.append({
            "cleaned": clean_text(raw),
            "source_file": filename,
            "source_name": make_source_name(meta["source_type"], meta["category"], title),
            "category": meta["category"],
        })
    return docs


def chunk_docs(docs, size, overlap):
    """Split the cleaned docs into chunk records at a given size/overlap."""
    chunks = []
    for d in docs:
        for i, piece in enumerate(split_text(d["cleaned"], size, overlap)):
            chunks.append({
                "text": piece, "source_file": d["source_file"],
                "source_name": d["source_name"], "category": d["category"],
                "chunk_index": i,
            })
    return chunks


def build_ephemeral(chunks, model, name):
    """Embed chunks locally and load them into an in-memory cosine collection.

    A fresh client per config keeps each chunking's index fully isolated (and
    sidesteps name collisions when we build several in one run).
    """
    client = chromadb.EphemeralClient()
    coll = client.create_collection(
        name=name, metadata={"hnsw:space": "cosine"},
    )
    embeddings = model.encode(
        [c["text"] for c in chunks], batch_size=64, show_progress_bar=False,
    ).tolist()
    coll.add(
        ids=[f"{c['source_file']}::{c['chunk_index']}" for c in chunks],
        embeddings=embeddings,
        documents=[c["text"] for c in chunks],
        metadatas=[{"source_file": c["source_file"]} for c in chunks],
    )
    return coll


def query(coll, model, q, k):
    res = coll.query(query_embeddings=model.encode([q]).tolist(), n_results=k)
    return [
        {"source_file": m["source_file"], "distance": d}
        for m, d in zip(res["metadatas"][0], res["distances"][0])
    ]


def main():
    print("Loading + cleaning documents once ...")
    docs = load_clean_docs()
    model = get_model()

    rows = []  # per-config summary
    for size, overlap in CONFIGS:
        chunks = chunk_docs(docs, size, overlap)
        coll = build_ephemeral(chunks, model, f"cmp_{size}_{overlap}")
        print("\n" + "=" * 78)
        print(f"CONFIG  size={size}  overlap={overlap}   ->  {len(chunks)} chunks")
        print("=" * 78)

        hits, top1_dists, q4_rank = 0, [], None
        for qi, item in enumerate(EVAL, start=1):
            deep = query(coll, model, item["q"], PROBE_DEPTH)
            rank = best_rank(deep, item["expected"])
            in_top5 = rank is not None and rank <= DEFAULT_K
            hits += int(in_top5)
            top1_dists.append(deep[0]["distance"])
            if qi == 4:
                q4_rank = rank
            tag = (f"#{rank}" + ("" if in_top5 else " (below top-5)")) if rank else "MISS"
            print(f"  Q{qi}: expected source best rank = {tag:<18} "
                  f"top-1 dist={deep[0]['distance']:.3f}")

        avg_top1 = sum(top1_dists) / len(top1_dists)
        rows.append((size, overlap, len(chunks), hits, avg_top1, q4_rank))

    # ---- summary -----------------------------------------------------------
    print("\n" + "=" * 78)
    print("SUMMARY — which chunking retrieves the expected sources best?")
    print("=" * 78)
    print(f"{'size/overlap':<16}{'chunks':>8}{'hits@5':>9}{'avg top-1 dist':>16}"
          f"{'Q4 rank':>10}")
    print("-" * 59)
    for size, overlap, n, hits, avg_top1, q4_rank in rows:
        q4 = f"#{q4_rank}" if q4_rank else "MISS"
        print(f"{f'{size}/{overlap}':<16}{n:>8}{f'{hits}/{len(EVAL)}':>9}"
              f"{avg_top1:>16.3f}{q4:>10}")
    print("-" * 59)
    print("hits@5 = how many of the 5 questions had an expected source in the "
          "top-5.\nQ4 rank = where the credit-history source landed (the "
          "documented failure).\n(Local embeddings only — no Groq API calls.)")


if __name__ == "__main__":
    main()
