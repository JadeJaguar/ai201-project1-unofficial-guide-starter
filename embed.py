"""
The Unofficial Guide — Milestone 4: embedding and retrieval.

This script does two jobs (see planning.md -> Retrieval Approach, stages 3
and 4 of the diagram):
  1. Load the chunks produced in Milestone 3 (chunks.jsonl), embed each one
     with all-MiniLM-L6-v2, and store the vectors plus all four metadata
     fields (source_file, source_name, category, chunk_index) in a local,
     persistent ChromaDB collection.
  2. Provide retrieve(query, k=5): embeds the query the same way and returns
     the top-k chunks with their distance scores and source metadata.

It does NOT re-clean or re-chunk anything. chunks.jsonl is the handoff from
Milestone 3 and is treated as read-only here.

Run:  python embed.py            # builds the index if needed, then runs a
                                 # few eval questions and prints the results
      python embed.py --rebuild  # wipe and re-embed from scratch

Storage lives in chroma_db/ (gitignored). Delete that folder, or pass
--rebuild, to force a clean re-embed.
"""

import csv
import json
import re
import sys
from pathlib import Path

import chromadb
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

# ----------------------------------------------------------------------------
# Config (mirrors planning.md -> Retrieval Approach)
# ----------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
CHUNKS_JSONL = ROOT / "chunks.jsonl"
CHROMA_DIR = ROOT / "chroma_db"
INVENTORY_CSV = ROOT / "document_inventory.csv"

MODEL_NAME = "all-MiniLM-L6-v2"   # local, 384-dim, no API key (see planning.md)
COLLECTION_NAME = "unofficial_guide"
DEFAULT_K = 5                     # top-k from planning.md; tune here if needed
METADATA_FIELDS = ("source_file", "source_name", "category", "chunk_index")

# Cosine distance so the "top results under ~0.5" yardstick in planning.md is
# meaningful. With cosine, a strong on-topic match lands around 0.2-0.4 and a
# weak/off-topic one drifts toward 0.6+.
DISTANCE_SPACE = "cosine"

# ----------------------------------------------------------------------------
# Stretch features: hybrid search + metadata filtering (see planning.md ->
# Stretch Features). All of this is local; no extra API calls.
# ----------------------------------------------------------------------------
# Coarse source buckets for the metadata-filtering dropdown. The inventory's
# raw source_type is granular (e.g. "University ISO guide (Dartmouth)"); we
# fold it into these so a user can filter "official rules" vs "student take".
SOURCE_GROUPS = (
    "Official Government", "University", "Law firm",
    "Student blog", "Reddit", "YouTube",
)

# Hybrid-search knobs. RRF fuses two rankings by rank position only, so cosine
# distance (lower=better) and BM25 score (higher=better) never have to share a
# scale. We pull POOL candidates from each ranker, then fuse.
RRF_K = 60        # standard RRF damping constant
HYBRID_POOL = 50  # how many candidates to pull from each ranker before fusing


def source_group_for(source_type):
    """Fold a granular inventory source_type into one coarse SOURCE_GROUPS bucket."""
    st = (source_type or "").lower()
    if st.startswith("official government"):
        return "Official Government"
    if st.startswith("university iso") or st.startswith("university handout"):
        return "University"
    if st.startswith("law firm"):
        return "Law firm"
    if st.startswith("student blog"):
        return "Student blog"
    if st.startswith("reddit"):
        return "Reddit"
    if st.startswith("youtube"):
        return "YouTube"
    return "Other"


def load_source_types(csv_path):
    """Return {filename: source_type} from the inventory CSV (for source_group)."""
    mapping = {}
    with open(csv_path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            mapping[row["filename"]] = row["source_type"]
    return mapping


# ----------------------------------------------------------------------------
# Shared resources (loaded once)
# ----------------------------------------------------------------------------
_model = None
_collection = None
_bm25 = None          # BM25Okapi index over all chunks (lazy)
_bm25_chunks = None   # the chunk dicts BM25 was built over, same order


def get_model():
    """Load the embedding model once and reuse it."""
    global _model
    if _model is None:
        _model = SentenceTransformer(MODEL_NAME)
    return _model


_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text):
    """Lowercase word/number tokens — the same scheme for indexing and queries."""
    return _TOKEN_RE.findall(text.lower())


def get_bm25():
    """Build (once) and return a BM25 index over chunks.jsonl plus the chunk list.

    Local and cheap: BM25 is pure term statistics, no model and no API call.
    The returned chunk list carries the same metadata as the Chroma records,
    including the source_group we add at index-build time, so BM25/hybrid
    results have the same shape as semantic ones.
    """
    global _bm25, _bm25_chunks
    if _bm25 is None:
        chunks = load_chunks(CHUNKS_JSONL)
        inv = load_source_types(INVENTORY_CSV)
        for c in chunks:
            c["source_group"] = source_group_for(inv.get(c["source_file"], ""))
        _bm25 = BM25Okapi([_tokenize(c["text"]) for c in chunks])
        _bm25_chunks = chunks
    return _bm25, _bm25_chunks


def get_collection():
    """Return the persistent ChromaDB collection, creating it if absent."""
    global _collection
    if _collection is None:
        client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        _collection = client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": DISTANCE_SPACE},
        )
    return _collection


# ----------------------------------------------------------------------------
# 1. Load chunks
# ----------------------------------------------------------------------------
def load_chunks(jsonl_path):
    """Read chunks.jsonl into a list of dicts (text + 4 metadata fields)."""
    chunks = []
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))
    return chunks


# ----------------------------------------------------------------------------
# 2. Build the index
# ----------------------------------------------------------------------------
def build_index(rebuild=False):
    """Embed all chunks and load them into ChromaDB.

    If the collection already holds the right number of chunks and rebuild is
    False, this is a no-op so re-running the script is cheap. Pass rebuild=True
    (or --rebuild on the CLI) to wipe and re-embed from scratch.
    """
    chunks = load_chunks(CHUNKS_JSONL)
    print(f"Loaded {len(chunks)} chunks from {CHUNKS_JSONL.name}")

    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    if rebuild:
        # Drop any existing collection so we start clean.
        try:
            client.delete_collection(COLLECTION_NAME)
            print("Deleted existing collection (--rebuild)")
        except Exception:
            pass

    global _collection
    _collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": DISTANCE_SPACE},
    )

    if not rebuild and _collection.count() == len(chunks):
        print(f"Collection already has {len(chunks)} chunks — skipping embed.")
        return _collection

    # ids: source_file + chunk_index is unique (chunk_index resets per file).
    ids = [f"{c['source_file']}::{c['chunk_index']}" for c in chunks]
    documents = [c["text"] for c in chunks]

    # Add the coarse source_group to each chunk's metadata so the UI can filter
    # by source type (official vs student vs Reddit, etc.). Derived from the
    # inventory's source_type column; this is why adding the filter needs one
    # local re-embed (--rebuild), but it costs no API calls.
    inv = load_source_types(INVENTORY_CSV)
    metadatas = []
    for c in chunks:
        meta = {k: c[k] for k in METADATA_FIELDS}
        meta["source_group"] = source_group_for(inv.get(c["source_file"], ""))
        metadatas.append(meta)

    print(f"Embedding {len(documents)} chunks with {MODEL_NAME} ...")
    model = get_model()
    embeddings = model.encode(
        documents, batch_size=64, show_progress_bar=True
    ).tolist()

    print("Writing vectors + metadata to ChromaDB ...")
    _collection.add(
        ids=ids,
        embeddings=embeddings,
        documents=documents,
        metadatas=metadatas,
    )
    print(f"Done. Collection now holds {_collection.count()} chunks.")
    return _collection


# ----------------------------------------------------------------------------
# 3. Retrieve
# ----------------------------------------------------------------------------
def _chroma_where(where):
    """Translate a {field: value} filter into a Chroma `where` clause (or None)."""
    if not where:
        return None
    # Single equality is the only filter the UI needs; Chroma wants $eq form.
    return {field: {"$eq": value} for field, value in where.items()}


def _matches(meta, where):
    """True if a chunk's metadata satisfies a {field: value} equality filter."""
    if not where:
        return True
    return all(meta.get(field) == value for field, value in where.items())


def semantic_search(query, k=DEFAULT_K, where=None):
    """Top-k chunks by embedding similarity (the original retrieval path).

    `where` is an optional {field: value} metadata filter, e.g.
    {"source_group": "Official Government"}, applied inside Chroma.

    Each result is a dict with:
        text      — the chunk text
        distance  — cosine distance (lower is closer; under ~0.5 is on-topic)
        source_file, source_name, category, chunk_index, source_group — metadata
    """
    model = get_model()
    collection = get_collection()
    query_embedding = model.encode([query]).tolist()

    res = collection.query(
        query_embeddings=query_embedding,
        n_results=k,
        where=_chroma_where(where),
    )

    results = []
    # Chroma returns one list per query; we only sent one query, so index [0].
    for doc, dist, meta in zip(
        res["documents"][0], res["distances"][0], res["metadatas"][0]
    ):
        results.append({"text": doc, "distance": dist, **meta})
    return results


def bm25_search(query, k=DEFAULT_K, where=None):
    """Top-k chunks by BM25 keyword score (higher score = better match).

    Local term-frequency search, no model and no API call. Good at exact
    keywords semantic search can miss (e.g. "Zolve", "Discover", "I-983").
    Results carry `bm25_score` and `distance=None` (BM25 has no distance).
    """
    bm25, chunks = get_bm25()
    scores = bm25.get_scores(_tokenize(query))
    ranked = sorted(range(len(chunks)), key=lambda i: scores[i], reverse=True)

    results = []
    for i in ranked:
        if not _matches(chunks[i], where):
            continue
        c = chunks[i]
        results.append({
            "text": c["text"], "distance": None, "bm25_score": float(scores[i]),
            "source_file": c["source_file"], "source_name": c["source_name"],
            "category": c["category"], "chunk_index": c["chunk_index"],
            "source_group": c.get("source_group", "Other"),
        })
        if len(results) >= k:
            break
    return results


def hybrid_search(query, k=DEFAULT_K, where=None, pool=HYBRID_POOL, rrf_k=RRF_K):
    """Fuse semantic and BM25 rankings with Reciprocal Rank Fusion.

    Each chunk's score is sum(1 / (rrf_k + rank)) over the two rankings, where
    rank is its 1-based position in that ranker's top `pool` (a chunk missing
    from a ranker contributes nothing for it). Using rank instead of raw score
    means cosine distance and BM25 score never have to share a scale.

    Returned chunks carry whatever each ranker knew: `distance` from semantic
    (None if only BM25 found it), `bm25_score` from BM25, and `rrf_score`.

    One guarantee on top of plain RRF: each ranker's single best hit is always
    kept. RRF scores by rank and rewards chunks that appear in BOTH rankings, so
    a chunk that one ranker nails at #1 but the other never returns gets a single
    tiny contribution (1/(rrf_k+1)) and loses to mediocre chunks present in both
    pools. That sinks exactly the case hybrid exists for — BM25 catching exact
    keywords ("Zolve", "Discover", "credit card") on a query whose casual phrasing
    the embedding maps elsewhere. Pinning each ranker's top-1 stops fusion from
    burying a landslide win in either direction.
    """
    sem = semantic_search(query, k=pool, where=where)
    kw = bm25_search(query, k=pool, where=where)

    def cid(c):  # stable id shared by both rankers
        return f"{c['source_file']}::{c['chunk_index']}"

    # Merge both rankers into one record per chunk: keep the base fields, carry
    # distance from semantic and bm25_score from BM25, and sum the RRF scores.
    fused = {}
    for ranking in (sem, kw):
        for rank, c in enumerate(ranking, start=1):
            key = cid(c)
            entry = fused.get(key)
            if entry is None:
                entry = {**c, "rrf_score": 0.0}
                fused[key] = entry
            entry["rrf_score"] += 1.0 / (rrf_k + rank)
            if c.get("distance") is not None:
                entry["distance"] = c["distance"]
            if c.get("bm25_score") is not None:
                entry["bm25_score"] = c["bm25_score"]

    ordered = sorted(fused.values(), key=lambda e: e["rrf_score"], reverse=True)

    # Pin each ranker's #1 hit (BM25 first so a keyword landslide leads), then
    # fill the remaining slots in RRF order, skipping anything already pinned.
    result, seen = [], set()
    for ranking in (kw, sem):
        if ranking:
            key = cid(ranking[0])
            if key not in seen:
                seen.add(key)
                result.append(fused[key])
    for c in ordered:
        if len(result) >= k:
            break
        if cid(c) not in seen:
            seen.add(cid(c))
            result.append(c)
    return result[:k]


def retrieve(query, k=DEFAULT_K, where=None, method="semantic"):
    """Dispatch to a search method. Default stays "semantic" so the Milestone 4-6
    behavior and the documented evaluation reproduce exactly; pass method="hybrid"
    (or "bm25") for the stretch-feature paths. `where` is an optional metadata
    filter, e.g. {"source_group": "Reddit"}.
    """
    if method == "hybrid":
        return hybrid_search(query, k=k, where=where)
    if method == "bm25":
        return bm25_search(query, k=k, where=where)
    return semantic_search(query, k=k, where=where)


# ----------------------------------------------------------------------------
# 4. Manual test against the evaluation plan
# ----------------------------------------------------------------------------
# A subset of the 5 eval questions from planning.md. Q5 is the designed
# failure case: retrieval should still surface the "150 days total" chunk;
# whether generation invents the 90+60 split is a Milestone 5/6 concern.
TEST_QUESTIONS = [
    "How many hours per week can an F-1 student work on campus while school "
    "is in session?",
    "What are the requirements to qualify for the STEM OPT 24-month extension?",
    "How many total days of unemployment am I allowed across OPT and STEM OPT?",
]


def run_eval_preview(k=DEFAULT_K):
    """Print top-k chunks and distances for each test question."""
    for i, q in enumerate(TEST_QUESTIONS, start=1):
        print("\n" + "=" * 78)
        print(f"Q{i}: {q}")
        print("=" * 78)
        for rank, r in enumerate(retrieve(q, k=k), start=1):
            preview = r["text"].replace("\n", " ").strip()
            if len(preview) > 220:
                preview = preview[:220] + " ..."
            print(f"\n  [{rank}] distance={r['distance']:.3f}  "
                  f"source={r['source_name']}  ({r['source_file']})")
            print(f"      {preview}")


if __name__ == "__main__":
    rebuild = "--rebuild" in sys.argv
    build_index(rebuild=rebuild)
    run_eval_preview()
