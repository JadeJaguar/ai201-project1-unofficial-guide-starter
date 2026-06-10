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

import json
import sys
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

# ----------------------------------------------------------------------------
# Config (mirrors planning.md -> Retrieval Approach)
# ----------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
CHUNKS_JSONL = ROOT / "chunks.jsonl"
CHROMA_DIR = ROOT / "chroma_db"

MODEL_NAME = "all-MiniLM-L6-v2"   # local, 384-dim, no API key (see planning.md)
COLLECTION_NAME = "unofficial_guide"
DEFAULT_K = 5                     # top-k from planning.md; tune here if needed
METADATA_FIELDS = ("source_file", "source_name", "category", "chunk_index")

# Cosine distance so the "top results under ~0.5" yardstick in planning.md is
# meaningful. With cosine, a strong on-topic match lands around 0.2-0.4 and a
# weak/off-topic one drifts toward 0.6+.
DISTANCE_SPACE = "cosine"


# ----------------------------------------------------------------------------
# Shared resources (loaded once)
# ----------------------------------------------------------------------------
_model = None
_collection = None


def get_model():
    """Load the embedding model once and reuse it."""
    global _model
    if _model is None:
        _model = SentenceTransformer(MODEL_NAME)
    return _model


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
    metadatas = [{k: c[k] for k in METADATA_FIELDS} for c in chunks]

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
def retrieve(query, k=DEFAULT_K):
    """Return the top-k chunks for a query.

    Each result is a dict with:
        text      — the chunk text
        distance  — cosine distance (lower is closer; under ~0.5 is on-topic)
        source_file, source_name, category, chunk_index — metadata
    """
    model = get_model()
    collection = get_collection()
    query_embedding = model.encode([query]).tolist()

    res = collection.query(
        query_embeddings=query_embedding,
        n_results=k,
    )

    results = []
    # Chroma returns one list per query; we only sent one query, so index [0].
    for doc, dist, meta in zip(
        res["documents"][0], res["distances"][0], res["metadatas"][0]
    ):
        results.append({"text": doc, "distance": dist, **meta})
    return results


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
