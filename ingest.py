"""
The Unofficial Guide — Milestone 3: document ingestion and chunking.

This script does two jobs (see planning.md, stages 1 and 2 of the diagram):
  1. Load every source document listed in document_inventory.csv
     (48 .txt files + 1 PDF). summaryOfDocs.txt is intentionally skipped
     because it is corpus metadata, not domain content.
  2. Clean each document, split it into ~800-character chunks with 150
     characters of overlap using a recursive character splitter, and tag
     every chunk with four metadata fields:
        source_file, source_name, category, chunk_index

Output: chunks.jsonl in the repo root (one JSON object per line). Milestone 4
reads this file to embed the chunks into ChromaDB.

Run:  python ingest.py
Only dependency beyond the standard library is pdfplumber (for the one PDF).
"""

import csv
import json
import re
from pathlib import Path

import pdfplumber

# ----------------------------------------------------------------------------
# Config (mirrors planning.md -> Chunking Strategy)
# ----------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
DOCS_DIR = ROOT / "documents"
INVENTORY_CSV = ROOT / "document_inventory.csv"
OUTPUT_JSONL = ROOT / "chunks.jsonl"

CHUNK_SIZE = 800       # characters (~120-150 words, roughly one paragraph)
CHUNK_OVERLAP = 150    # characters (~19%, repeats the trailing sentence)
MIN_CHUNK_CHARS = 50   # drop heading-only fragments that can't stand alone
SEPARATORS = ["\n\n", "\n", ". ", " ", ""]  # try biggest natural break first


# ----------------------------------------------------------------------------
# 1. Loading
# ----------------------------------------------------------------------------
def load_inventory(csv_path):
    """Return {filename: {category, source_type}} from the inventory CSV.

    The CSV is the source of truth for which files are part of the corpus and
    what category each belongs to. Any file not listed here (e.g.
    summaryOfDocs.txt) is excluded from ingestion by design.
    """
    inventory = {}
    with open(csv_path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            inventory[row["filename"]] = {
                "category": row["category"],
                "source_type": row["source_type"],
            }
    return inventory


def load_txt(path):
    return path.read_text(encoding="utf-8", errors="replace")


def load_pdf(path):
    """Extract text from a digitally-created PDF with pdfplumber (no OCR)."""
    with pdfplumber.open(path) as pdf:
        pages = [p.extract_text() for p in pdf.pages]
    return "\n\n".join(p for p in pages if p)


# ----------------------------------------------------------------------------
# 2. Cleaning
# ----------------------------------------------------------------------------
# Provenance header lines we strip from every doc (they aren't content).
_HEADER_PREFIXES = ("SOURCE:", "FILENAME:", "Posted by")

# Transcript / forum / Q&A artifacts. Lines matching these are dropped whole.
_DROP_LINE_PATTERNS = [
    re.compile(r"^-{3,}$"),                       # "----" separators
    re.compile(r"^u/\S+\s*\|\s*\d+\s*upvotes?", re.I),  # reddit byline
    re.compile(r"satisfied customers", re.I),     # JustAnswer boilerplate
    re.compile(r"Lawyer's Assistant", re.I),
    re.compile(r"profile photo", re.I),
    re.compile(r"customer rating", re.I),
    re.compile(r"Secure connection icon", re.I),
    re.compile(r"^Specialities include:", re.I),
    re.compile(r"^(Expert|Customer)$", re.I),
    re.compile(r"For\s+\S+\s+Only", re.I),        # "For [user] Only" footer
]

# Bracketed transcript markers: [music], [snorts], [applause], etc.
_BRACKET_MARKER = re.compile(r"\[[a-zA-Z][a-zA-Z ]{0,14}\]")

# Inline link-accessibility boilerplate fused onto words by the source page,
# e.g. "...State websiteexternal site (opens in a new window)". Removing the
# phrase leaves the real word ("website") intact.
_INLINE_NOISE = re.compile(r"(external site\s*)?\(opens in a new window\)", re.I)

# Leftover HTML entities, just in case any slipped through M1 cleaning.
_HTML_ENTITIES = {
    "&amp;": "&", "&nbsp;": " ", "&#39;": "'", "&quot;": '"',
    "&lt;": "<", "&gt;": ">",
}

# Normalize fancy Unicode punctuation to plain ASCII so chunks are uniform
# (curly quotes, en/em dashes, ellipsis, non-breaking space).
_UNICODE_PUNCT = {
    "‘": "'", "’": "'", "“": '"', "”": '"', "„": '"',
    "–": "-", "—": "-", "…": "...", " ": " ",
}


def extract_title(raw):
    """Pull a 'TITLE: ...' line if present (used for readable reddit names)."""
    m = re.search(r"^TITLE:\s*(.+)$", raw, flags=re.MULTILINE)
    return m.group(1).strip() if m else None


def clean_text(raw):
    """Strip provenance headers, artifacts, markers, and fix whitespace."""
    for entity, repl in _HTML_ENTITIES.items():
        raw = raw.replace(entity, repl)
    for fancy, repl in _UNICODE_PUNCT.items():
        raw = raw.replace(fancy, repl)

    raw = _BRACKET_MARKER.sub(" ", raw)
    raw = _INLINE_NOISE.sub("", raw)

    kept = []
    for line in raw.splitlines():
        stripped = line.strip()
        if stripped.startswith(_HEADER_PREFIXES):
            continue
        if stripped.startswith("TITLE:"):
            stripped = stripped[len("TITLE:"):].strip()  # keep title text
        if any(p.search(stripped) for p in _DROP_LINE_PATTERNS):
            continue
        kept.append(re.sub(r"[ \t]+", " ", stripped))  # collapse inner spaces

    text = "\n".join(kept)
    text = re.sub(r"\n{3,}", "\n\n", text)  # collapse runs of blank lines
    return text.strip()


def make_source_name(source_type, category, title):
    """Readable citation label, e.g. 'University ISO guide (Dartmouth), Maintaining Status'."""
    if title:  # reddit threads carry a descriptive title
        return f"{source_type}: {title}"
    readable = category.replace("_", " ").title()
    return f"{source_type}, {readable}"


# ----------------------------------------------------------------------------
# 3. Chunking — recursive character splitter (LangChain-style algorithm)
# ----------------------------------------------------------------------------
def _merge_splits(splits, separator, chunk_size, overlap):
    """Greedily merge small splits into chunks <= chunk_size, carrying overlap."""
    sep_len = len(separator)
    chunks, current, total = [], [], 0
    for piece in splits:
        addition = len(piece) + (sep_len if current else 0)
        if total + addition > chunk_size and current:
            chunk = separator.join(current).strip()
            if chunk:
                chunks.append(chunk)
            # Drop from the front until the carried-over tail fits the overlap.
            while total > overlap and current:
                total -= len(current[0]) + (sep_len if len(current) > 1 else 0)
                current = current[1:]
        current.append(piece)
        total += len(piece) + (sep_len if len(current) > 1 else 0)
    chunk = separator.join(current).strip()
    if chunk:
        chunks.append(chunk)
    return chunks


def _recursive_split(text, separators, chunk_size, overlap):
    """Split using the first usable separator, recursing into oversized pieces."""
    separator = separators[-1]
    remaining = []
    for i, sep in enumerate(separators):
        if sep == "" or sep in text:
            separator = sep
            remaining = separators[i + 1:]
            break

    splits = text.split(separator) if separator else list(text)
    splits = [s for s in splits if s != ""]

    final, good = [], []
    for piece in splits:
        if len(piece) < chunk_size:
            good.append(piece)
            continue
        if good:
            final.extend(_merge_splits(good, separator, chunk_size, overlap))
            good = []
        if remaining:
            final.extend(_recursive_split(piece, remaining, chunk_size, overlap))
        else:
            final.append(piece)  # nothing left to split on; keep as-is
    if good:
        final.extend(_merge_splits(good, separator, chunk_size, overlap))
    return final


def split_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    chunks = _recursive_split(text, SEPARATORS, chunk_size, overlap)
    # Drop empties and heading-only fragments that can't answer a query alone.
    return [c for c in chunks if len(c.strip()) >= MIN_CHUNK_CHARS]


# ----------------------------------------------------------------------------
# 4. Pipeline
# ----------------------------------------------------------------------------
def build_chunks():
    inventory = load_inventory(INVENTORY_CSV)
    all_chunks = []
    per_file = {}

    for filename, meta in sorted(inventory.items()):
        path = DOCS_DIR / filename
        if not path.exists():
            print(f"  WARNING: listed in inventory but missing on disk: {filename}")
            continue

        raw = load_pdf(path) if path.suffix.lower() == ".pdf" else load_txt(path)
        title = extract_title(raw)
        cleaned = clean_text(raw)
        source_name = make_source_name(meta["source_type"], meta["category"], title)

        pieces = split_text(cleaned)
        per_file[filename] = len(pieces)
        for i, piece in enumerate(pieces):
            all_chunks.append({
                "text": piece,
                "source_file": filename,
                "source_name": source_name,
                "category": meta["category"],
                "chunk_index": i,
            })

    return all_chunks, per_file


def write_jsonl(chunks, path):
    with open(path, "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")


# ----------------------------------------------------------------------------
# 5. Verification report
# ----------------------------------------------------------------------------
def report(chunks, per_file):
    total = len(chunks)
    print("\n" + "=" * 70)
    print(f"TOTAL CHUNKS: {total}  (from {len(per_file)} documents)")
    print("=" * 70)

    lengths = [len(c["text"]) for c in chunks]
    print(f"Chunk length (chars):  min={min(lengths)}  "
          f"avg={sum(lengths)//total}  max={max(lengths)}")

    by_cat = {}
    for c in chunks:
        by_cat[c["category"]] = by_cat.get(c["category"], 0) + 1
    print("\nChunks per category:")
    for cat in sorted(by_cat):
        print(f"  {cat:<22} {by_cat[cat]}")

    # Spot-check the two messiest source files were actually cleaned.
    print("\nSanity check on noisiest files (chunk counts):")
    for f in ("youtube_how_to_apply_f1_j1.txt.txt",
              "youtube_visa_interview_2026.txt.txt",
              "travel_justanswer.txt.txt"):
        print(f"  {f:<40} {per_file.get(f, 'MISSING')} chunks")

    # 5 sample chunks spread across the corpus.
    print("\n" + "=" * 70)
    print("5 SAMPLE CHUNKS")
    print("=" * 70)
    step = max(1, total // 5)
    for c in [chunks[i] for i in range(0, total, step)][:5]:
        print(f"\n[{c['source_name']}]  ({c['source_file']}, chunk {c['chunk_index']}, "
              f"{len(c['text'])} chars)")
        print("-" * 70)
        print(c["text"])


def main():
    print("Loading and chunking documents from", DOCS_DIR, "...")
    chunks, per_file = build_chunks()
    write_jsonl(chunks, OUTPUT_JSONL)
    report(chunks, per_file)
    print(f"\nWrote {len(chunks)} chunks to {OUTPUT_JSONL.name}")


if __name__ == "__main__":
    main()
