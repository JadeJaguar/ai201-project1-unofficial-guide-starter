"""
Stretch feature 1 — Hybrid Search: semantic vs BM25 vs hybrid (RRF).

Compares the three retrievers on the 5 evaluation questions from planning.md.
The point is to see whether hybrid (semantic + keyword, fused with Reciprocal
Rank Fusion) rescues the documented Q4 failure, where semantic-only retrieval
refused "building credit history" even though the answer is in the Reddit docs.

This is a retrieval-only comparison. It runs fully local (MiniLM + BM25) and
makes ZERO Groq API calls — no answers are generated, we only look at which
chunks each method pulls and where the known-good source ranks.

Run:  python hybrid_compare.py
Needs the ChromaDB index built first (python embed.py).
"""

from embed import semantic_search, bm25_search, hybrid_search, DEFAULT_K

# Each eval question plus the source file(s) that actually contain the answer
# (straight from the planning.md Evaluation Plan table). We judge a retriever
# by whether it surfaces one of these expected sources, and how high.
EVAL = [
    {
        "q": "How many hours per week can an F-1 student work on campus while "
             "school is in session?",
        "expected": {"work_auth_dhs_working.txt", "maintaining_status_dartmouth.txt",
                     "work_auth_ice_sevis.txt"},
    },
    {
        "q": "What is the grace period for an F-1 student after they complete "
             "their program?",
        "expected": {"maintaining_status_dartmouth.txt", "maintaining_status_dhs.txt"},
    },
    {
        "q": "What are the requirements to qualify for the STEM OPT 24-month "
             "extension?",
        "expected": {"work_auth_dartmouth_stem_opt_about.txt",
                     "work_auth_dartmouth_stem_opt_types.txt"},
    },
    {
        "q": "What do students recommend doing to start building credit history "
             "as a new international student?",
        "expected": {"reddit_tips_incoming_students.txt.txt",
                     "reddit_leaving_us_advice.txt.txt"},
    },
    {
        "q": "What is the total number of unemployment days allowed across "
             "regular OPT and the STEM OPT extension combined?",
        "expected": {"work_auth_dartmouth_stem_opt_types.txt"},
    },
]

METHODS = {
    "semantic": semantic_search,
    "bm25": bm25_search,
    "hybrid": hybrid_search,
}

# Probe deeper than k=5 so we can report where an expected source landed even
# when it missed the top-5 (e.g. Q4's Reddit chunks at rank ~9 in the README).
PROBE_DEPTH = 15


def best_rank(results, expected):
    """1-based rank of the first chunk from an expected source, or None."""
    for rank, c in enumerate(results, start=1):
        if c["source_file"] in expected:
            return rank
    return None


def score_str(c):
    bits = []
    if c.get("distance") is not None:
        bits.append(f"d={c['distance']:.3f}")
    if c.get("bm25_score") is not None:
        bits.append(f"bm25={c['bm25_score']:.2f}")
    if c.get("rrf_score") is not None:
        bits.append(f"rrf={c['rrf_score']:.4f}")
    return " ".join(bits)


def main():
    summary = []  # (q_index, method, in_top5, best_rank)

    for qi, item in enumerate(EVAL, start=1):
        q, expected = item["q"], item["expected"]
        print("\n" + "=" * 78)
        print(f"Q{qi}: {q}")
        print(f"     expected source(s): {', '.join(sorted(expected))}")
        print("=" * 78)

        for name, fn in METHODS.items():
            deep = fn(q, k=PROBE_DEPTH)
            rank = best_rank(deep, expected)
            in_top5 = rank is not None and rank <= DEFAULT_K
            summary.append((qi, name, in_top5, rank))

            hit = "MISS"
            if rank is not None:
                hit = f"rank #{rank}" + ("  (in top-5)" if in_top5 else "  (below top-5)")
            print(f"\n  {name:>8} -> expected source: {hit}")
            for r, c in enumerate(deep[:DEFAULT_K], start=1):
                mark = " <-- expected" if c["source_file"] in expected else ""
                print(f"      [{r}] {score_str(c):<34} {c['source_file']}{mark}")

    # ---- summary table -----------------------------------------------------
    print("\n" + "=" * 78)
    print("SUMMARY — expected source in top-5? (rank of best expected source)")
    print("=" * 78)
    header = f"{'Question':<10}" + "".join(f"{m:>14}" for m in METHODS)
    print(header)
    print("-" * len(header))
    for qi in range(1, len(EVAL) + 1):
        row = f"Q{qi:<9}"
        for m in METHODS:
            rec = next(s for s in summary if s[0] == qi and s[1] == m)
            _, _, in_top5, rank = rec
            cell = ("YES" if in_top5 else "no") + (f" (#{rank})" if rank else " (--)")
            row += f"{cell:>14}"
        print(row)
    print("-" * len(header))
    totals = f"{'TOTAL':<10}"
    for m in METHODS:
        n = sum(1 for s in summary if s[1] == m and s[2])
        totals += f"{f'{n}/{len(EVAL)}':>14}"
    print(totals)
    print("\n(Local retrieval only — no Groq API calls were made.)")


if __name__ == "__main__":
    main()
