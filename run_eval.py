"""
The Unofficial Guide — Milestone 6: full evaluation run.

Runs the 5 evaluation questions from planning.md verbatim, plus failure-case
candidates that target the multi-column PDF (Employment-Options-Handout-Merged.pdf),
where pdfplumber read straight across the two-column CPT/OPT table and interleaved
the columns during ingestion.

For each question it records the answer, the programmatic source list, and every
retrieved chunk with its cosine distance and source — so the README evaluation
table and failure-case analysis are backed by captured output, not memory.

Run:  python run_eval.py            # prints to stdout AND writes eval_output.txt
This spends one Groq API call per question (see API budget note in README).
"""

import io
import sys

from query import ask

# The 5 evaluation questions, copied verbatim from planning.md -> Evaluation Plan.
PLANNING_QUESTIONS = [
    ("Q1", "How many hours per week can an F-1 student work on campus while "
           "school is in session?"),
    ("Q2", "What is the grace period for an F-1 student after they complete "
           "their program?"),
    ("Q3", "What are the requirements to qualify for the STEM OPT 24-month "
           "extension?"),
    ("Q4", "What do students recommend doing to start building credit history "
           "as a new international student?"),
    ("Q5", "What is the total number of unemployment days allowed across "
           "regular OPT and the STEM OPT extension combined, and how is that "
           "split between regular OPT and the STEM extension?"),
]

# Failure-case candidates: questions whose answers live in the scrambled PDF
# table. The goal is to surface the column-interleaving ingestion failure.
FAILURE_QUESTIONS = [
    ("F1", "How many hours per week can a J-1 student work on campus, and what "
           "is academic training (AT) and its overall time limit?"),
    ("F2", "According to the university employment options handout, what are "
           "the limitations on using CPT, and can using full-time CPT affect "
           "my OPT eligibility?"),
]


def run_block(title, questions, out):
    print("\n" + "#" * 78, file=out)
    print(f"# {title}", file=out)
    print("#" * 78, file=out)
    for tag, q in questions:
        print("\n" + "=" * 78, file=out)
        print(f"{tag}: {q}", file=out)
        print("-" * 78, file=out)
        result = ask(q)
        print("ANSWER:\n" + result["answer"], file=out)
        print("\nSOURCES:", file=out)
        if result["sources"]:
            for s in result["sources"]:
                print(f"  - {s}", file=out)
        else:
            print("  (none — refused / out of scope)", file=out)
        print("\nRETRIEVED CHUNKS (text trimmed):", file=out)
        for rank, c in enumerate(result["chunks"], start=1):
            preview = c["text"].replace("\n", " ").strip()
            if len(preview) > 260:
                preview = preview[:260] + " ..."
            print(f"  [{rank}] dist={c['distance']:.3f}  src={c['source_name']} "
                  f"({c['source_file']})", file=out)
            print(f"      {preview}", file=out)


def main():
    buf = io.StringIO()
    run_block("PLANNING.MD EVALUATION QUESTIONS (Q1-Q5)", PLANNING_QUESTIONS, buf)
    run_block("FAILURE-CASE CANDIDATES (scrambled PDF table)", FAILURE_QUESTIONS, buf)

    text = buf.getvalue()
    sys.stdout.write(text)
    with open("eval_output.txt", "w", encoding="utf-8") as f:
        f.write(text)
    print("\n\n[written to eval_output.txt]")


if __name__ == "__main__":
    main()
