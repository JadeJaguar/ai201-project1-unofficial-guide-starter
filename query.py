"""
The Unofficial Guide — Milestone 5: grounded generation.

This is the end-to-end query layer (stage 5 of the diagram). ask(question)
does three things:
  1. Retrieve the top-k chunks for the question (reuses retrieve() from
     embed.py — no new embedding work, ChromaDB is already built).
  2. Build a prompt that hands the LLM ONLY those chunks as context and
     instructs it to answer from that context alone, or to refuse.
  3. Call Groq's llama-3.3-70b-versatile and return the answer plus a list
     of the source documents the context came from.

Grounding is enforced two ways:
  - The system prompt forbids using outside knowledge and pins an exact
    refusal phrase for when the context is insufficient.
  - Source attribution is built in code from the retrieved chunks, not
    trusted to the model to produce. A refusal returns no sources, so the
    system never cites documents for an answer it didn't actually give.

Run:  python query.py    # one batched test (4 API calls) — see __main__
"""

import os

from dotenv import load_dotenv
from groq import Groq

from embed import retrieve, DEFAULT_K

# ----------------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------------
load_dotenv()  # reads GROQ_API_KEY from .env

MODEL = "llama-3.3-70b-versatile"   # Groq free tier (see planning.md)
REFUSAL = "I don't have enough information on that."

SYSTEM_PROMPT = (
    "You are The Unofficial Guide, a question-answering assistant for "
    "international students on F-1 visas. You must follow these rules "
    "without exception:\n"
    "1. Answer using ONLY the information in the numbered context passages "
    "provided in the user message. Do not use any outside or prior "
    "knowledge, even if you are confident it is correct.\n"
    "2. If the context passages do not contain enough information to answer "
    f"the question, reply with exactly this sentence and nothing else: "
    f"\"{REFUSAL}\"\n"
    "3. Do not guess, infer beyond what the passages state, or fill gaps "
    "with general knowledge. If the passages give a total but not a "
    "breakdown, give the total and say the breakdown is not specified.\n"
    "4. Be concise and factual. Do not add a sources list yourself; sources "
    "are attached separately by the system."
)

_client = None


def get_client():
    """Create the Groq client once, reading the API key from the environment."""
    global _client
    if _client is None:
        _client = Groq(api_key=os.environ["GROQ_API_KEY"])
    return _client


def build_context(chunks):
    """Format retrieved chunks into numbered, source-labeled passages."""
    blocks = []
    for i, c in enumerate(chunks, start=1):
        blocks.append(
            f"[Passage {i}] (source: {c['source_name']})\n{c['text']}"
        )
    return "\n\n".join(blocks)


def generate(question, chunks):
    """Generate a grounded answer from already-retrieved chunks.

    Split out from ask() so callers (e.g. the UI) can run retrieval and
    generation as separate, visible phases. Returns a dict with answer and
    sources (see ask() for the field meanings).
    """
    context = build_context(chunks)

    user_message = (
        f"Context passages:\n\n{context}\n\n"
        f"Question: {question}\n\n"
        "Answer using only the passages above."
    )

    client = get_client()
    completion = client.chat.completions.create(
        model=MODEL,
        temperature=0,  # deterministic: same context -> same answer
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
    )
    answer = completion.choices[0].message.content.strip()

    # Source attribution is built from the retrieved chunks, not the model.
    # On a refusal we attach no sources so we never cite docs for a non-answer.
    if answer.strip().rstrip(".") == REFUSAL.rstrip("."):
        sources = []
    else:
        seen = set()
        sources = []
        for c in chunks:
            name = c["source_name"]
            if name not in seen:
                seen.add(name)
                sources.append(name)

    return {"answer": answer, "sources": sources}


def ask(question, k=DEFAULT_K):
    """Answer a question grounded in retrieved context.

    Returns a dict:
        answer  — the model's grounded answer, or the refusal sentence
        sources — list of distinct source_name values for the retrieved
                  chunks, or [] when the model refused
        chunks  — the raw retrieved chunks (text + distance + metadata), so
                  the UI can show what context the answer was grounded in
    """
    chunks = retrieve(question, k=k)
    result = generate(question, chunks)
    result["chunks"] = chunks
    return result


# ----------------------------------------------------------------------------
# Batched end-to-end test (Milestone 5 checkpoint)
# ----------------------------------------------------------------------------
# Kept deliberately small to limit API usage: 2 grounded questions, the
# designed failure case, and 1 out-of-scope question that should be refused.
TEST_QUESTIONS = [
    "How many hours per week can an F-1 student work on campus while school "
    "is in session?",
    "What are the requirements to qualify for the STEM OPT 24-month extension?",
    "How many total days of unemployment am I allowed across OPT and STEM OPT, "
    "and how is that split between OPT and the STEM extension?",
    "How much does tuition cost at Harvard for international students?",
]


if __name__ == "__main__":
    for i, q in enumerate(TEST_QUESTIONS, start=1):
        print("\n" + "=" * 78)
        print(f"Q{i}: {q}")
        print("-" * 78)
        result = ask(q)
        print("ANSWER:\n" + result["answer"])
        print("\nSOURCES:")
        if result["sources"]:
            for s in result["sources"]:
                print(f"  - {s}")
        else:
            print("  (none — refused / out of scope)")
