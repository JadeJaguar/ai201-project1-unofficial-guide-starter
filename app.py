"""
The Unofficial Guide — Milestone 5: Gradio query interface.

A travel-themed web UI over ask() from query.py. Type a question (or pick an
example), get a grounded answer, the source documents it was drawn from, and
a collapsible panel showing the exact passages that were retrieved so you can
see the answer is grounded in real context.

Run:  python app.py
Then open http://localhost:7860 in a browser.

Note: no LLM call happens until you submit a question, so launching the app
and clicking example questions (which only fill the box) costs nothing
against the Groq API.
"""

from pathlib import Path

import gradio as gr

from embed import retrieve, SOURCE_GROUPS
from query import generate

# Metadata-filtering dropdown choices. "All sources" means no filter.
ALL_SOURCES = "All sources"
SOURCE_CHOICES = [ALL_SOURCES, *SOURCE_GROUPS]

# Search-mode toggle. Hybrid (semantic + BM25) is the default because it's the
# stronger retriever; "Semantic only" reproduces the original baseline.
HYBRID_LABEL = "Hybrid (semantic + keyword)"
SEMANTIC_LABEL = "Semantic only"
METHOD_CHOICES = [HYBRID_LABEL, SEMANTIC_LABEL]

# Example questions populate the input box only — they do NOT auto-run, so
# they never spend an API call until the user actually clicks Ask.
EXAMPLE_QUESTIONS = [
    "How many hours per week can I work on campus while school is in session?",
    "What is the grace period after I complete my F-1 program?",
    "What are the requirements for the STEM OPT 24-month extension?",
    "How do international students start building credit history?",
    "How many total days of unemployment am I allowed on OPT and STEM OPT?",
]

# ----------------------------------------------------------------------------
# Travel-themed styling. The CSS lives in styles.css (alongside this file) and
# is read in here so Gradio still receives it as a string — same wiring as
# before, just kept in its own file instead of a big inline literal.
# ----------------------------------------------------------------------------
CSS = (Path(__file__).resolve().parent / "styles.css").read_text(encoding="utf-8")

HEADER_HTML = """
<div class="guide-header">
    <p class="title">✈️ The Unofficial Guide</p>
    <p class="subtitle">
        Your grounded companion for the <strong>F-1 student visa</strong> — status,
        work authorization (CPT / OPT / STEM OPT), travel &amp; re-entry, taxes, and the
        practical survival stuff. Every answer is drawn <strong>only</strong> from real
        collected documents, and tells you where it came from. 🌍
    </p>
    <hr class="ticket-strip" />
</div>
"""

PLACEHOLDER_ANSWER = "_Ask a question above and your grounded answer will land here._ 🛬"


def _score_label(c):
    """Readable score line for a chunk under any search method.

    Semantic chunks have a cosine distance; BM25-only chunks don't (distance is
    None). Hybrid chunks may carry both plus an RRF score. Show what we have.
    """
    parts = []
    if c.get("distance") is not None:
        parts.append(f"distance `{c['distance']:.3f}`")
    if c.get("bm25_score") is not None:
        parts.append(f"bm25 `{c['bm25_score']:.2f}`")
    if c.get("rrf_score") is not None:
        parts.append(f"rrf `{c['rrf_score']:.4f}`")
    return " &nbsp;·&nbsp; ".join(parts) if parts else "_no score_"


def format_passages(chunks):
    """Render retrieved chunks as readable markdown for the details panel."""
    if not chunks:
        return "_No passages retrieved._"
    blocks = []
    for i, c in enumerate(chunks, start=1):
        text = c["text"].strip()
        blocks.append(
            f"**Passage {i}** &nbsp;·&nbsp; {_score_label(c)}  \n"
            f"*Source: {c['source_name']}* (`{c['source_file']}`)\n\n"
            f"> {text}"
        )
    return "\n\n---\n\n".join(blocks)


SOURCES_PLACEHOLDER = "_Sources will be listed here._"
PASSAGES_PLACEHOLDER = "_Retrieved passages will appear here._"


def handle_query(question, source_choice, method_choice):
    """Two-phase generator: yields interim status, then the final result.

    Each yield is (answer, sources, passages, ask_button_update). Yielding an
    interim state the moment the user clicks gives instant feedback even
    though the Groq call takes a few seconds, and disabling the button while
    we work prevents a double-click from firing a second API call.

    source_choice / method_choice come from the metadata-filter dropdown and
    the search-mode toggle (stretch features). They only change retrieval, so
    they still cost zero extra API calls.
    """
    busy_btn = gr.update(value="Searching…", interactive=False)
    idle_btn = gr.update(value="Ask ✈️", interactive=True)

    if not question or not question.strip():
        yield ("⚠️ Please enter a question.", SOURCES_PLACEHOLDER,
               PASSAGES_PLACEHOLDER, idle_btn)
        return

    # Map the UI controls onto retrieve()'s arguments.
    where = None if source_choice == ALL_SOURCES else {"source_group": source_choice}
    method = "semantic" if method_choice == SEMANTIC_LABEL else "hybrid"

    # Phase 1 — retrieval (appears instantly on click).
    yield ("⏳ Searching the documents…", "_…_", "_…_", busy_btn)
    chunks = retrieve(question, where=where, method=method)

    # Phase 2 — generation.
    yield (f"🔎 Found {len(chunks)} relevant passages. ✈️ Writing a grounded "
           "answer…", "_…_", "_…_", busy_btn)
    result = generate(question, chunks)

    if result["sources"]:
        sources = "\n".join(f"- 📄 {s}" for s in result["sources"])
    else:
        sources = "_(no sources — the system did not have enough information)_"
    passages = format_passages(chunks)

    yield (result["answer"], sources, passages, idle_btn)


with gr.Blocks(title="The Unofficial Guide") as demo:
    with gr.Column(elem_classes="guide-card"):
        gr.HTML(HEADER_HTML)

        inp = gr.Textbox(
            label="Your question",
            placeholder="e.g. How many hours can I work on campus during the term?",
            autofocus=True,
            elem_classes="field q-field",
        )

        # Stretch-feature controls: limit which sources answer, and pick the
        # retriever. Both only affect retrieval, so they spend no extra API call.
        with gr.Row():
            source_dd = gr.Dropdown(
                choices=SOURCE_CHOICES,
                value=ALL_SOURCES,
                label="📂 Limit to source type",
                scale=1,
                elem_classes="field filter-field",
            )
            method_radio = gr.Radio(
                choices=METHOD_CHOICES,
                value=HYBRID_LABEL,
                label="🔀 Search mode",
                scale=1,
                elem_classes="field mode-field",
            )

        with gr.Row():
            ask_btn = gr.Button("Ask ✈️", variant="primary", scale=2)
            clear_btn = gr.Button("Clear", scale=1)

        with gr.Accordion("💡 Example questions (click to fill the box)", open=False):
            gr.Examples(
                examples=[[q] for q in EXAMPLE_QUESTIONS],
                inputs=inp,
                label="",
            )

        gr.HTML('<div class="section-label" id="answer-anchor">✈️ Answer</div>')
        answer = gr.Markdown(PLACEHOLDER_ANSWER, elem_classes="answer-box")

        gr.HTML('<div class="section-label">📚 Retrieved from</div>')
        sources = gr.Markdown(SOURCES_PLACEHOLDER, elem_classes="sources-box")

        with gr.Accordion("🔎 Show retrieved passages", open=False):
            passages = gr.Markdown(PASSAGES_PLACEHOLDER, elem_classes="passages-box")

    # Smoothly scroll down to the Answer section the moment a query fires, so
    # the user isn't left looking at the top of the page while it works.
    # NOTE: a js handler's return value replaces the Python fn's inputs, so it
    # must take ALL inputs and return them unchanged — otherwise any input it
    # drops arrives as null (the question would look empty, the filter/mode
    # would reset). We now have three inputs, so pass all three straight back.
    scroll_js = (
        "(question, source, method) => { setTimeout(() => { const el = "
        "document.getElementById('answer-anchor'); if (el) { "
        "el.scrollIntoView({behavior: 'smooth', block: 'start'}); } }, 100); "
        "return [question, source, method]; }"
    )

    inputs = [inp, source_dd, method_radio]
    outputs = [answer, sources, passages, ask_btn]
    ask_btn.click(handle_query, inputs=inputs, outputs=outputs, js=scroll_js)
    inp.submit(handle_query, inputs=inputs, outputs=outputs, js=scroll_js)
    clear_btn.click(
        lambda: ("", PLACEHOLDER_ANSWER, SOURCES_PLACEHOLDER, PASSAGES_PLACEHOLDER),
        outputs=[inp, answer, sources, passages],
    )


if __name__ == "__main__":
    demo.launch(theme=gr.themes.Soft(), css=CSS)
