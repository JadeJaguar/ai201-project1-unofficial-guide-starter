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

import gradio as gr

from embed import retrieve
from query import generate

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
# Travel-themed styling (pure CSS — no external image, works offline)
# ----------------------------------------------------------------------------
CSS = """
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700&display=swap');

/* Sky-to-sand gradient background with soft cloud glows */
.gradio-container, body {
    background:
        radial-gradient(900px 380px at 12% -8%, rgba(255,255,255,0.85), transparent 60%),
        radial-gradient(700px 320px at 88% 4%, rgba(255,255,255,0.55), transparent 55%),
        linear-gradient(180deg, #8fd3f4 0%, #b8e6f7 28%, #dff3fb 55%, #fdf3e3 100%)
        fixed !important;
    font-family: 'Poppins', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif !important;
    min-height: 100vh;
}

/* Centered frosted-glass card holding the whole app */
.guide-card {
    max-width: 880px !important;
    margin: 28px auto !important;
    padding: 36px 40px 40px !important;
    background: rgba(255, 255, 255, 0.78) !important;
    border: 1px solid rgba(255, 255, 255, 0.9) !important;
    border-radius: 26px !important;
    box-shadow: 0 18px 50px rgba(31, 81, 120, 0.22) !important;
    backdrop-filter: blur(10px);
}

/* Header */
.guide-header {
    text-align: center;
    margin-bottom: 8px;
}
.guide-header .title {
    font-size: 2.25rem;
    font-weight: 700;
    color: #1f5178;
    margin: 0;
    letter-spacing: -0.5px;
}
.guide-header .subtitle {
    font-size: 1rem;
    color: #4a6b85;
    margin: 8px auto 0;
    max-width: 620px;
    line-height: 1.55;
}
.guide-header .ticket-strip {
    margin: 18px auto 4px;
    width: 70%;
    border: none;
    border-top: 2px dashed #b9d4e6;
}

/* Section labels (✈️ Answer, etc.) */
.section-label {
    font-size: 0.78rem;
    font-weight: 600;
    letter-spacing: 1.4px;
    text-transform: uppercase;
    color: #2f7ab0;
    margin: 22px 0 8px 2px;
}

/* Answer / sources / passages cards */
.answer-box, .sources-box, .passages-box {
    background: #ffffff !important;
    border-radius: 16px !important;
    padding: 18px 22px !important;
    box-shadow: 0 4px 16px rgba(31, 81, 120, 0.10) !important;
    line-height: 1.65 !important;
    color: #243b4a !important;
}
.answer-box {
    border-left: 5px solid #2f7ab0 !important;
    font-size: 1.02rem;
}
.sources-box {
    border-left: 5px solid #e0a458 !important;
    font-size: 0.95rem;
}
.answer-box p, .sources-box p, .passages-box p { margin: 0.4em 0; }
.answer-box ol, .answer-box ul { margin: 0.4em 0 0.4em 1.2em; }
.answer-box li { margin: 0.25em 0; }
.passages-box blockquote {
    border-left: 3px solid #cfe2ef;
    margin: 0.5em 0;
    padding-left: 12px;
    color: #4a6b85;
}

/* Primary Ask button */
button.primary {
    background: linear-gradient(135deg, #2f7ab0, #1f5178) !important;
    border: none !important;
    font-weight: 600 !important;
    letter-spacing: 0.3px;
}
"""

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


def format_passages(chunks):
    """Render retrieved chunks as readable markdown for the details panel."""
    if not chunks:
        return "_No passages retrieved._"
    blocks = []
    for i, c in enumerate(chunks, start=1):
        text = c["text"].strip()
        blocks.append(
            f"**Passage {i}** &nbsp;·&nbsp; distance `{c['distance']:.3f}`  \n"
            f"*Source: {c['source_name']}* (`{c['source_file']}`)\n\n"
            f"> {text}"
        )
    return "\n\n---\n\n".join(blocks)


SOURCES_PLACEHOLDER = "_Sources will be listed here._"
PASSAGES_PLACEHOLDER = "_Retrieved passages will appear here._"


def handle_query(question):
    """Two-phase generator: yields interim status, then the final result.

    Each yield is (answer, sources, passages, ask_button_update). Yielding an
    interim state the moment the user clicks gives instant feedback even
    though the Groq call takes a few seconds, and disabling the button while
    we work prevents a double-click from firing a second API call.
    """
    busy_btn = gr.update(value="Searching…", interactive=False)
    idle_btn = gr.update(value="Ask ✈️", interactive=True)

    if not question or not question.strip():
        yield ("⚠️ Please enter a question.", SOURCES_PLACEHOLDER,
               PASSAGES_PLACEHOLDER, idle_btn)
        return

    # Phase 1 — retrieval (appears instantly on click).
    yield ("⏳ Searching the documents…", "_…_", "_…_", busy_btn)
    chunks = retrieve(question)

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
    # must take the question and return it unchanged — otherwise the input
    # arrives as null and every query looks empty.
    scroll_js = (
        "(question) => { setTimeout(() => { const el = "
        "document.getElementById('answer-anchor'); if (el) { "
        "el.scrollIntoView({behavior: 'smooth', block: 'start'}); } }, 100); "
        "return question; }"
    )

    outputs = [answer, sources, passages, ask_btn]
    ask_btn.click(handle_query, inputs=inp, outputs=outputs, js=scroll_js)
    inp.submit(handle_query, inputs=inp, outputs=outputs, js=scroll_js)
    clear_btn.click(
        lambda: ("", PLACEHOLDER_ANSWER, SOURCES_PLACEHOLDER, PASSAGES_PLACEHOLDER),
        outputs=[inp, answer, sources, passages],
    )


if __name__ == "__main__":
    demo.launch(theme=gr.themes.Soft(), css=CSS)
