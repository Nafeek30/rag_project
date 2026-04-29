import io
import time
import streamlit as st
import requests
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt

API_URL = "http://127.0.0.1:8000/ask"

MODEL_META = {
    "groq":  {"label": "🟢 Groq  — Llama 3.1 8B",   "badge": "☁️ Cloud", "timeout": 30},
    "claude": {"label": "🟣 Claude — Sonnet 4.6",      "badge": "☁️ Cloud", "timeout": 60},
    "grok":  {"label": "🔴 Grok  — xAI Grok 3 Mini",  "badge": "☁️ Cloud", "timeout": 60},
    "qwen3": {"label": "🔵 Qwen3 — 4B (offline)",      "badge": "💻 Local",  "timeout": 120},
}

PIPELINE_STEPS = ["Route", "Retrieve", "Grade", "Generate", "Validate"]

WC_STOPWORDS = STOPWORDS | {
    "what", "is", "the", "a", "an", "how", "does", "do", "can", "explain",
    "tell", "me", "about", "difference", "between", "and", "or", "in", "of",
    "for", "with", "to", "that", "it", "are", "was", "were", "be", "been",
    "have", "has", "had", "will", "would", "could", "should", "use", "used",
}

st.set_page_config(
    page_title="RAG Knowledge Assistant",
    page_icon="🧠",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Session state init
# ---------------------------------------------------------------------------

defaults = {
    "messages": [],
    "total_queries": 0,
    "kb_hits": 0,
    "model_counts": {k: 0 for k in MODEL_META},
    "response_times": [],
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("⚙️ Settings")

    model_choice = st.radio(
        "Model",
        options=list(MODEL_META.keys()),
        format_func=lambda x: MODEL_META[x]["label"],
        index=0,
    )
    st.caption(MODEL_META[model_choice]["badge"])

    thinking_on = False
    if model_choice == "qwen3":
        st.divider()
        thinking_on = st.toggle(
            "🧠 Thinking mode",
            value=False,
            help="Enables Qwen3 chain-of-thought. Slower but more thorough.",
        )
        st.caption("Chain-of-thought ON — deeper answers, ~2× slower" if thinking_on
                   else "Chain-of-thought OFF — fast direct answers")

    st.divider()
    if model_choice == "groq":
        st.info("**Groq — Llama 3.1 8B**\n\nUltra-fast LPU inference. Best for quick factual answers.")
    elif model_choice == "claude":
        st.info("**Claude — Sonnet 4.6**\n\nAnthropic's latest. Best for nuanced, well-reasoned answers.")
    elif model_choice == "grok":
        st.info("**Grok 3 Mini — xAI**\n\nStrong at logical and analytical questions.")
    else:
        st.info("**Qwen3 4B — local GPU**\n\nRuns 100% offline on your RTX 3070.")

    st.divider()
    st.markdown("**📖 Knowledge Base**")
    st.markdown(
        "- Attention & Transformers\n"
        "- RAG & retrieval techniques\n"
        "- LLM training & fine-tuning\n"
        "- Embeddings & vector search\n"
        "- NLP benchmarks & datasets"
    )

    st.divider()
    st.markdown("**How it works**")
    st.markdown(
        "1. Question routed by LLM\n"
        "2. Relevant chunks fetched from Pinecone\n"
        "3. Selected model generates answer\n"
        "4. Hallucination check validates response"
    )

    # ── Session stats ────────────────────────────────────────────────
    if st.session_state.total_queries > 0:
        st.divider()
        st.markdown("**📊 Session Stats**")
        col1, col2 = st.columns(2)
        col1.metric("Queries", st.session_state.total_queries)
        avg_t = round(
            sum(st.session_state.response_times) / len(st.session_state.response_times), 1
        ) if st.session_state.response_times else 0
        col2.metric("Avg time", f"{avg_t}s")

        kb_pct = round(st.session_state.kb_hits / st.session_state.total_queries * 100)
        st.markdown(f"**KB hit rate** — {kb_pct}%")
        st.progress(kb_pct / 100)

        used = {k: v for k, v in st.session_state.model_counts.items() if v > 0}
        if used:
            st.markdown("**Model usage**")
            st.bar_chart(used, height=120)

        if st.button("🗑️ Clear session", use_container_width=True):
            for k, v in defaults.items():
                st.session_state[k] = v if not isinstance(v, dict) else {kk: 0 for kk in v}
            st.rerun()

    st.divider()
    st.markdown("**🔤 Query Topics**")
    wc_placeholder = st.empty()

# ---------------------------------------------------------------------------
# Word cloud helper
# ---------------------------------------------------------------------------

def build_wordcloud_image(questions: list):
    text = " ".join(questions)
    wc = WordCloud(
        width=280,
        height=180,
        background_color="#0e1117",
        colormap="Blues",
        stopwords=WC_STOPWORDS,
        max_words=40,
        min_font_size=9,
        prefer_horizontal=0.8,
    ).generate(text)

    fig, ax = plt.subplots(figsize=(2.8, 1.8))
    fig.patch.set_facecolor("#0e1117")
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    plt.tight_layout(pad=0)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight", facecolor="#0e1117")
    buf.seek(0)
    plt.close(fig)
    return buf


def render_wordcloud(placeholder):
    questions = [m["content"] for m in st.session_state.messages if m["role"] == "user"]
    with placeholder.container():
        if questions:
            img = build_wordcloud_image(questions)
            st.image(img, use_container_width=True)
            st.caption(f"{len(questions)} question(s) this session")
        else:
            st.caption("Ask questions to see topic trends here")


def render_assistant_bubble(content, model, sources, from_kb, elapsed, placeholder=None):
    target = placeholder if placeholder else st
    target.markdown(content)

    if sources:
        with st.expander(f"📚 Sources — {len(sources)} chunk(s) from Pinecone"):
            for i, src in enumerate(sources, 1):
                label = src.get("source", "Research Paper")
                page = src.get("page", "")
                header = f"**Chunk {i}** — `{label}`" + (f"  ·  p.{page}" if page else "")
                st.markdown(header)
                st.caption(src.get("snippet", "")[:300])
                if i < len(sources):
                    st.divider()

    col1, col2, col3 = st.columns([3, 1, 1])
    col1.caption(f"Answered by {MODEL_META[model]['label']}")
    col2.caption("📖 KB" if from_kb else "💭 General")
    col3.caption(f"⏱ {elapsed}s")


# ---------------------------------------------------------------------------
# Main layout — chat (left) + word cloud (right)
# ---------------------------------------------------------------------------

st.title("🧠 RAG Knowledge Assistant")
st.caption("SELF-RAG · Pinecone vector search · NLP/ML research papers")

render_wordcloud(wc_placeholder)

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg["role"] == "assistant":
            render_assistant_bubble(
                msg["content"], msg["model"],
                msg.get("sources", []), msg.get("from_kb", False),
                msg.get("elapsed", 0),
            )
        else:
            st.markdown(msg["content"])

# ---------------------------------------------------------------------------
# Chat input (always full-width at the bottom)
# ---------------------------------------------------------------------------

if prompt := st.chat_input("Ask about attention, RAG, transformers, embeddings…"):
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        pipeline_placeholder = st.empty()

        def show_pipeline(active: int):
            steps_html = ""
            for i, step in enumerate(PIPELINE_STEPS):
                if i < active:
                    style = "color:#4CAF50;font-weight:bold;"
                    icon = "✅"
                elif i == active:
                    style = "color:#2196F3;font-weight:bold;"
                    icon = "⚡"
                else:
                    style = "color:#555;"
                    icon = "○"
                steps_html += f'<span style="{style}">{icon} {step}</span>'
                if i < len(PIPELINE_STEPS) - 1:
                    steps_html += ' <span style="color:#444;">→</span> '
            pipeline_placeholder.markdown(
                f'<div style="font-size:0.8em;padding:4px 0;">{steps_html}</div>',
                unsafe_allow_html=True,
            )

        show_pipeline(0)
        t_start = time.time()
        answer_placeholder = st.empty()

        try:
            with st.spinner("Running pipeline…"):
                resp = requests.post(
                    API_URL,
                    json={"question": prompt, "model": model_choice, "thinking": thinking_on},
                    timeout=MODEL_META[model_choice]["timeout"],
                )
                resp.raise_for_status()
                data = resp.json()
                answer = data.get("answer", "No answer returned.")
                used_model = data.get("model", model_choice)
                sources = data.get("sources", [])
                from_kb = data.get("from_kb", False)

            show_pipeline(4)

            words = answer.split()
            streamed = ""
            for word in words:
                streamed += word + " "
                answer_placeholder.markdown(streamed + "▌")
                time.sleep(0.025)
            answer_placeholder.markdown(answer)

        except requests.exceptions.ConnectionError:
            answer = "❌ Cannot reach the API. Run: `uvicorn api.api:app --reload`"
            used_model, sources, from_kb = model_choice, [], False
            answer_placeholder.markdown(answer)
        except requests.exceptions.Timeout:
            t = MODEL_META[model_choice]["timeout"]
            answer = f"❌ Timed out after {t}s. Check Ollama is running for Qwen3."
            used_model, sources, from_kb = model_choice, [], False
            answer_placeholder.markdown(answer)
        except Exception as e:
            answer = f"❌ Error: {e}"
            used_model, sources, from_kb = model_choice, [], False
            answer_placeholder.markdown(answer)

        elapsed = round(time.time() - t_start, 1)
        pipeline_placeholder.empty()

        if sources:
            with st.expander(f"📚 Sources — {len(sources)} chunk(s) from Pinecone"):
                for i, src in enumerate(sources, 1):
                    label = src.get("source", "Research Paper")
                    page = src.get("page", "")
                    header = f"**Chunk {i}** — `{label}`" + (f"  ·  p.{page}" if page else "")
                    st.markdown(header)
                    st.caption(src.get("snippet", "")[:300])
                    if i < len(sources):
                        st.divider()

        c1, c2, c3 = st.columns([3, 1, 1])
        c1.caption(f"Answered by {MODEL_META[used_model]['label']}")
        c2.caption("📖 KB" if from_kb else "💭 General")
        c3.caption(f"⏱ {elapsed}s")

        # ── Update session stats ──────────────────────────────────
        st.session_state.total_queries += 1
        st.session_state.model_counts[used_model] += 1
        st.session_state.response_times.append(elapsed)
        if from_kb:
            st.session_state.kb_hits += 1

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "model": used_model,
        "sources": sources,
        "from_kb": from_kb,
        "elapsed": elapsed,
    })

    # Refresh word cloud with new question included
    render_wordcloud(wc_placeholder)
