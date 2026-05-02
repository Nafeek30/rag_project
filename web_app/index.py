import io
import time
import numpy as np
import streamlit as st
import requests
import plotly.graph_objects as go
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt

CACHE_PATH = "web_app/vector_cache.npz"
CLUSTER_COLORS = [
    "#4FC3F7", "#81C784", "#FFB74D",
    "#F06292", "#CE93D8", "#80CBC4",
]

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
    "last_query_sources": [],
    "last_question": "",
    "last_query_vector": [],
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
# Main layout
# ---------------------------------------------------------------------------

st.title("🧠 RAG Knowledge Assistant")
st.caption("SELF-RAG · Pinecone vector search · NLP/ML research papers")

render_wordcloud(wc_placeholder)

tab_chat, tab_vector = st.tabs(["💬 Chat", "🌐 Vector Space"])

# ---------------------------------------------------------------------------
# Vector Space tab
# ---------------------------------------------------------------------------

with tab_vector:
    try:
        cache = np.load(CACHE_PATH, allow_pickle=True)
        coords  = cache["coords"]       # (N, 3)
        texts   = cache["texts"]        # (N,)
        sources = cache["sources"]      # (N,)
        labels  = cache["labels"]       # (N,)
        pca_components = cache["pca_components"]  # (3, 768)
        pca_mean       = cache["pca_mean"]        # (768,)

        # Build base scatter — one trace per cluster
        fig = go.Figure()
        for c in range(int(labels.max()) + 1):
            mask = labels == c
            fig.add_trace(go.Scatter3d(
                x=coords[mask, 0], y=coords[mask, 1], z=coords[mask, 2],
                mode="markers",
                name=f"Cluster {c + 1}",
                marker=dict(size=2.5, color=CLUSTER_COLORS[c], opacity=0.55),
                text=texts[mask],
                hovertemplate="<b>%{text}</b><extra></extra>",
            ))

        # Overlay query vector + nearest cached neighbours
        last_q = st.session_state.get("last_question", "")
        qv = st.session_state.get("last_query_vector", [])
        if qv:
            q = np.array(qv, dtype=np.float32)
            q3d = ((q - pca_mean) @ pca_components.T).reshape(1, 3)

            # 5 nearest cached points to the projected query
            dists = np.linalg.norm(coords - q3d, axis=1)
            nn_idx = np.argsort(dists)[:5]
            nn_coords = coords[nn_idx]
            nn_texts = texts[nn_idx]

            # Draw lines from query to each neighbour
            for nc, nt in zip(nn_coords, nn_texts):
                fig.add_trace(go.Scatter3d(
                    x=[q3d[0,0], nc[0]], y=[q3d[0,1], nc[1]], z=[q3d[0,2], nc[2]],
                    mode="lines",
                    line=dict(color="#FF5252", width=2),
                    showlegend=False,
                    hoverinfo="skip",
                ))

            # Nearest neighbours (red diamonds)
            fig.add_trace(go.Scatter3d(
                x=nn_coords[:, 0], y=nn_coords[:, 1], z=nn_coords[:, 2],
                mode="markers",
                name="Nearest chunks",
                marker=dict(size=7, color="#FF5252", symbol="diamond",
                            line=dict(width=1, color="white")),
                text=nn_texts,
                hovertemplate="<b>%{text}</b><extra></extra>",
            ))

            # Query point (gold star)
            fig.add_trace(go.Scatter3d(
                x=q3d[:, 0], y=q3d[:, 1], z=q3d[:, 2],
                mode="markers",
                name="Query",
                marker=dict(size=10, color="#FFD600", symbol="diamond",
                            line=dict(width=2, color="white")),
                hovertemplate=f"<b>Query:</b> {last_q[:60]}<extra></extra>",
            ))

        fig.update_layout(
            height=620,
            margin=dict(l=0, r=0, t=30, b=0),
            paper_bgcolor="#0e1117",
            scene=dict(
                bgcolor="#0e1117",
                xaxis=dict(title="PC1", showgrid=False, zeroline=False,
                           tickfont=dict(color="#555")),
                yaxis=dict(title="PC2", showgrid=False, zeroline=False,
                           tickfont=dict(color="#555")),
                zaxis=dict(title="PC3", showgrid=False, zeroline=False,
                           tickfont=dict(color="#555")),
            ),
            legend=dict(font=dict(color="white"), bgcolor="rgba(0,0,0,0)"),
        )

        col_info, col_plot = st.columns([1, 3])
        with col_info:
            st.markdown("**About this view**")
            st.caption(
                f"**{len(coords):,}** chunks sampled from {len(labels):,} total. "
                f"768-dim embeddings reduced to 3D via PCA "
                f"(20.3% variance explained). "
                f"Coloured by KMeans cluster (6 groups)."
            )
            st.divider()
            if last_q:
                st.markdown("**Last query**")
                st.info(f'"{last_q}"')
                st.caption("Showing query position 🟡 and 5 nearest cached chunks 🔴")
            else:
                st.caption("Ask a question in the Chat tab to see retrieved chunks highlighted here.")

        with col_plot:
            st.plotly_chart(fig, use_container_width=True)

    except FileNotFoundError:
        st.warning(
            "Vector cache not found. Run this once from the project root:\n\n"
            "```bash\npython scripts/build_vector_cache.py\n```"
        )

with tab_chat:
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
                    query_vector = data.get("query_vector", [])

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
                used_model, sources, from_kb, query_vector = model_choice, [], False, []
                answer_placeholder.markdown(answer)
            except requests.exceptions.Timeout:
                t = MODEL_META[model_choice]["timeout"]
                answer = f"❌ Timed out after {t}s. Check Ollama is running for Qwen3."
                used_model, sources, from_kb, query_vector = model_choice, [], False, []
                answer_placeholder.markdown(answer)
            except Exception as e:
                answer = f"❌ Error: {e}"
                used_model, sources, from_kb, query_vector = model_choice, [], False, []
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

            st.session_state.total_queries += 1
            st.session_state.model_counts[used_model] += 1
            st.session_state.response_times.append(elapsed)
            if from_kb:
                st.session_state.kb_hits += 1
            st.session_state.last_query_sources = sources
            st.session_state.last_question = prompt
            st.session_state.last_query_vector = query_vector

        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
            "model": used_model,
            "sources": sources,
            "from_kb": from_kb,
            "elapsed": elapsed,
        })

        render_wordcloud(wc_placeholder)
