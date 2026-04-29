import streamlit as st
import requests

API_URL = "http://127.0.0.1:8000/ask"

MODEL_META = {
    "groq":  {"label": "🟢 Groq  — Llama 3.1 8B",     "badge": "☁️ Cloud"},
    "claude": {"label": "🟣 Claude — Sonnet 4.6",        "badge": "☁️ Cloud"},
    "qwen3":  {"label": "🔵 Qwen3 — 4B (offline)",       "badge": "💻 Local"},
}

st.set_page_config(
    page_title="RAG Knowledge Assistant",
    page_icon="🧠",
    layout="centered",
)

st.title("🧠 RAG Knowledge Assistant")
st.caption("SELF-RAG pipeline · Pinecone vector search · multi-model")

# ---------------------------------------------------------------------------
# Sidebar — model selector
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("⚙️ Model")

    model_choice = st.radio(
        "Select LLM",
        options=list(MODEL_META.keys()),
        format_func=lambda x: MODEL_META[x]["label"],
        index=0,
    )

    meta = MODEL_META[model_choice]
    st.caption(meta["badge"])

    st.divider()

    if model_choice == "groq":
        st.info(
            "**Groq — Llama 3.1 8B Instant**\n\n"
            "Ultra-fast cloud inference via Groq's LPU hardware. "
            "Best for quick, factual answers."
        )
    elif model_choice == "claude":
        st.info(
            "**Claude — Sonnet 4.6**\n\n"
            "Anthropic's latest Sonnet. "
            "Excels at nuanced, well-reasoned answers."
        )
    else:
        st.info(
            "**Qwen3 4B — runs on your GPU**\n\n"
            "Alibaba's state-of-the-art 4B model (April 2025). "
            "Runs 100% offline on your RTX 3070. "
            "Supports chain-of-thought thinking."
        )

    st.divider()
    st.markdown("**How it works**")
    st.markdown(
        "1. Question is routed by the LLM\n"
        "2. Relevant chunks fetched from Pinecone\n"
        "3. Selected model generates an answer\n"
        "4. Hallucination check validates the response"
    )

# ---------------------------------------------------------------------------
# Chat history
# ---------------------------------------------------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and "model" in msg:
            st.caption(f"Answered by {MODEL_META[msg['model']]['label']}")

# ---------------------------------------------------------------------------
# Input
# ---------------------------------------------------------------------------

if prompt := st.chat_input("Ask a question…"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        spinner_label = MODEL_META[model_choice]["label"].split("—")[0].strip()
        with st.spinner(f"Thinking with {spinner_label}…"):
            try:
                resp = requests.post(
                    API_URL,
                    json={"question": prompt, "model": model_choice},
                    timeout=300,
                )
                resp.raise_for_status()
                data = resp.json()
                answer = data.get("answer", "No answer returned.")
                used_model = data.get("model", model_choice)
            except requests.exceptions.ConnectionError:
                answer = (
                    "❌ Cannot reach the API. "
                    "Run: `uvicorn api.api:app --reload` in your terminal."
                )
                used_model = model_choice
            except Exception as e:
                answer = f"❌ Error: {e}"
                used_model = model_choice

        st.markdown(answer)
        st.caption(f"Answered by {MODEL_META[used_model]['label']}")

    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "model": used_model}
    )
