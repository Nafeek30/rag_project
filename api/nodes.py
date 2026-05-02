import os
from dotenv import load_dotenv
from langchain_core.documents import Document

from api.state import GraphState
from api.prompts import (
    router_prompt, router_parser,
    grader_prompt, grader_parser,
    hallucination_prompt, hallucination_parser,
    generation_prompt,
)
from api.llm_config import get_llm

load_dotenv()

# ---------------------------------------------------------------------------
# Pinecone retriever — initialized lazily on first use so the server starts
# even if PINECONE_API_KEY is not yet set.
# ---------------------------------------------------------------------------

_retriever = None

def _get_retriever():
    global _retriever
    if _retriever is not None:
        return _retriever

    from langchain_pinecone import PineconeVectorStore
    from langchain_huggingface import HuggingFaceEmbeddings
    from pinecone import Pinecone

    embeddings = HuggingFaceEmbeddings(
        model_name="nomic-ai/nomic-embed-text-v1.5",
        model_kwargs={"trust_remote_code": True},
        encode_kwargs={"normalize_embeddings": True},
    )

    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    index_name = os.getenv("PINECONE_INDEX_NAME", "rag-knowledge")

    vector_store = PineconeVectorStore(
        index=pc.Index(index_name),
        embedding=embeddings,
        text_key="text",
    )
    _retriever = vector_store.as_retriever(search_kwargs={"k": 5})
    return _retriever


# ---------------------------------------------------------------------------
# Helper: build per-request LLMs from the model_provider in state
# ---------------------------------------------------------------------------

def _get_llms(state: GraphState):
    provider = state.get("model_provider", "groq")
    return get_llm(is_json=True, provider=provider), get_llm(is_json=False, provider=provider)


def _thinking_prefix(state: GraphState, is_json: bool = False) -> str:
    """Return /think or /no_think prefix for Qwen3 prompts, empty string for other models."""
    if state.get("model_provider") != "qwen3":
        return ""
    if is_json or not state.get("thinking", False):
        return "/no_think "
    return "/think "


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------

def retrieve_document(state: GraphState) -> dict:
    """Retrieve relevant documents from Pinecone for the user question."""
    print("---NODE: RETRIEVE---")
    if not os.getenv("PINECONE_API_KEY"):
        return {
            "documents": [],
            "generation": "⚠️ Pinecone API key is not set. Add PINECONE_API_KEY to your .env file to enable retrieval.",
        }
    retriever = _get_retriever()
    docs = retriever.invoke(state["question"])
    print(f"    Retrieved {len(docs)} documents")
    query_vector = retriever.vectorstore.embeddings.embed_query(state["question"])
    return {"documents": docs, "query_vector": query_vector}


def _strip_thinking(text: str) -> str:
    """Remove Qwen3 <think>...</think> chain-of-thought tokens from the response."""
    import re
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def generate_answer(state: GraphState) -> dict:
    """Generate a final answer using the selected LLM and retrieved documents."""
    print("---NODE: GENERATE ANSWER---")
    question = _thinking_prefix(state) + state["question"]
    documents = state.get("documents", [])
    _, standard_llm = _get_llms(state)

    context = "\n\n".join([doc.page_content for doc in documents])
    response = (generation_prompt | standard_llm).invoke({"question": question, "context": context})
    text = _strip_thinking(response.content)
    attempts = state.get("attempts", 0) + 1

    sources = []
    for doc in documents:
        meta = doc.metadata or {}
        sources.append({
            "snippet": doc.page_content[:300],
            "source": meta.get("source", meta.get("title", "Research Paper")),
            "page": str(meta.get("page", "")),
        })

    return {"generation": text, "attempts": attempts, "sources": sources, "from_kb": bool(documents)}


def out_of_scope_answer(state: GraphState) -> dict:
    """Return a clear message when retrieved docs are not relevant to the question."""
    print("---NODE: OUT OF SCOPE---")
    question = state["question"]
    _, standard_llm = _get_llms(state)

    response = standard_llm.invoke(
        f'The user asked: "{question}"\n\n'
        "The knowledge base (NLP/ML research papers) did not contain relevant information. "
        "Briefly tell the user this topic is not covered in the knowledge base and what topics are covered "
        "(attention mechanisms, transformers, RAG, embeddings, LLM training). "
        "If you can give a short general answer, do so and clearly label it as general knowledge."
    )
    text = _strip_thinking(response.content)
    return {"generation": text, "sources": [], "from_kb": False}


def grade_relevance(state: GraphState) -> dict:
    """Grade whether the top retrieved document is relevant to the question."""
    print("---NODE: GRADE RELEVANCE---")
    question = _thinking_prefix(state, is_json=True) + state["question"]
    documents = state["documents"]
    json_llm, _ = _get_llms(state)

    doc_text = documents[0].page_content if documents else ""
    result = (grader_prompt | json_llm | grader_parser).invoke({"question": question, "document": doc_text})
    return {"revision_needed": "yes" if result.is_relevant else "no"}


def check_hallucinations(state: GraphState) -> dict:
    """Check if the generated answer is grounded in the retrieved documents."""
    print("---NODE: CHECK HALLUCINATIONS---")
    documents = state["documents"]
    generation = state["generation"]
    json_llm, _ = _get_llms(state)

    doc_text = documents[0].page_content if documents else ""
    result = (hallucination_prompt | json_llm | hallucination_parser).invoke(
        {"document": doc_text, "generation": generation}
    )
    return {"revision_needed": "no" if result.is_grounded else "yes"}


def route_question(state: GraphState) -> str:
    """Decide whether retrieval is needed for the given question."""
    print("---EDGE: ROUTE QUESTION---")
    json_llm, _ = _get_llms(state)
    question = _thinking_prefix(state, is_json=True) + state["question"]
    result = (router_prompt | json_llm | router_parser).invoke({"question": question})
    return "retrieve_document" if result.needs_retrieval else "generate_answer"
