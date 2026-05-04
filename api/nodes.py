import hashlib
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
RETRIEVAL_K = 5

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
    _retriever = vector_store.as_retriever(search_kwargs={"k": RETRIEVAL_K})
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


def _append_trace(state: GraphState, node: str, data: dict) -> list[dict]:
    return [*state.get("node_trace", []), {"node": node, **data}]


def _safe_metadata(metadata: dict | None) -> dict:
    safe = {}
    for key, value in (metadata or {}).items():
        if isinstance(value, (str, int, float, bool)) or value is None:
            safe[key] = value
        elif isinstance(value, list):
            safe[key] = [
                item if isinstance(item, (str, int, float, bool)) or item is None else str(item)
                for item in value
            ]
        else:
            safe[key] = str(value)
    return safe


def _stable_doc_id(doc: Document, rank: int) -> str:
    metadata = doc.metadata or {}
    for key in ("id", "_id", "node_id", "document_id"):
        value = metadata.get(key)
        if value:
            return str(value)
    digest = hashlib.md5(doc.page_content.encode("utf-8")).hexdigest()[:16]
    return f"content_{digest}_r{rank}"


def _source_label(metadata: dict) -> str:
    source_files = metadata.get("source_files")
    if isinstance(source_files, list) and source_files:
        return ", ".join(str(item) for item in source_files[:2])
    return str(
        metadata.get("source")
        or metadata.get("source_file")
        or metadata.get("title")
        or "Research Paper"
    )


def _format_ranked_documents(documents: list[Document], doc_ids: list[str] | None = None) -> str:
    blocks = []
    for idx, doc in enumerate(documents, start=1):
        metadata = _safe_metadata(doc.metadata)
        doc_id = doc_ids[idx - 1] if doc_ids and idx - 1 < len(doc_ids) else _stable_doc_id(doc, idx)
        blocks.append(
            f"[Document {idx}]\n"
            f"id: {doc_id}\n"
            f"source: {_source_label(metadata)}\n"
            f"node_type: {metadata.get('node_type', '')}\n"
            f"text: {doc.page_content}"
        )
    return "\n\n".join(blocks)


def _retrieved_doc_payload(
    documents: list[Document],
    doc_ids: list[str],
    scores: list[float | None],
) -> list[dict]:
    payload = []
    for idx, doc in enumerate(documents, start=1):
        metadata = _safe_metadata(doc.metadata)
        score = scores[idx - 1] if idx - 1 < len(scores) else None
        payload.append(
            {
                "rank": idx,
                "id": doc_ids[idx - 1] if idx - 1 < len(doc_ids) else _stable_doc_id(doc, idx),
                "score": score,
                "text": doc.page_content,
                "snippet": doc.page_content[:300],
                "metadata": metadata,
                "source": _source_label(metadata),
                "node_type": metadata.get("node_type", ""),
            }
        )
    return payload


def _model_to_dict(value) -> dict:
    if hasattr(value, "model_dump"):
        return value.model_dump()
    return value.dict()


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------

def retrieve_document(state: GraphState) -> dict:
    """Retrieve relevant documents from Pinecone for the user question."""
    print("---NODE: RETRIEVE---")
    if not os.getenv("PINECONE_API_KEY"):
        return {
            "documents": [],
            "retrieved_docs": [],
            "retrieved_doc_ids": [],
            "retrieved_scores": [],
            "generation": "⚠️ Pinecone API key is not set. Add PINECONE_API_KEY to your .env file to enable retrieval.",
            "node_trace": _append_trace(
                state,
                "retrieve_document",
                {"error": "PINECONE_API_KEY is not set", "retrieved_count": 0},
            ),
        }
    retriever = _get_retriever()
    scores: list[float | None] = []
    try:
        docs_and_scores = retriever.vectorstore.similarity_search_with_score(
            state["question"],
            k=RETRIEVAL_K,
        )
        docs = [doc for doc, _score in docs_and_scores]
        scores = [float(_score) if _score is not None else None for _, _score in docs_and_scores]
    except Exception:
        docs = retriever.invoke(state["question"])
        scores = [None] * len(docs)

    print(f"    Retrieved {len(docs)} documents")
    query_vector = retriever.vectorstore.embeddings.embed_query(state["question"])
    doc_ids = [_stable_doc_id(doc, idx) for idx, doc in enumerate(docs, start=1)]
    retrieved_docs = _retrieved_doc_payload(docs, doc_ids, scores)
    return {
        "documents": docs,
        "query_vector": query_vector,
        "retrieved_docs": retrieved_docs,
        "retrieved_doc_ids": doc_ids,
        "retrieved_scores": scores,
        "node_trace": _append_trace(
            state,
            "retrieve_document",
            {
                "retrieved_count": len(docs),
                "retrieved_doc_ids": doc_ids,
                "retrieved_scores": scores,
            },
        ),
    }


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
    prompt_text = generation_prompt.format(question=question, context=context)
    response = (generation_prompt | standard_llm).invoke({"question": question, "context": context})
    text = _strip_thinking(response.content)
    attempts = state.get("attempts", 0) + 1

    sources = []
    doc_ids = state.get("retrieved_doc_ids", [])
    scores = state.get("retrieved_scores", [])
    for idx, doc in enumerate(documents, start=1):
        meta = _safe_metadata(doc.metadata)
        sources.append({
            "rank": idx,
            "id": doc_ids[idx - 1] if idx - 1 < len(doc_ids) else _stable_doc_id(doc, idx),
            "score": scores[idx - 1] if idx - 1 < len(scores) else None,
            "snippet": doc.page_content[:300],
            "source": _source_label(meta),
            "page": str(meta.get("page", "")),
            "metadata": meta,
        })

    return {
        "generation": text,
        "attempts": attempts,
        "sources": sources,
        "from_kb": bool(documents),
        "context_text": context,
        "generation_prompt_text": prompt_text,
        "node_trace": _append_trace(
            state,
            "generate_answer",
            {"attempt": attempts, "context_doc_count": len(documents), "from_kb": bool(documents)},
        ),
    }


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
    return {
        "generation": text,
        "sources": [],
        "from_kb": False,
        "node_trace": _append_trace(state, "out_of_scope_answer", {"from_kb": False}),
    }


def grade_relevance(state: GraphState) -> dict:
    """Grade whether any retrieved document is relevant to the question."""
    print("---NODE: GRADE RELEVANCE---")
    question = _thinking_prefix(state, is_json=True) + state["question"]
    documents = state["documents"]
    json_llm, _ = _get_llms(state)

    doc_ids = state.get("retrieved_doc_ids", [])
    docs_text = _format_ranked_documents(documents, doc_ids) if documents else ""
    result = (grader_prompt | json_llm | grader_parser).invoke(
        {"question": question, "documents": docs_text}
    )
    relevance_grades = [_model_to_dict(grade) for grade in result.documents]
    relevant_doc_indices = [
        grade["rank"]
        for grade in relevance_grades
        if grade.get("is_relevant") and 1 <= int(grade.get("rank", 0)) <= len(documents)
    ]
    is_relevant = bool(relevant_doc_indices)
    return {
        "revision_needed": "yes" if is_relevant else "no",
        "relevance_grade": is_relevant,
        "relevance_grades": relevance_grades,
        "relevant_doc_indices": relevant_doc_indices,
        "node_trace": _append_trace(
            state,
            "grade_relevance",
            {
                "relevance_grade": is_relevant,
                "relevant_doc_indices": relevant_doc_indices,
                "graded_doc_count": len(relevance_grades),
            },
        ),
    }


def check_hallucinations(state: GraphState) -> dict:
    """Check if the generated answer is grounded in the retrieved documents."""
    print("---NODE: CHECK HALLUCINATIONS---")
    documents = state["documents"]
    generation = state["generation"]
    json_llm, _ = _get_llms(state)

    doc_ids = state.get("retrieved_doc_ids", [])
    docs_text = _format_ranked_documents(documents, doc_ids) if documents else ""
    result = (hallucination_prompt | json_llm | hallucination_parser).invoke(
        {"documents": docs_text, "generation": generation}
    )
    hallucination_grade = _model_to_dict(result)
    return {
        "revision_needed": "no" if result.is_grounded else "yes",
        "hallucination_grade": hallucination_grade,
        "is_grounded": result.is_grounded,
        "node_trace": _append_trace(
            state,
            "check_hallucinations",
            {
                "is_grounded": result.is_grounded,
                "unsupported_claim_count": len(result.unsupported_claims),
                "contradicted_claim_count": len(result.contradicted_claims),
            },
        ),
    }


def route_question(state: GraphState) -> dict:
    """Decide whether retrieval is needed for the given question."""
    print("---NODE: ROUTE QUESTION---")
    json_llm, _ = _get_llms(state)
    question = _thinking_prefix(state, is_json=True) + state["question"]
    result = (router_prompt | json_llm | router_parser).invoke({"question": question})
    return {
        "route_needs_retrieval": result.needs_retrieval,
        "node_trace": _append_trace(
            state,
            "route_question",
            {"route_needs_retrieval": result.needs_retrieval},
        ),
    }
