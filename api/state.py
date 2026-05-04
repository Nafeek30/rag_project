from typing import Any, TypedDict, List
from langchain_core.documents import Document


class GraphState(TypedDict):
    """State dictionary passed between every node in the SELF-RAG graph."""
    question: str
    documents: List[Document]
    generation: str
    revision_needed: str
    model_provider: str
    attempts: int  # guards against infinite hallucination-retry loops
    sources: List[dict]  # metadata of retrieved chunks shown in UI
    from_kb: bool  # True when answer is grounded in retrieved documents
    thinking: bool  # Qwen3 only — enables chain-of-thought reasoning
    query_vector: List[float]  # embedding of the question, used for 3D visualization
    route_needs_retrieval: bool
    retrieved_docs: List[dict]
    retrieved_doc_ids: List[str]
    retrieved_scores: List[float | None]
    context_text: str
    generation_prompt_text: str
    relevance_grade: bool
    relevance_grades: List[dict]
    relevant_doc_indices: List[int]
    hallucination_grade: dict
    is_grounded: bool
    node_trace: List[dict]
    evaluation: dict[str, Any]
