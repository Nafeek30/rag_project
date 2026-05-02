from typing import TypedDict, List
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
