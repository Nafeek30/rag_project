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
