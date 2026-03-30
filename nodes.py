from state import GraphState
from langchain_core.documents import Document

def retrieve_mock_document(state: GraphState) -> dict:
    """
    A 'dumb' retriever node that ignores the actual user question 
    and returns a hardcoded paragraph about Strep Throat.
    """
    print("---NODE: MOCK RETRIEVER---")
    
    # I purposefully ignore state["question"] for this demo
    
    mock_text = (
        "Strep throat is a bacterial infection that can make your throat feel sore and scratchy. "
        "Common symptoms include throat pain, swollen lymph nodes, and red tonsils. "
        "It is typically treated with a course of oral antibiotics like amoxicillin."
    )
    
    doc = Document(page_content=mock_text, metadata={"source": "mock_medical_db"})
    
    return {"documents": [doc]}