from state import GraphState
from langchain_core.documents import Document
from langchain_ollama import ChatOllama
from prompts import (
    router_prompt,
    router_parser,
    grader_prompt,
    grader_parser,
    hallucination_prompt,
    hallucination_parser,
    generation_prompt,
)

json_llm = ChatOllama(model="llama3", format="json", temperature=0)
standard_llm = ChatOllama(model="llama3", temperature=0)

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


def generate_answer(state: GraphState) -> dict:
    """Generates the final answer using the user's question and retrieved documents."""
    print("---NODE: GENERATE ANSWER---")
    question = state["question"]
    documents = state.get("documents", [])
    
    # Extract text from documents if they exist
    context = "\n".join([doc.page_content for doc in documents])
    
    # Build the chain and generate
    generation_chain = generation_prompt | standard_llm
    response = generation_chain.invoke({"question": question, "context": context})
    
    return {"generation": response.content}


def grade_relevance(state: GraphState) -> dict:
    """Grades whether the retrieved document is relevant to the question."""
    print("---NODE: GRADE RELEVANCE---")
    question = state["question"]
    documents = state["documents"]
    
    # I only have one mock document, so I just grade the first one
    doc_text = documents[0].page_content
    
    # Build the chain and parse the JSON response
    grader_chain = grader_prompt | json_llm | grader_parser
    result = grader_chain.invoke({"question": question, "document": doc_text})
    
    # Return 'yes' or 'no' for routing later
    status = "no" if not result.is_relevant else "yes"
    return {"revision_needed": status}


def check_hallucinations(state: GraphState) -> dict:
    """Checks if the generated answer is grounded in the retrieved document."""
    print("---NODE: CHECK HALLUCINATIONS---")
    documents = state["documents"]
    generation = state["generation"]
    
    doc_text = documents[0].page_content
    
    hallucination_chain = hallucination_prompt | json_llm | hallucination_parser
    result = hallucination_chain.invoke({"document": doc_text, "generation": generation})
    
    status = "no" if result.is_grounded else "yes"
    return {"revision_needed": status}


def route_question(state: GraphState) -> str:
    """Route function to decide if retrieval is needed."""
    print("---EDGE: ROUTE QUESTION---")
    question = state["question"]
    
    router_chain = router_prompt | json_llm | router_parser
    result = router_chain.invoke({"question": question})
    
    if result.needs_retrieval:
        return "retrieve_mock_document"
    return "generate_answer"