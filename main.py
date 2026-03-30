from langgraph.graph import StateGraph, START, END
from state import GraphState
from nodes import (
    retrieve_mock_document,
    generate_answer,
    grade_relevance,
    check_hallucinations,
    route_question,
)


# --- CONDITIONAL EDGE FUNCTIONS ---

def relevance_edge(state: GraphState) -> str:
    """Reads the relevance flag to determine the next step."""
    if state.get("revision_needed") == "yes":
        return "generate_answer"

    # If the document fails relevance, end the graph to avoid an infinite loop
    print("---EDGE: DOCUMENT NOT RELEVANT. ENDING.---")
    return END


def post_generation_edge(state: GraphState) -> str:
    """Only run the hallucination checker if a document was retrieved."""
    if state.get("documents"):
        return "check_hallucinations"

    print("---EDGE: NO DOCUMENTS RETRIEVED. SKIPPING HALLUCINATION CHECK.---")
    return END


def hallucination_edge(state: GraphState) -> str:
    """Reads the hallucination flag to determine the next step."""
    if state.get("revision_needed") == "no":
        return END

    # If it hallucinated, loop back and tell it to try generating again
    print("---EDGE: HALLUCINATION DETECTED. RE-GENERATING.---")
    return "generate_answer"


# --- BUILD THE GRAPH ---

workflow = StateGraph(GraphState)

workflow.add_node("retrieve_mock_document", retrieve_mock_document)
workflow.add_node("grade_relevance", grade_relevance)
workflow.add_node("generate_answer", generate_answer)
workflow.add_node("check_hallucinations", check_hallucinations)

# Start -> Router
workflow.add_conditional_edges(START, route_question)

# Retriever -> Relevance Grader
workflow.add_edge("retrieve_mock_document", "grade_relevance")

# Relevance Grader -> (Generate OR End)
workflow.add_conditional_edges("grade_relevance", relevance_edge)

# Generation -> (Hallucination Checker OR End)
workflow.add_conditional_edges("generate_answer", post_generation_edge)

# Hallucination Checker -> (End OR Generate)
workflow.add_conditional_edges("check_hallucinations", hallucination_edge)

app = workflow.compile()


# --- EXECUTE THE DEMO ---

if __name__ == "__main__":
    inputs = {"question": "What is the recommended antibiotic for strep throat?"}
    
    print(f"User Query: {inputs['question']}\n")
    print("-" * 40)

    final_generation = "No answer generated."
    
    # Run the graph and stream the outputs step-by-step
    for output in app.stream(inputs):
        for key, value in output.items():
            print(f"Node Executed: {key}")
            print("-" * 40)

            if "generation" in value:
                final_generation = value["generation"]
            
    # Extract and print the final generated answer
    print("\nFINAL ANSWER:")
    print(final_generation)