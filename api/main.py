from langgraph.graph import StateGraph, START, END
from api.state import GraphState
from api.nodes import (
    retrieve_document,
    generate_answer,
    grade_relevance,
    check_hallucinations,
    route_question,
    out_of_scope_answer,
)


# ---------------------------------------------------------------------------
# Conditional edge functions
# ---------------------------------------------------------------------------

def relevance_edge(state: GraphState) -> str:
    # Short-circuit if retrieval node already set an error message
    if state.get("generation"):
        return END
    if state.get("revision_needed") == "yes":
        return "generate_answer"
    print("---EDGE: DOCUMENT NOT RELEVANT. ROUTING TO FALLBACK.---")
    return "out_of_scope_answer"


def post_generation_edge(state: GraphState) -> str:
    if state.get("documents"):
        return "check_hallucinations"
    print("---EDGE: NO DOCUMENTS. SKIPPING HALLUCINATION CHECK.---")
    return END


def hallucination_edge(state: GraphState) -> str:
    if state.get("revision_needed") == "no":
        return END
    if state.get("attempts", 0) >= 2:
        print("---EDGE: MAX RETRIES REACHED. RETURNING BEST ANSWER.---")
        return END
    print("---EDGE: HALLUCINATION DETECTED. RE-GENERATING.---")
    return "generate_answer"


# ---------------------------------------------------------------------------
# Build and compile the graph
# ---------------------------------------------------------------------------

workflow = StateGraph(GraphState)

workflow.add_node("retrieve_document", retrieve_document)
workflow.add_node("grade_relevance", grade_relevance)
workflow.add_node("generate_answer", generate_answer)
workflow.add_node("check_hallucinations", check_hallucinations)
workflow.add_node("out_of_scope_answer", out_of_scope_answer)

workflow.add_conditional_edges(START, route_question)
workflow.add_edge("retrieve_document", "grade_relevance")
workflow.add_conditional_edges("grade_relevance", relevance_edge)
workflow.add_conditional_edges("generate_answer", post_generation_edge)
workflow.add_conditional_edges("check_hallucinations", hallucination_edge)

app = workflow.compile()
