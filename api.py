from fastapi import FastAPI
from pydantic import BaseModel
from main import app as rag_graph

app = FastAPI(title="SELF-RAG API", description="API for the LangGraph SELF-RAG pipeline")

class QueryRequest(BaseModel):
    question: str

@app.post("/ask")
async def ask_question(request: QueryRequest):
    """
    Receives a question, runs it through the SELF-RAG graph, 
    and returns the final generated answer.
    """
    inputs = {"question": request.question}
    final_generation = "No answer generated."

    # Stream the graph execution to capture the final output
    for output in rag_graph.stream(inputs):
        for key, value in output.items():
            if "generation" in value:
                final_generation = value["generation"]

    # Return the result as JSON
    return {"answer": final_generation}