from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Literal
from api.main import app as rag_graph

app = FastAPI(
    title="SELF-RAG API",
    description="LangGraph SELF-RAG pipeline with Claude and Groq model switching",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class QueryRequest(BaseModel):
    question: str
    model: Literal["groq", "claude", "grok", "qwen3"] = "groq"
    thinking: bool = False


@app.post("/ask")
async def ask_question(request: QueryRequest):
    """Run a question through the SELF-RAG pipeline with the chosen LLM."""
    inputs = {
        "question": request.question,
        "model_provider": request.model,
        "thinking": request.thinking,
    }
    final_generation = "No answer generated."
    sources = []
    from_kb = False

    for output in rag_graph.stream(inputs):
        for key, value in output.items():
            if "generation" in value:
                final_generation = value["generation"]
            if "sources" in value:
                sources = value["sources"]
            if "from_kb" in value:
                from_kb = value["from_kb"]

    return {"answer": final_generation, "model": request.model, "sources": sources, "from_kb": from_kb}


@app.get("/health")
async def health():
    return {"status": "ok"}
