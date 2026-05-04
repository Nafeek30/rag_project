from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Any, Literal
from api.main import app as rag_graph

app = FastAPI(
    title="SELF-RAG API",
    description="LangGraph SELF-RAG pipeline with multi-model switching",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class QueryRequest(BaseModel):
    question: str
    model: Literal["groq", "claude", "grok", "qwen3", "openai"] = "groq"
    thinking: bool = False


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_jsonable(v) for v in value]
    if isinstance(value, tuple):
        return [_jsonable(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if hasattr(value, "model_dump"):
        return _jsonable(value.model_dump())
    if hasattr(value, "dict"):
        return _jsonable(value.dict())
    return str(value)


def _run_graph(request: QueryRequest) -> tuple[dict, dict]:
    inputs = {
        "question": request.question,
        "model_provider": request.model,
        "thinking": request.thinking,
    }
    final_generation = "No answer generated."
    sources = []
    from_kb = False
    query_vector = []
    trace = {
        "question": request.question,
        "model": request.model,
        "thinking": request.thinking,
        "node_outputs": [],
    }

    for output in rag_graph.stream(inputs):
        for node_name, value in output.items():
            serializable_update = {
                key: _jsonable(val)
                for key, val in value.items()
                if key != "documents"
            }
            trace["node_outputs"].append(
                {"node": node_name, "updated_keys": list(serializable_update.keys())}
            )
            trace.update(serializable_update)
            if "generation" in value:
                final_generation = value["generation"]
            if "sources" in value:
                sources = value["sources"]
            if "from_kb" in value:
                from_kb = value["from_kb"]
            if "query_vector" in value:
                query_vector = value["query_vector"]

    public_response = {
        "answer": final_generation,
        "model": request.model,
        "sources": _jsonable(sources),
        "from_kb": from_kb,
        "query_vector": _jsonable(query_vector),
    }
    return public_response, trace


@app.post("/ask")
async def ask_question(request: QueryRequest):
    """Run a question through the SELF-RAG pipeline with the chosen LLM."""
    public_response, _ = _run_graph(request)
    return public_response


@app.post("/ask_debug")
async def ask_question_debug(request: QueryRequest):
    """Run a question through the SELF-RAG pipeline and return evaluation trace data."""
    public_response, trace = _run_graph(request)
    return {**public_response, "trace": trace}


@app.get("/health")
async def health():
    return {"status": "ok"}
