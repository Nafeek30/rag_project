from datetime import datetime, timezone
from typing import Literal
from uuid import uuid4

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from langchain_core.documents import Document
from pydantic import BaseModel, Field

from api.main import run_self_rag

app = FastAPI(title="SELF-RAG API", description="API for the LangGraph SELF-RAG pipeline")

app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"https?://(localhost|127\.0\.0\.1)(:\d+)?$",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class QueryRequest(BaseModel):
    question: str


class SourceReference(BaseModel):
    id: str
    title: str
    excerpt: str
    url: str | None = None


class AskResponse(BaseModel):
    answer: str
    sources: list[SourceReference] = Field(default_factory=list)


class ChatHistoryMessage(BaseModel):
    role: Literal["assistant", "user"]
    content: str


class ChatRequest(BaseModel):
    conversation_id: str
    message: str
    history: list[ChatHistoryMessage] = Field(default_factory=list)


class AssistantMessage(BaseModel):
    id: str
    role: Literal["assistant"] = "assistant"
    content: str
    created_at: str
    sources: list[SourceReference] = Field(default_factory=list)


class ChatResponse(BaseModel):
    conversation_id: str
    answer: str
    message: AssistantMessage
    sources: list[SourceReference] = Field(default_factory=list)


def build_excerpt(text: str, length: int = 180) -> str:
    normalized_text = " ".join(text.split())
    if len(normalized_text) <= length:
        return normalized_text
    return f"{normalized_text[: length - 3].rstrip()}..."


def serialize_sources(documents: list[Document]) -> list[SourceReference]:
    sources: list[SourceReference] = []

    for index, document in enumerate(documents, start=1):
        metadata = document.metadata or {}
        source_label = metadata.get("source")
        title = str(source_label) if source_label else f"Source {index}"
        url = metadata.get("url")

        sources.append(
            SourceReference(
                id=f"source_{index}",
                title=title,
                excerpt=build_excerpt(document.page_content),
                url=str(url) if url else None,
            )
        )

    return sources


def build_chat_response(conversation_id: str, answer: str, documents: list[Document]) -> ChatResponse:
    created_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    sources = serialize_sources(documents)

    return ChatResponse(
        conversation_id=conversation_id,
        answer=answer,
        message=AssistantMessage(
            id=f"assistant_{uuid4()}",
            content=answer,
            created_at=created_at,
            sources=sources,
        ),
        sources=sources,
    )


@app.post("/ask", response_model=AskResponse)
async def ask_question(request: QueryRequest) -> AskResponse:
    """
    Receives a question, runs it through the SELF-RAG graph,
    and returns the final generated answer.
    """
    answer, documents = run_self_rag(request.question)
    return AskResponse(answer=answer, sources=serialize_sources(documents))


@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    """
    Compatibility endpoint for the React chat UI.

    The current SELF-RAG graph is single-turn, so `history` is accepted
    for forward compatibility but not yet incorporated into generation.
    """
    answer, documents = run_self_rag(request.message)
    return build_chat_response(request.conversation_id, answer, documents)
