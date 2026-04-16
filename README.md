# SELF-RAG With React UI

This repository combines the existing SELF-RAG LangGraph backend with the `web_page` React chat interface. The backend remains the primary implementation of the retrieval, grading, and hallucination-checking pipeline, while the frontend talks to it through a thin FastAPI compatibility layer.

## Project structure

```text
rag_project/
├── api/                 # FastAPI app plus the compiled SELF-RAG graph
├── indexing/            # Ingestion and indexing experiments
├── web_app/             # React/Vite chat interface
├── pyproject.toml       # Python dependencies managed by uv
└── uv.lock              # Deterministic Python lockfile
```

## Backend setup

1. Install Python dependencies:

```bash
uv sync
```

2. Create a backend environment file:

```bash
cp .env.example .env
```

3. Set `LLM_PROVIDER` to `ollama`, `groq`, or `openai`, then provide the matching API key if needed.

4. Start the FastAPI server:

```bash
uv run uvicorn api.api:app --reload
```

The API runs at `http://127.0.0.1:8000` and exposes Swagger docs at `http://127.0.0.1:8000/docs`.

## Frontend setup

1. Install Node dependencies:

```bash
cd web_app
npm install
```

2. Create a frontend environment file:

```bash
cp .env.example .env.local
```

3. Start the Vite dev server:

```bash
npm run dev
```

By default, the example frontend config points at the local FastAPI server.

## API endpoints

### `POST /ask`

Single-turn backend endpoint for directly exercising the SELF-RAG pipeline.

Request:

```json
{
  "question": "What are the common symptoms of strep throat?"
}
```

Response:

```json
{
  "answer": "Common symptoms include throat pain, swollen lymph nodes, and red tonsils.",
  "sources": [
    {
      "id": "source_1",
      "title": "mock_medical_db",
      "excerpt": "Strep throat is a bacterial infection..."
    }
  ]
}
```

### `POST /api/chat`

UI-facing compatibility endpoint used by the React app.

Request:

```json
{
  "conversation_id": "conversation_123",
  "message": "What are the symptoms of strep throat?",
  "history": [
    {
      "role": "assistant",
      "content": "How can I help?"
    }
  ]
}
```

Response:

```json
{
  "conversation_id": "conversation_123",
  "answer": "Common symptoms include throat pain, swollen lymph nodes, and red tonsils.",
  "message": {
    "id": "assistant_123",
    "role": "assistant",
    "content": "Common symptoms include throat pain, swollen lymph nodes, and red tonsils.",
    "created_at": "2026-04-15T12:00:00Z",
    "sources": []
  },
  "sources": []
}
```

`history` is accepted for forward compatibility, but the current graph still operates as a single-turn SELF-RAG flow.
