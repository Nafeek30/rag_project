# RAG Workspace UI

React front end for a ChatGPT-like assistant experience backed by the SELF-RAG FastAPI service in this repository.

## Stack

- React 19
- TypeScript
- Vite
- CSS Modules

## Getting started

1. Install dependencies with `npm install`.
2. Copy `.env.example` to `.env.local`.
3. Start the app with `npm run dev`.

## FastAPI integration

The example environment file points to the local FastAPI server:

```bash
VITE_API_BASE_URL=http://127.0.0.1:8000
VITE_USE_MOCK_API=false
```

UI expects a `POST /api/chat` style endpoint and sends a payload shaped like:

```json
{
  "conversation_id": "conversation_123",
  "message": "What did we cover in lecture 4?",
  "history": [
    {
      "role": "assistant",
      "content": "How can I help?"
    },
    {
      "role": "user",
      "content": "What did we cover in lecture 4?"
    }
  ]
}
```

The response normalizer accepts either:

- `reply` or `answer` string fields
- a `message` object with `content`
- optional `sources` metadata for retrieval references

Set `VITE_USE_MOCK_API=true` if you want to run the UI without the backend during frontend-only work.
