# RAG Workspace UI

React front end for a ChatGPT like AI assistant experience. The app is structured so the UI can run against a mock adapter during development and switch to a FastAPI backend when the API is ready.

## Stack

- React 19
- TypeScript
- Vite
- CSS Modules

## Getting started

1. Install dependencies with `npm install`.
2. Copy `.env.example` to `.env.local` if you want to override defaults.
3. Start the app with `npm run dev`.

## FastAPI integration

By default, the UI uses a mock chat service. For a real backend, set:

```bash
VITE_USE_MOCK_API=false
VITE_API_BASE_URL=http://127.0.0.1:8000
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
