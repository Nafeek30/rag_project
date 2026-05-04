# 🧠 Adv. NLP RAG Project: SELF-RAG Pipeline

This repository contains a modular, graph-based Retrieval-Augmented Generation (RAG) pipeline built with LangGraph, LangChain, and FastAPI. It implements the SELF-RAG architecture, allowing the LLM to actively route queries, grade document relevance, and self-correct hallucinations before returning a final answer.


## 📂 Project Structure

```text
rag_project/
├── .gitignore
├── README.md
├── pyproject.toml       # Managed by uv
├── uv.lock              # Deterministic dependency lockfile
├── indexing/            # Vector database ingestion and chunking scripts
├── web_app/             # Frontend user interface
└── api/                 # LangGraph and FastAPI backend
    ├── __init__.py
    ├── api.py           # FastAPI server and endpoints
    ├── main.py          # LangGraph edge logic and compilation
    ├── nodes.py         # LangGraph node execution functions
    ├── prompts.py       # Pydantic schemas and LangChain prompts
    └── llm_config.py    # LLM provider routing (Local vs. Cloud)
```

## Getting Started 

This project uses `uv` for lightning-fast, deterministic dependency management within a WSL/Linux environment.

1. Install Dependencies

Ensure you have [uv](https://docs.astral.sh/uv/) installed. Then, clone the repo and sync the environment.
```bash
git clone git@github.com:Nafeek30/rag_project.git
cd rag_project
uv sync
```

2. Configure Your LLM Provider

You can run this pipeline using a local LLM via Ollama/Qwen3, cloud providers via Groq, Anthropic Claude, xAI Grok, or OpenAI.

```bash
cp .env.example .env
```

Copy the format of the `.env.example` file and set your LLM_PROVIDER to `ollama`, `groq`, `claude`, `grok`, `qwen3`, or `openai`. Add the matching provider API key for whichever cloud model you use.

*Note: If you are using Ollama, ensure you have pulled the model inside your WSL environment (`ollama run llama3`).*

3. Run the API Server

Start the FastAPI backend using uvicorn:

```bash
uv run uvicorn api.api:app --reload
```

The server will start on http://127.0.0.1:8000.


### API Endpoints

Once the server is running, you can interact with the graph via the REST API or view the interactive Swagger documentation at http://127.0.0.1:8000/docs.

### `POST /ask`

Submits a query to the SELF-RAG pipeline.

Request Body:

```JSON
{
  "question": "What are the common symptoms of strep throat?"
}
```
Response:

```JSON
{
  "answer": "Common symptoms include throat pain, swollen lymph nodes, and red tonsils."
}
```


### Development Notes

Use `uv add <package>` to safely update the pyproject.toml and uv.lock files.
