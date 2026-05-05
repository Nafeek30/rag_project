import os
from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from langchain_anthropic import ChatAnthropic

load_dotenv()

DEFAULT_PROVIDER = os.getenv("LLM_PROVIDER", "groq").lower()

PROVIDER_LABELS = {
    "groq":  "Groq (Llama 3.1)",
    "claude": "Claude (Sonnet 4.6)",
    "grok":  "Grok (xAI)",
    "qwen3": "Qwen3 4B (offline)",
    "openai": "OpenAI (GPT-4o mini)",
}


def get_llm(is_json: bool = False, provider: str = None, thinking: bool = False):
    """Return the appropriate LLM instance for the given provider."""
    p = (provider or DEFAULT_PROVIDER).lower()

    if p in ("claude", "anthropic"):
        return ChatAnthropic(
            model="claude-sonnet-4-6",
            temperature=0,
            anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"),
        )

    elif p == "openai":
        kwargs = {"model": "gpt-4o-mini", "temperature": 0}
        if is_json:
            kwargs["model_kwargs"] = {"response_format": {"type": "json_object"}}
        return ChatOpenAI(**kwargs)

    elif p == "groq":
        kwargs = {
            "model": "llama-3.1-8b-instant",
            "temperature": 0,
            "groq_api_key": os.getenv("GROQ_API_KEY"),
        }
        if is_json:
            kwargs["model_kwargs"] = {"response_format": {"type": "json_object"}}
        return ChatGroq(**kwargs)

    elif p == "grok":
        # xAI Grok — OpenAI-compatible API
        kwargs = {
            "model": "grok-3-mini",
            "temperature": 0,
            "openai_api_key": os.getenv("GROK_API_KEY"),
            "openai_api_base": "https://api.x.ai/v1",
        }
        if is_json:
            kwargs["model_kwargs"] = {"response_format": {"type": "json_object"}}
        return ChatOpenAI(**kwargs)

    elif p == "qwen3":
        # Allow overriding model via env var, default to common local models
        model = os.getenv("OLLAMA_MODEL", "qwen3:4b")
        kwargs = {
            "model": model,
            "temperature": 0,
            "base_url": os.getenv("OLLAMA_HOST", "http://localhost:11434"),
        }
        if is_json:
            kwargs["format"] = "json"
        return ChatOllama(**kwargs)

    else:
        kwargs = {"model": "llama3", "temperature": 0}
        if is_json:
            kwargs["format"] = "json"
        return ChatOllama(**kwargs)
