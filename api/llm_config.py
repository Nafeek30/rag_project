import os
from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from langchain_anthropic import ChatAnthropic

load_dotenv()

DEFAULT_PROVIDER = os.getenv("LLM_PROVIDER", "groq").lower()

PROVIDER_LABELS = {
    "groq": "Groq (Llama 3.1)",
    "claude": "Claude (Sonnet 4.6)",
    "qwen3": "Qwen3 4B (offline)",
}


def get_llm(is_json: bool = False, provider: str = None):
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

    elif p == "qwen3":
        # Qwen3 supports /no_think suffix to disable chain-of-thought for faster responses.
        # We keep thinking ON for generation and OFF for structured grading/routing calls.
        model_name = "qwen3:4b" if not is_json else "qwen3:4b"
        kwargs = {
            "model": model_name,
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


# Module-level defaults
json_llm = get_llm(is_json=True)
standard_llm = get_llm(is_json=False)
