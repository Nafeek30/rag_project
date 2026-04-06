import os
from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI

load_dotenv()

# defaulting to 'ollama'
PROVIDER = os.getenv("LLM_PROVIDER", "ollama").lower()

def get_llm(is_json: bool = False):
    """Returns the appropriate LLM based on environment configuration."""
    if PROVIDER == "openai":
        # Uses OPENAI_API_KEY from the environment
        kwargs = {"model": "gpt-4o-mini", "temperature": 0}
        if is_json:
            kwargs["model_kwargs"] = {"response_format": {"type": "json_object"}}
        return ChatOpenAI(**kwargs)

    else:
        # Defaults to local Ollama setup
        kwargs = {"model": "llama3", "temperature": 0}
        if is_json:
            kwargs["format"] = "json"
        return ChatOllama(**kwargs)

# Initialize critic and generator
json_llm = get_llm(is_json=True)
standard_llm = get_llm(is_json=False)