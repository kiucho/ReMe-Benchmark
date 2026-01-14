import os

from dotenv import load_dotenv
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_deepseek import ChatDeepSeek
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI, AzureChatOpenAI

load_dotenv()


def _get_reasoning_model_kwargs() -> dict:
    return {"reasoning_effort": "medium"}


def load_model(backend_model: str = "gpt-oss:20b") -> BaseChatModel:
    if backend_model in ["gpt-oss:20b", "qwen3:32b"]:
        llm = ChatOllama(
            model=backend_model,
            temperature=0,
            validate_model_on_init=True,
            base_url=os.getenv("OLLAMA_API_URL"),
        )
    elif backend_model in ["gpt-5-mini", "gpt-5"]:
        model_kwargs = _get_reasoning_model_kwargs()
        llm = ChatOpenAI(
            model_name=backend_model,
            model_kwargs=model_kwargs,
        )
    elif backend_model in ["deepseek-chat", "deepseek-reasoner"]:
        llm = ChatDeepSeek(
            model=backend_model,
            base_url="https://api.deepseek.com",
        )
    elif backend_model.startswith("azure/"):
        model_kwargs = _get_reasoning_model_kwargs()
        deployment_name = backend_model.replace("azure/", "")
        llm = AzureChatOpenAI(
            azure_deployment=deployment_name,
            api_key=os.getenv("AZURE_LLM_API_KEY") or os.getenv("AZURE_API_KEY"),
            azure_endpoint=os.getenv("AZURE_LLM_API_BASE") or os.getenv("AZURE_API_BASE"),
            api_version=os.getenv("AZURE_LLM_API_VERSION") or os.getenv("AZURE_API_VERSION") or "2024-02-01",
            temperature=0,
            timeout=180,  # 3 minutes timeout
            max_retries=2,  # Retry up to 2 times on failure
            # model_kwargs=model_kwargs,
        )
    else:
        raise ValueError(f"Unsupported backend model: {backend_model}")
    return llm
