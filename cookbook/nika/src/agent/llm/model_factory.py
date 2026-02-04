import os
import httpx
from dotenv import load_dotenv
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_deepseek import ChatDeepSeek
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI, AzureChatOpenAI

load_dotenv()


def _get_reasoning_model_kwargs() -> dict:
    return {"reasoning_effort": "high"}


def load_model(backend_model: str = "gpt-oss:20b", temperature: float | None = None) -> BaseChatModel:
    # if backend_model in ["gpt-oss:20b", "qwen3:32b"]:
    #     llm = ChatOllama(
    #         model=backend_model,
    #         temperature=0,
    #         validate_model_on_init=True,
    #         base_url=os.getenv("OLLAMA_API_URL"),
    #     )
    # elif backend_model in ["gpt-5-mini", "gpt-5"]:
    #     model_kwargs = _get_reasoning_model_kwargs()
    #     llm = ChatOpenAI(
    #         model_name=backend_model,
    #         model_kwargs=model_kwargs,
    #     )
    # elif backend_model in ["deepseek-chat", "deepseek-reasoner"]:
    #     llm = ChatDeepSeek(
    #         model=backend_model,
    #         base_url="https://api.deepseek.com",
    #     )
    
    if backend_model.startswith("azure/"):
        model_kwargs = _get_reasoning_model_kwargs()
        deployment_name = backend_model.replace("azure/", "")
        azure_temperature = 0 if temperature is None else temperature
        llm = AzureChatOpenAI(
            azure_deployment=deployment_name,
            api_key=os.getenv("AZURE_LLM_API_KEY") or os.getenv("AZURE_API_KEY"),
            azure_endpoint=os.getenv("AZURE_LLM_API_BASE") or os.getenv("AZURE_API_BASE"),
            api_version=os.getenv("AZURE_LLM_API_VERSION") or os.getenv("AZURE_API_VERSION") or "2024-02-01",
            temperature=azure_temperature,
            timeout=180,  # 3 minutes timeout
            max_retries=2,  # Retry up to 2 times on failure
            # model_kwargs=model_kwargs,
        )
    elif backend_model.startswith("gpt-oss-120b"):
        _http_client = httpx.Client(verify=False, timeout=60.0)
        _async_http_client = httpx.AsyncClient(verify=False, timeout=60.0)
        _GPT_OSS_API_URL = os.getenv("GPT_OSS_API_URL")
        _GPT_OSS_API_KEY = os.getenv("GPT_OSS_API_KEY")
        llm = ChatOpenAI(
            model="kt-gpt-oss-rh014",
            base_url=_GPT_OSS_API_URL,
            api_key=_GPT_OSS_API_KEY,
            http_client=_http_client,
            http_async_client=_async_http_client,
            temperature=temperature,
            timeout=180,  # 3 minutes timeout
            max_retries=2,  # Retry up to 2 times on failure
        )
    else:
        raise ValueError(f"Unsupported backend model: {backend_model}")
    return llm
