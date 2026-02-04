import os

import httpx


def _normalize_url(url: str) -> str:
    return url.rstrip("/")


def is_gpt_oss_endpoint(base_url: str) -> bool:
    """Return True when base_url matches GPT_OSS_API_URL.

    We use this to apply HTTPX settings needed for some self-hosted endpoints.
    """

    if not base_url:
        return False
    gpt_oss_url = os.getenv("GPT_OSS_API_URL") or ""
    if not gpt_oss_url:
        return False
    return _normalize_url(base_url) == _normalize_url(gpt_oss_url)


def make_httpx_client_for_openai(base_url: str) -> httpx.Client | None:
    if not is_gpt_oss_endpoint(base_url):
        return None
    return httpx.Client(verify=False, timeout=60.0)


def make_httpx_async_client_for_openai(base_url: str) -> httpx.AsyncClient | None:
    if not is_gpt_oss_endpoint(base_url):
        return None
    return httpx.AsyncClient(verify=False, timeout=60.0)
