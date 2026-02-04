"""Synchronous OpenAI-compatible embedding model implementation for ReMe."""

from openai import OpenAI

from .openai_embedding_model import OpenAIEmbeddingModel
from ..context import C
from ..utils.openai_httpx import make_httpx_client_for_openai


@C.register_embedding_model("openai_sync")
class OpenAIEmbeddingModelSync(OpenAIEmbeddingModel):
    """Synchronous embedding model implementation that extends the asynchronous OpenAI model."""

    def _create_client(self):
        """Create and return an internal synchronous OpenAI client instance."""
        http_client = make_httpx_client_for_openai(self.base_url)
        if http_client is None:
            return OpenAI(api_key=self.api_key, base_url=self.base_url)
        return OpenAI(api_key=self.api_key, base_url=self.base_url, http_client=http_client)

    def _get_embeddings_sync(self, input_text: list[str], **kwargs) -> list[list[float]]:
        """Fetch embeddings synchronously from the API for a batch of strings."""
        completion = self._client.embeddings.create(
            model=self.model_name,
            input=input_text,
            dimensions=self.dimensions,
            encoding_format=self.encoding_format,
            **self.kwargs,
            **kwargs,
        )

        result_emb = [[] for _ in range(len(input_text))]
        for emb in completion.data:
            result_emb[emb.index] = emb.embedding
        return result_emb

    def close_sync(self):
        """Close the synchronous OpenAI client and release network resources."""
        self._client.close()
