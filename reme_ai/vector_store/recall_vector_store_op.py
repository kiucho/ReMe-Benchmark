"""Operation for recalling memories from the vector store based on a query."""

import math
from typing import List

from flowllm.core.context import C
from flowllm.core.op import BaseAsyncOp
from flowllm.core.schema import VectorNode
from loguru import logger

from reme_ai.schema.memory import BaseMemory, vector_node_to_memory


@C.register_op()
class RecallVectorStoreOp(BaseAsyncOp):
    """Operation that retrieves relevant memories from the vector store.

    This operation performs a semantic search on the vector store to find
    memories relevant to a given query. It supports optional score filtering
    and deduplication based on memory content.
    """

    async def async_execute(self):
        """Execute the memory recall operation.

        Performs a semantic search in the vector store using the provided query,
        retrieves the top-k most relevant memories, and optionally filters them
        by a score threshold. Duplicate memories (based on content) are removed.

        Expected context attributes:
            workspace_id: The workspace ID to search memories in.
            query: The search query string (or key specified by recall_key).

        Expected op_params:
            recall_key: Key in context containing the query (default: "query").
            threshold_score: Optional minimum score threshold for filtering.

        Context attributes used:
            top_k: Number of top results to retrieve (default: 3).

        Sets context attributes:
            response.metadata["memory_list"]: List of retrieved BaseMemory objects.
        """
        recall_key: str = self.op_params.get("recall_key", "query")
        top_k: int = self.context.get("top_k", 3)

        query: str = self.context[recall_key]
        assert query, "query should be not empty!"

        workspace_id: str = self.context.workspace_id
        nodes: List[VectorNode] = await self.vector_store.async_search(
            query=query,
            workspace_id=workspace_id,
            top_k=top_k,
        )
        await self._ensure_embedding_scores(query, nodes)
        memory_list: List[BaseMemory] = []
        memory_content_list: List[str] = []
        for node in nodes:
            memory: BaseMemory = vector_node_to_memory(node)
            if memory.content not in memory_content_list:
                memory_list.append(memory)
                memory_content_list.append(memory.content)
        logger.info(f"retrieve memory.size={len(memory_list)}")

        threshold_score: float | None = self.op_params.get("threshold_score", None)
        if threshold_score is not None:
            memory_list = [mem for mem in memory_list if mem.score >= threshold_score or mem.score is None]
            logger.info(f"after filter by threshold_score size={len(memory_list)}")

        self.context.response.metadata["memory_list"] = memory_list

    async def _ensure_embedding_scores(self, query: str, nodes: List[VectorNode]) -> None:
        missing_scores = [
            node
            for node in nodes
            if isinstance(getattr(node, "metadata", None), dict) and "_score" not in node.metadata
        ]
        if not missing_scores:
            return
        query_vector = await self._get_embedding(query)
        if not query_vector:
            return

        nodes_without_vector = [node for node in missing_scores if not getattr(node, "vector", None) and node.content]
        if nodes_without_vector:
            embeddings = await self._get_embeddings([node.content for node in nodes_without_vector])
            if len(embeddings) == len(nodes_without_vector):
                for node, vector in zip(nodes_without_vector, embeddings):
                    node.vector = vector
            else:
                logger.warning(
                    f"Embedding batch size mismatch: {len(embeddings)} vs {len(nodes_without_vector)}",
                )

        for node in missing_scores:
            metadata = getattr(node, "metadata", None)
            if not isinstance(metadata, dict):
                node.metadata = {}
                metadata = node.metadata
            vector = getattr(node, "vector", None)
            if not vector:
                continue
            score = self._cosine_similarity(query_vector, vector)
            if score is None:
                continue
            metadata["_score"] = score

    async def _get_embedding(self, text: str) -> List[float]:
        try:
            if hasattr(self.vector_store, "get_embedding"):
                return await self.vector_store.get_embedding(text)
        except Exception as e:
            logger.warning(f"Vector store embedding failed: {e}")
        try:
            embedding_model = getattr(self, "embedding_model", None)
            if embedding_model and hasattr(embedding_model, "get_embedding"):
                return await embedding_model.get_embedding(text)
        except Exception as e:
            logger.warning(f"Embedding model failed: {e}")
        return []

    async def _get_embeddings(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        try:
            if hasattr(self.vector_store, "get_embeddings"):
                return await self.vector_store.get_embeddings(texts)
        except Exception as e:
            logger.warning(f"Vector store batch embedding failed: {e}")
        try:
            embedding_model = getattr(self, "embedding_model", None)
            if embedding_model and hasattr(embedding_model, "get_embeddings"):
                return await embedding_model.get_embeddings(texts)
        except Exception as e:
            logger.warning(f"Embedding model batch failed: {e}")
        embeddings: List[List[float]] = []
        for text in texts:
            embeddings.append(await self._get_embedding(text))
        return embeddings

    @staticmethod
    def _cosine_similarity(vec1: List[float], vec2: List[float]) -> float | None:
        if not vec1 or not vec2 or len(vec1) != len(vec2):
            return None
        dot = 0.0
        norm1 = 0.0
        norm2 = 0.0
        for a, b in zip(vec1, vec2):
            dot += a * b
            norm1 += a * a
            norm2 += b * b
        if norm1 == 0.0 or norm2 == 0.0:
            return None
        return dot / (math.sqrt(norm1) * math.sqrt(norm2))
