"""ReMe classes for simplified configuration and execution."""

from typing import Literal

from .core.application import Application
from .core.config import ReMeConfigParser
from .core.context import C
from .core.enumeration import Role
from .core.vector_store import BaseVectorStore
from .mem_agent.summarizer import (
    ReMeSummarizer,
    # ToolSummarizer,
    PersonalSummarizer,
    ProceduralSummarizer,
    # IdentitySummarizer,
)
from .mem_agent.retriever import ReMeRetriever

# from .mem_agent.chat import ReMyAgent
from .mem_tool import (
    HandsOffTool,
    ReadHistoryMemory,
    # ReadIdentityMemory,
    # UpdateIdentityMemory,
    AddMetaMemory,
    AddMemory,
    AddSummaryMemory,
    DeleteMemory,
    UpdateMemory,
    VectorRetrieveMemory,
)
from .core.schema import Message


class ReMe(Application):
    """Simplified ReMe application that auto-initializes the service context."""

    def __init__(
        self,
        *args,
        llm_api_key: str | None = None,
        llm_api_base: str | None = None,
        embedding_api_key: str | None = None,
        embedding_api_base: str | None = None,
        enable_logo: bool = True,
        llm: dict | None = None,
        embedding_model: dict | None = None,
        vector_store: dict | None = None,
        token_counter: dict | None = None,
        enable_identity_memory: bool = True,
        enable_tool_memory: bool = True,
        force_tool_language: bool = True,
        add_think_tool: bool = False,
        **kwargs,
    ):
        super().__init__(
            *args,
            llm_api_key=llm_api_key,
            llm_api_base=llm_api_base,
            embedding_api_key=embedding_api_key,
            embedding_api_base=embedding_api_base,
            service_config=None,
            parser=ReMeConfigParser,
            config_path=None,
            enable_logo=enable_logo,
            llm=llm,
            embedding_model=embedding_model,
            vector_store=vector_store,
            token_counter=token_counter,
            **kwargs,
        )

        C.initialize_service_context()
        self.enable_identity_memory = enable_identity_memory
        self.enable_tool_memory = enable_tool_memory
        self.force_tool_language = force_tool_language
        self.add_think_tool = add_think_tool

        self._personal_summarizer = PersonalSummarizer(
            tools=[VectorRetrieveMemory(), AddMemory(), DeleteMemory(), UpdateMemory()],
        )
        self._procedural_summarizer = ProceduralSummarizer(
            tools=[VectorRetrieveMemory(), AddMemory(), DeleteMemory(), UpdateMemory()],
        )
        hands_off_tool = HandsOffTool(memory_agents=[self._personal_summarizer, self._procedural_summarizer])
        self._reme_summarizer = ReMeSummarizer(
            tools=[AddMetaMemory(), AddSummaryMemory(), hands_off_tool],
            enable_identity_memory=self.enable_identity_memory,
            enable_tool_memory=self.enable_tool_memory,
            force_tool_language=self.force_tool_language,
            add_think_tool=self.add_think_tool,
        )
        self._reme_retriever = ReMeRetriever(
            tools=[VectorRetrieveMemory(add_memory_type_target=True), ReadHistoryMemory()],
        )

        self.vector_store: BaseVectorStore = C.get_vector_store("default")

    @staticmethod
    def _prepare_messages(messages: list[dict | Message], user_id: str, assistant_id: str):
        if not messages:
            return []

        messages = [Message(**m) if isinstance(m, dict) else m for m in messages]
        for message in messages:
            if message.role is Role.USER and user_id:
                message.name = user_id
            elif message.role is Role.ASSISTANT and assistant_id:
                message.name = assistant_id
        return messages

    async def summary(
        self,
        messages: list[dict],
        description: str = "",
        user_id: str = "",
        assistant_id: str = "",
        memory_mode: Literal["personal", "procedural", "auto"] = "personal",
        **kwargs,
    ):
        """Summarizes messages and stores them as memory based on the specified memory mode."""
        messages = self._prepare_messages(messages, user_id, assistant_id)

        if memory_mode == "personal":
            return await self._personal_summarizer.call(
                messages=messages,
                description=description,
                memory_target=user_id,
                **kwargs,
            )
        elif memory_mode == "procedural":
            return await self._procedural_summarizer.call(
                messages=messages,
                description=description,
                memory_target=user_id,
                **kwargs,
            )
        else:
            return await self._reme_summarizer.call(
                messages=messages,
                description=description,
                memory_target=user_id,
                **kwargs,
            )

    async def retrieve(
        self,
        query: str = "",
        messages: list[dict] | None = None,
        description: str = "",
        user_id: str = "",
        assistant_id: str = "",
        memory_mode: Literal["personal", "procedural", "auto"] = "personal",
        **kwargs,
    ):
        """Retrieves relevant memories based on the query and specified memory mode."""
        messages = self._prepare_messages(messages, user_id, assistant_id)

        if memory_mode == "personal":
            self._reme_retriever.meta_memories = [{"memory_type": "personal", "memory_target": user_id}]
            return await self._reme_retriever.call(query=query, messages=messages, description=description, **kwargs)

        elif memory_mode == "procedural":
            self._reme_retriever.meta_memories = [{"memory_type": "procedural", "memory_target": user_id}]
            return await self._reme_retriever.call(query=query, messages=messages, description=description, **kwargs)

        else:
            return await self._reme_retriever.call(query=query, messages=messages, description=description, **kwargs)
