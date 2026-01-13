"""Base class for memory tool"""

from abc import ABCMeta
from pathlib import Path

from ..core.enumeration import MemoryType
from ..core.op import BaseOp
from ..core.schema import ToolCall, MemoryNode
from ..core.utils import CacheHandler


class BaseMemoryTool(BaseOp, metaclass=ABCMeta):
    """Base class for memory tool"""

    def __init__(
        self,
        enable_multiple: bool = True,
        enable_thinking_params: bool = False,
        meta_memory_path: str = "./meta_memory",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.enable_multiple: bool = enable_multiple
        self.enable_thinking_params: bool = enable_thinking_params
        self.meta_memory_path: str = meta_memory_path
        self._meta_memory: CacheHandler | None = None

    def _build_parameters(self) -> dict:
        return {}

    def _build_multiple_parameters(self) -> dict:
        return {}

    def _build_tool_call(self) -> ToolCall:
        tool_call_params: dict = {
            "description": self.get_prompt("tool" + ("_multiple" if self.enable_multiple else "")),
        }

        if self.enable_multiple:
            parameters = self._build_multiple_parameters()
        else:
            parameters = self._build_parameters()

        if parameters:
            tool_call_params["parameters"] = parameters

            if self.enable_thinking_params and "thinking" not in parameters["properties"]:
                parameters["properties"] = {
                    "thinking": {
                        "type": "string",
                        "description": "Your thinking and reasoning about how to fill in the parameters",
                    },
                    **parameters["properties"],
                }
                parameters["required"] = ["thinking", *parameters["required"]]

        return ToolCall(**tool_call_params)

    @property
    def meta_memory(self) -> CacheHandler:
        """Get or create the meta memory cache handler."""
        if self._meta_memory is None:
            self._meta_memory = CacheHandler(Path(self.meta_memory_path) / self.vector_store.collection_name)
        return self._meta_memory

    @property
    def memory_type(self) -> MemoryType:
        """Get the memory type from context."""
        return MemoryType(self.context.get("memory_type"))

    @property
    def memory_target(self) -> str:
        """Get the memory target from context."""
        return self.context.get("memory_target", "")

    @property
    def ref_memory_id(self) -> str:
        """Get the reference memory ID from context."""
        return self.context.get("ref_memory_id", "")

    @property
    def author(self) -> str:
        """Get the author from context."""
        return self.context.get("author", "")

    def _build_memory_node(
        self,
        memory_content: str,
        memory_type: MemoryType | None = None,
        memory_target: str = "",
        ref_memory_id: str = "",
        when_to_use: str = "",
        author: str = "",
        metadata: dict | None = None,
    ) -> MemoryNode:
        """Build MemoryNode from content, when_to_use, and metadata."""
        return MemoryNode(
            memory_type=memory_type or self.memory_type,
            memory_target=memory_target or self.memory_target,
            when_to_use=when_to_use or "",
            content=memory_content,
            ref_memory_id=ref_memory_id or self.ref_memory_id,
            author=author or self.author,
            metadata=metadata or {},
        )
