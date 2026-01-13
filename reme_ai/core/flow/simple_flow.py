"""Simple flow implementation that directly uses a predefined flow operation."""

from .base_flow import BaseFlow
from ..op import BaseOp
from ..schema import ToolCall


class SimpleFlow(BaseFlow):
    """Simple flow that directly uses a predefined flow operation."""

    def _build_flow(self) -> BaseOp:
        assert self._flow_op is not None
        return self._flow_op.copy()

    def _build_tool_call(self) -> ToolCall:
        assert self._flow_op is not None
        return self._flow_op.tool_call
