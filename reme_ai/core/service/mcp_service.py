"""Model Context Protocol (MCP) service implementation."""

from typing import Any

from fastmcp import FastMCP
from fastmcp.tools import FunctionTool

from .base_service import BaseService
from ..context import C
from ..flow import BaseFlow


@C.register_service("mcp")
class MCPService(BaseService):
    """Expose flows as Model Context Protocol (MCP) tools."""

    def __init__(self, **kwargs: Any):
        """Initialize FastMCP instance with service settings."""
        super().__init__(**kwargs)
        self.mcp = FastMCP(name=C.service_config.app_name)

    def integrate_flow(self, flow: BaseFlow) -> str | None:
        """Register a non-streaming flow as an MCP tool."""
        if flow.stream:
            return None

        tool_call, request_model = self._prepare_route(flow)

        async def execute_tool(**kwargs):
            """Execute flow logic and return the string answer."""
            request_instance = request_model(**kwargs)
            response = await flow.call(**request_instance.model_dump(exclude_none=True))
            return response.answer

        self.mcp.add_tool(
            FunctionTool(
                name=tool_call.name,  # noqa
                description=tool_call.description,  # noqa
                fn=execute_tool,
                parameters=tool_call.parameters.simple_input_dump(),
            ),
        )
        return tool_call.name

    def run(self):
        """Run the MCP server with specified transport protocol."""
        super().run()
        cfg = C.service_config.mcp

        run_args: dict = {"transport": cfg.transport, "show_banner": False, **cfg.model_extra}

        # Add network settings for non-stdio transports
        if cfg.transport != "stdio":
            run_args.update({"host": cfg.host, "port": cfg.port})

        self.mcp.run(**run_args)
