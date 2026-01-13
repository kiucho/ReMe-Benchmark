"""Command service module for managing and executing command-based workflows."""

from loguru import logger

from .base_service import BaseService
from ..context import C
from ..flow import CmdFlow, BaseFlow
from ..utils.common_utils import run_coro_safely


@C.register_service("cmd")
class CmdService(BaseService):
    """Service implementation for handling command flow execution logic."""

    def __init__(self, **kwargs):
        """Initialize the command service instance."""
        super().__init__(**kwargs)
        self._cmd_flow: CmdFlow | None = None

    def integrate_flow(self, flow: BaseFlow) -> str | None:
        """Integrate the workflow configuration into the command service."""
        self._cmd_flow = CmdFlow(flow=C.service_config.flow)

    def run(self):
        """Execute the command flow in either asynchronous or synchronous mode."""
        super().run()

        if self._cmd_flow.async_mode:
            response = run_coro_safely(self._cmd_flow.call(**C.service_config.cmd.model_extra))
        else:
            response = self._cmd_flow.call_sync(**C.service_config.cmd.model_extra)

        if response.answer:
            logger.info(f"response.answer={response.answer}")
