"""Main application module for managing ReMe AI service lifecycle and flow execution."""

import asyncio
import os

from .context import C
from .flow import BaseFlow
from .schema import ServiceConfig, Response
from .utils import PydanticConfigParser, init_logger, execute_stream_task, run_coro_safely, load_env


class Application:
    """
    Main application class for managing the lifecycle of ReMe AI services.

    Handles initialization, configuration, service management, and flow execution
    for both synchronous and asynchronous contexts.
    """

    def __init__(
        self,
        *args,
        llm_api_key: str | None = None,
        llm_api_base: str | None = None,
        embedding_api_key: str | None = None,
        embedding_api_base: str | None = None,
        service_config: ServiceConfig | None = None,
        parser: type[PydanticConfigParser] | None = None,
        config_path: str | None = None,
        enable_logo: bool = True,
        llm: dict | None = None,
        embedding_model: dict | None = None,
        vector_store: dict | None = None,
        token_counter: dict | None = None,
        **kwargs,
    ):
        """
        Initialize the Application with configuration settings.

        Args:
            *args: Additional arguments passed to parser. Examples:
                - "llm.default.model_name=qwen3-30b-a3b-thinking-2507"
                - "llm.default.backend=openai_compatible"
                - "llm.default.temperature=0.6"
                - "embedding_model.default.model_name=text-embedding-v4"
                - "embedding_model.default.backend=openai_compatible"
                - "embedding_model.default.dimensions=1024"
                - "vector_store.default.backend=memory"
                - "vector_store.default.embedding_model=default"
            llm_api_key: API key for LLM service
            llm_api_base: Base URL for LLM service
            embedding_api_key: API key for embedding service
            embedding_api_base: Base URL for embedding service
            service_config: Pre-built service configuration object
            parser: Custom parser class for configuration (defaults to PydanticConfigParser)
            config_path: Path to configuration file
            enable_logo: Whether to display the ReMe logo on startup
            llm: LLM configuration dictionary
            embedding_model: Embedding model configuration dictionary
            vector_store: Vector store configuration dictionary
            token_counter: Token counter configuration dictionary
            **kwargs: Additional keyword arguments passed to parser. Same format as args but as kwargs. Examples:
                - **{"llm.default.model_name": "qwen3-30b-a3b-thinking-2507"}
        """

        load_env()
        self._update_env("REME_LLM_API_KEY", llm_api_key)
        self._update_env("REME_LLM_BASE_URL", llm_api_base)
        self._update_env("REME_EMBEDDING_API_KEY", embedding_api_key)
        self._update_env("REME_EMBEDDING_BASE_URL", embedding_api_base)

        init_logger()

        # Use default parser if not provided
        parser_class = parser if parser is not None else PydanticConfigParser
        self.parser = parser_class(ServiceConfig)

        if service_config is None:
            input_args = []
            if config_path:
                input_args.append(f"config={config_path}")
            if args:
                input_args.extend(args)
            if kwargs:
                input_args.extend([f"{k}={v}" for k, v in kwargs.items()])
            service_config = self.parser.parse_args(*input_args)

        C.service_config = service_config

        if llm:
            C.update_section_config("llm", **llm)
        if embedding_model:
            C.update_section_config("embedding_model", **embedding_model)
        if vector_store:
            C.update_section_config("vector_store", **vector_store)
        if token_counter:
            C.update_section_config("token_counter", **token_counter)
        C.service_config.enable_logo = enable_logo
        C.print_logo()

    @staticmethod
    def _update_env(key: str, value: str | None):
        """Update environment variable if value is provided."""
        if value:
            os.environ[key] = value

    @staticmethod
    async def start():
        """Initialize the service context and prepare external MCP servers."""
        C.initialize_service_context()
        await C.prepare_mcp_servers()

    @staticmethod
    def start_sync():
        """Synchronous version of start()."""
        C.initialize_service_context()
        run_coro_safely(C.prepare_mcp_servers())

    @staticmethod
    async def stop(wait_thread_pool: bool = True, wait_ray: bool = True):
        """
        Stop the application and cleanup resources.

        Args:
            wait_thread_pool: Whether to wait for thread pool shutdown
            wait_ray: Whether to wait for Ray shutdown
        """
        await C.close()
        C.shutdown_thread_pool(wait=wait_thread_pool)
        C.shutdown_ray(wait=wait_ray)

    @staticmethod
    def stop_sync(wait_thread_pool: bool = True, wait_ray: bool = True):
        """Synchronous version of stop()."""
        C.close_sync()
        C.shutdown_thread_pool(wait=wait_thread_pool)
        C.shutdown_ray(wait=wait_ray)

    async def __aenter__(self):
        """Async context manager entry."""
        await self.start()
        return self

    def __enter__(self):
        """Context manager entry."""
        self.start_sync()
        return self

    async def __aexit__(self, exc_type=None, exc_val=None, exc_tb=None):
        """Async context manager exit."""
        await self.stop()
        return False

    def __exit__(self, exc_type=None, exc_val=None, exc_tb=None):
        """Context manager exit."""
        self.stop_sync()
        return False

    @staticmethod
    async def execute_flow(name: str, **kwargs) -> Response:
        """
        Execute a flow asynchronously.

        Args:
            name: Name of the flow to execute
            **kwargs: Arguments to pass to the flow

        Returns:
            Response object from the flow execution
        """
        flow: BaseFlow = C.get_flow(name)
        return await flow.call(**kwargs)

    @staticmethod
    def execute_flow_sync(name: str, **kwargs) -> Response:
        """
        Execute a flow synchronously.

        Args:
            name: Name of the flow to execute
            **kwargs: Arguments to pass to the flow

        Returns:
            Response object from the flow execution
        """
        flow: BaseFlow = C.get_flow(name)
        return flow.call_sync(**kwargs)

    @staticmethod
    async def execute_stream_flow(name: str, **kwargs):
        """
        Execute a streaming flow asynchronously.

        Args:
            name: Name of the streaming flow to execute
            **kwargs: Arguments to pass to the flow

        Yields:
            Stream chunks from the flow execution

        Raises:
            AssertionError: If the flow is not configured for streaming
        """
        flow: BaseFlow = C.get_flow(name)
        assert flow.stream is True, "non-stream flow is not supported in execute_stream_flow!"
        stream_queue = asyncio.Queue()
        task = asyncio.create_task(flow.call(stream_queue=stream_queue, **kwargs))

        async for chunk in execute_stream_task(
            stream_queue=stream_queue,
            task=task,
            task_name=name,
            as_bytes=False,
        ):
            yield chunk

    @staticmethod
    def run_service():
        """Run the configured service (HTTP, MCP, or CMD)."""
        C.get_service().run()
