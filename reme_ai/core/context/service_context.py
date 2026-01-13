"""Module for managing global service configurations and component registries via a singleton context."""

from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING

from loguru import logger

from .base_context import BaseContext
from .registry import Registry
from ..enumeration import RegistryEnum
from ..schema import ServiceConfig
from ..utils import singleton, print_logo

if TYPE_CHECKING:
    from ..llm import BaseLLM
    from ..embedding import BaseEmbeddingModel
    from ..vector_store import BaseVectorStore
    from ..token_counter import BaseTokenCounter
    from ..flow import BaseFlow
    from ..service import BaseService


@singleton
class ServiceContext(BaseContext):
    """A singleton container for global application state, thread pools, and component registries.

    This class serves as the central management hub for the entire ReMe application, providing:
    - Service configuration management
    - Component registration and instantiation (LLMs, embeddings, vector stores, etc.)
    - Thread pool and Ray distributed computing management
    - MCP (Model Context Protocol) server integration

    The singleton pattern ensures only one instance exists throughout the application lifecycle,
    accessible via the global `C` variable exported at the bottom of this module.
    """

    def __init__(self, **kwargs):
        """Initialize the global context with configuration objects and specialized registries.

        Sets up:
        - Empty service configuration placeholder
        - Thread pool for concurrent operations
        - Registry dictionaries for class registration (templates)
        - Instance dictionaries for instantiated objects (actual instances)
        - MCP server mapping for external tool integration
        """
        super().__init__(**kwargs)

        # Service configuration and runtime settings
        self.service_config: ServiceConfig | None = None
        self.language: str = ""
        self.thread_pool: ThreadPoolExecutor | None = None

        # Registry system: stores class definitions for different component types
        self.registry_dict: dict[RegistryEnum, Registry] = {v: Registry() for v in RegistryEnum.__members__.values()}

        # Instance system: stores instantiated objects created from registered classes
        self.instance_dict: dict[RegistryEnum, dict] = {v: {} for v in RegistryEnum.__members__.values()}

        # MCP server mapping: maps server_name -> {tool_name: ToolCall}
        self.mcp_server_mapping: dict[str, dict] = {}

        # Initialization flag: ensures initialize_service_context is called only once
        self._initialized: bool = False

    def register(self, name: str, register_type: RegistryEnum):
        """Return a decorator to register a component within a specific registry category.

        Args:
            name: The registration name for the component (used for lookup)
            register_type: The type of registry (LLM, EMBEDDING_MODEL, VECTOR_STORE, etc.)

        Returns:
            A decorator function that registers the decorated class

        Example:
            @C.register("my_llm", RegistryEnum.LLM)
            class MyLLM(BaseLLM):
                pass
        """
        return self.registry_dict[register_type].register(name=name)

    def register_llm(self, name: str = ""):
        """Register a Large Language Model class."""
        return self.register(name=name, register_type=RegistryEnum.LLM)

    def register_embedding_model(self, name: str = ""):
        """Register an embedding model class."""
        return self.register(name=name, register_type=RegistryEnum.EMBEDDING_MODEL)

    def register_vector_store(self, name: str = ""):
        """Register a vector store implementation class."""
        return self.register(name=name, register_type=RegistryEnum.VECTOR_STORE)

    def register_op(self, name: str = ""):
        """Register an operation (Op) class."""
        return self.register(name=name, register_type=RegistryEnum.OP)

    def register_flow(self, name: str = ""):
        """Register a workflow or logic flow class."""
        return self.register(name=name, register_type=RegistryEnum.FLOW)

    def register_service(self, name: str = ""):
        """Register a backend service class."""
        return self.register(name=name, register_type=RegistryEnum.SERVICE)

    def register_token_counter(self, name: str = ""):
        """Register a token counting utility class."""
        return self.register(name=name, register_type=RegistryEnum.TOKEN_COUNTER)

    def get_model_class(self, name: str, register_type: RegistryEnum):
        """Retrieve a registered class by name from a specific registry category.

        Args:
            name: The registration name of the class
            register_type: The type of registry to search in

        Returns:
            The registered class (not an instance, but the class itself)

        Raises:
            AssertionError: If the class is not found in the registry
        """
        assert name in self.registry_dict[register_type], f"{name} not in registry_dict[{register_type}]"
        return self.registry_dict[register_type][name]

    def get_llm_class(self, name: str):
        """Get the LLM class registered under the given name."""
        return self.get_model_class(name, RegistryEnum.LLM)

    def get_embedding_model_class(self, name: str):
        """Get the embedding model class registered under the given name."""
        return self.get_model_class(name, RegistryEnum.EMBEDDING_MODEL)

    def get_vector_store_class(self, name: str):
        """Get the vector store class registered under the given name."""
        return self.get_model_class(name, RegistryEnum.VECTOR_STORE)

    def get_op_class(self, name: str):
        """Get the operation class registered under the given name."""
        return self.get_model_class(name, RegistryEnum.OP)

    def get_flow_class(self, name: str):
        """Get the flow class registered under the given name."""
        return self.get_model_class(name, RegistryEnum.FLOW)

    def get_service_class(self, name: str):
        """Get the service class registered under the given name."""
        return self.get_model_class(name, RegistryEnum.SERVICE)

    def get_token_counter_class(self, name: str):
        """Get the token counter class registered under the given name."""
        return self.get_model_class(name, RegistryEnum.TOKEN_COUNTER)

    def get_llm(self, name: str) -> "BaseLLM":
        """Retrieve a specific LLM instance by name.

        Args:
            name: The name of the LLM instance (typically 'default' or custom name)

        Returns:
            The instantiated LLM object

        Raises:
            KeyError: If no LLM with the given name exists
        """
        return self.instance_dict[RegistryEnum.LLM][name]

    def get_embedding_model(self, name: str) -> "BaseEmbeddingModel":
        """Retrieve a specific embedding model instance by name.

        Args:
            name: The name of the embedding model instance (typically 'default')

        Returns:
            The instantiated embedding model object

        Raises:
            KeyError: If no embedding model with the given name exists
        """
        return self.instance_dict[RegistryEnum.EMBEDDING_MODEL][name]

    def get_vector_store(self, name: str) -> "BaseVectorStore":
        """Retrieve a specific vector store instance by name.

        Args:
            name: The name of the vector store instance (typically 'default')

        Returns:
            The instantiated vector store object

        Raises:
            KeyError: If no vector store with the given name exists
        """
        return self.instance_dict[RegistryEnum.VECTOR_STORE][name]

    def get_token_counter(self, name: str) -> "BaseTokenCounter":
        """Retrieve a specific token counter instance by name.

        Args:
            name: The name of the token counter instance (typically 'default')

        Returns:
            The instantiated token counter object

        Raises:
            KeyError: If no token counter with the given name exists
        """
        return self.instance_dict[RegistryEnum.TOKEN_COUNTER][name]

    def get_flow(self, name: str) -> "BaseFlow":
        """Retrieve a specific flow instance by name.

        Args:
            name: The name of the flow instance

        Returns:
            The instantiated flow object

        Raises:
            KeyError: If no flow with the given name exists
        """
        return self.instance_dict[RegistryEnum.FLOW][name]

    def get_service(self) -> "BaseService":
        """Retrieve the default service instance.

        Returns:
            The instantiated service backend (HTTP, MCP, or CMD service)

        Raises:
            KeyError: If the default service was not initialized
        """
        return self.instance_dict[RegistryEnum.SERVICE]["default"]

    def update_section_config(self, section_name: str, **kwargs):
        """Update a specific section of the service config with new values.

        Args:
            section_name: Name of the config section (e.g., 'llm', 'embedding_model')
            **kwargs: Key-value pairs to update in the default configuration

        Raises:
            KeyError: If the default config for the section doesn't exist

        Example:
            update_section_config('llm', temperature=0.8, max_tokens=1000)
        """
        if not hasattr(self.service_config, section_name) or not kwargs:
            return

        section_dict: dict = getattr(self.service_config, section_name)
        if "default" not in section_dict:
            raise KeyError(f"Default `{section_name}` config not found")

        current_config = section_dict["default"]
        section_dict["default"] = current_config.model_copy(update=kwargs, deep=True)

    def initialize_service_context(self):
        """Initialize the service context with the configuration.

        This is the main initialization method that sets up all system components in order:
        1. Language settings
        2. Thread pool for concurrent operations
        3. Ray cluster (if configured for distributed computing)
        4. LLM instances
        5. Embedding model instances
        6. Token counter instances
        7. Vector store instances (with their embedding models)
        8. Flow instances (both registered and configured)
        9. Service backend instance

        Note: This method should be called after service_config is set.
        This method can only be called once. Subsequent calls will be ignored.
        """
        if self._initialized:
            logger.warning("initialize_service_context has already been called. Skipping re-initialization.")
            return

        self.language = self.service_config.language
        self.thread_pool = ThreadPoolExecutor(max_workers=self.service_config.thread_pool_max_workers)

        # Initialize Ray for distributed computing if configured
        if self.service_config.ray_max_workers > 1:
            import ray

            ray.init(num_cpus=self.service_config.ray_max_workers)

        # Initialize components in dependency order
        self._initialize_llm()
        self._initialize_embedding_model()
        self._initialize_token_counter()
        self._initialize_vector_store()  # Depends on embedding models
        self._initialize_flow()
        self._initialize_service()

        # Mark as initialized
        self._initialized = True

    def _initialize_llm(self):
        """Initialize all configured LLM instances.

        For each LLM configuration:
        - Retrieves the corresponding registered LLM class by backend name
        - Instantiates it with model_name and additional configuration
        - Stores the instance in instance_dict for later retrieval
        """
        for name, config in self.service_config.llm.items():
            llm_cls = self.get_llm_class(config.backend)
            self.instance_dict[RegistryEnum.LLM][name] = llm_cls(model_name=config.model_name, **config.model_extra)

    def _initialize_embedding_model(self):
        """Initialize all configured embedding model instances.

        For each embedding model configuration:
        - Retrieves the corresponding registered embedding model class by backend name
        - Instantiates it with model_name and additional configuration
        - Stores the instance in instance_dict for later retrieval
        """
        for name, config in self.service_config.embedding_model.items():
            embedding_model_cls = self.get_embedding_model_class(config.backend)
            self.instance_dict[RegistryEnum.EMBEDDING_MODEL][name] = embedding_model_cls(
                model_name=config.model_name,
                **config.model_extra,
            )

    def _initialize_token_counter(self):
        """Initialize all configured token counter instances.

        For each token counter configuration:
        - Retrieves the corresponding registered token counter class by backend name
        - Instantiates it with model_name and additional configuration
        - Stores the instance in instance_dict for later retrieval
        """
        for name, config in self.service_config.token_counter.items():
            token_counter_cls = self.get_token_counter_class(config.backend)
            self.instance_dict[RegistryEnum.TOKEN_COUNTER][name] = token_counter_cls(
                model_name=config.model_name,
                **config.model_extra,
            )

    def _initialize_vector_store(self):
        """Initialize all configured vector stores with their embedding models.

        For each vector store configuration:
        - Retrieves the corresponding registered vector store class by backend name
        - Retrieves the associated embedding model instance by name
        - Instantiates the vector store with collection name, embedding model, and extra config
        - Stores the instance in instance_dict for later retrieval

        Note: This must be called after _initialize_embedding_model() since vector stores
        depend on embedding model instances.
        """
        for name, config in self.service_config.vector_store.items():
            vector_store_cls = self.get_vector_store_class(config.backend)
            self.instance_dict[RegistryEnum.VECTOR_STORE][name] = vector_store_cls(
                collection_name=config.collection_name,
                embedding_model=self.instance_dict[RegistryEnum.EMBEDDING_MODEL][config.embedding_model],
                **config.model_extra,
            )

    def _filter_flows(self, name: str) -> bool:
        """Filter flows based on enabled_flows and disabled_flows configuration.

        The filtering logic follows this priority:
        1. If enabled_flows is set: only flows in the list are loaded
        2. Else if disabled_flows is set: all flows except those in the list are loaded
        3. Otherwise: all flows are loaded

        Args:
            name: The flow name to check

        Returns:
            True if the flow should be loaded, False otherwise
        """
        if self.service_config.enabled_flows:
            return name in self.service_config.enabled_flows
        elif self.service_config.disabled_flows:
            return name not in self.service_config.disabled_flows
        else:
            return True

    def _initialize_flow(self):
        """Initialize all flows from both registry and configuration.

        Flows can be defined in two ways:
        1. Registered flows: Python classes decorated with @register_flow
        2. Configuration flows: Defined in config as ExpressionFlow instances

        Process:
        1. First, instantiate all registered flow classes (from decorators)
           - Filter based on enabled_flows/disabled_flows
           - Create instance with the flow name

        2. Then, instantiate all configured flows (from config file)
           - Filter based on enabled_flows/disabled_flows
           - Create ExpressionFlow instances with flow configuration

        Note: Configuration flows can override registered flows with the same name.
        """

        # Initialize flows from registry (decorator-based registration)
        for name, flow_cls in self.registry_dict[RegistryEnum.FLOW].items():
            if not self._filter_flows(name):
                continue
            flow: "BaseFlow" = flow_cls(name=name)
            self.instance_dict[RegistryEnum.FLOW][flow.name] = flow

        # Initialize flows from configuration (config-based definition)
        from ..flow import ExpressionFlow

        for name, flow_config in self.service_config.flow.items():
            if not self._filter_flows(name):
                continue
            flow_config.name = name
            flow: BaseFlow = ExpressionFlow(flow_config=flow_config)
            self.instance_dict[RegistryEnum.FLOW][name] = flow

    def _initialize_service(self):
        """Initialize the service backend instance.

        Creates an instance of the configured service backend (e.g., HTTP, MCP, or CMD service)
        and stores it in the instance dictionary under the 'default' key.
        """
        service_cls = self.get_service_class(self.service_config.backend)
        self.instance_dict[RegistryEnum.SERVICE]["default"] = service_cls()

    async def prepare_mcp_servers(self):
        """Prepare and initialize MCP (Model Context Protocol) server connections.

        This method:
        1. Checks if MCP servers are configured
        2. Creates an MCP client instance
        3. For each configured server:
           - Lists available tool calls from the server
           - Builds a mapping of tool_name -> ToolCall object
           - Logs available tools for debugging

        The mcp_server_mapping is structured as:
        {
            "server_name": {
                "tool_name": ToolCall(...),
                ...
            },
            ...
        }

        This allows the application to discover and use external tools provided by MCP servers.
        """
        if not self.service_config.mcp_servers:
            return

        from ..utils import MCPClient

        mcp_client = MCPClient(config={"mcpServers": self.service_config.mcp_servers})
        for server_name in self.service_config.mcp_servers.keys():
            try:
                # Retrieve all available tool calls from this MCP server
                tool_calls = await mcp_client.list_tool_calls(server_name=server_name, return_dict=False)

                # Build mapping: tool_name -> ToolCall for quick lookup
                self.mcp_server_mapping[server_name] = {tool_call.name: tool_call for tool_call in tool_calls}

                # Log discovered tools for debugging
                for tool_call in tool_calls:
                    logger.info(f"list_tool_calls: {server_name}@{tool_call.name} {tool_call.simple_input_dump()}")

            except Exception as e:
                logger.exception(f"list_tool_calls: {server_name} error: {e}")

    def print_logo(self):
        """Print the ReMe logo if enabled in configuration."""
        if self.service_config.enable_logo:
            print_logo(service_config=self.service_config)

    async def close(self):
        """Close all service components asynchronously.

        Gracefully closes all instantiated components in order:
        1. Vector stores (closes database connections)
        2. LLMs (closes API clients and connections)
        3. Embedding models (closes API clients and connections)

        This method should be called when shutting down the application
        to ensure all resources are properly released.
        """
        for _, vector_store in self.instance_dict[RegistryEnum.VECTOR_STORE].items():
            await vector_store.close()

        for _, llm in self.instance_dict[RegistryEnum.LLM].items():
            await llm.close()

        for _, embedding_model in self.instance_dict[RegistryEnum.EMBEDDING_MODEL].items():
            await embedding_model.close()

    def close_sync(self):
        """Close all service components synchronously.

        Synchronous version of close() for non-async contexts.
        Closes LLMs and embedding models without using async/await.

        Note: Vector stores are not closed here as they typically require async operations.
        """
        for _, llm in self.instance_dict[RegistryEnum.LLM].items():
            llm.close_sync()

        for _, embedding_model in self.instance_dict[RegistryEnum.EMBEDDING_MODEL].items():
            embedding_model.close_sync()

    def shutdown_thread_pool(self, wait: bool = True):
        """Shutdown the thread pool executor.

        Args:
            wait: If True, blocks until all pending futures are executed.
                  If False, returns immediately and pending futures may be cancelled.
        """
        if self.thread_pool:
            self.thread_pool.shutdown(wait=wait)

    def shutdown_ray(self, wait: bool = True):
        """Shutdown Ray cluster if it was initialized.

        Args:
            wait: If True, waits for Ray to fully shutdown.
                  If False, returns immediately without waiting.

        Note: Only shuts down Ray if it was configured with ray_max_workers > 1.
        """
        if self.service_config and self.service_config.ray_max_workers > 1:
            import ray

            ray.shutdown(_exiting_interpreter=not wait)


# Export a global singleton instance for easy access across the application
# This is the primary way to access the service context throughout the codebase
C = ServiceContext()
