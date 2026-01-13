"""Module providing a registry class for managing class-to-name mappings via decorators."""

from .base_context import BaseContext


class Registry(BaseContext):
    """A registry container that uses decorators to map and store class references."""

    def register(self, name: str = "", add_cls: bool = True):
        """Return a decorator that registers a class under a specific name in the registry."""

        def decorator(cls):
            if add_cls:
                # Use provided name or default to the class name as the key
                key = name or cls.__name__
                self[key] = cls
            return cls

        return decorator
