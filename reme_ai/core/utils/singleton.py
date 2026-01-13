"""Module providing a decorator to implement the Singleton design pattern."""


def singleton(cls):
    """A class decorator that ensures only one instance of a class exists."""

    # Dictionary to cache the single instance of the class
    _instance = {}

    def _singleton(*args, **kwargs):
        """Return the existing instance or create a new one if it doesn't exist."""
        if cls not in _instance:
            # Create and store the instance if it's the first call
            _instance[cls] = cls(*args, **kwargs)
        return _instance[cls]

    return _singleton
