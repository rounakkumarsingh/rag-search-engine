class IndexNotFoundError(Exception):
    """Raised when a cached index is missing or cannot be loaded."""


class CacheInvalidError(Exception):
    """Raised when a cache artifact is stale, partial, or misaligned."""


class EmptyQueryError(ValueError):
    """Raised when a query is empty after preprocessing."""


class GenerationError(RuntimeError):
    """Raised when an LLM call fails or returns no usable content."""
