"""Train-only demonstration retrieval for FinPRM inference."""

from .index import (
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_EMBEDDING_REVISION,
    DEFAULT_K,
    RETRIEVAL_MODES,
    DemonstrationRetriever,
    build_index,
    format_demonstrations,
)

__all__ = [
    "DEFAULT_EMBEDDING_MODEL",
    "DEFAULT_EMBEDDING_REVISION",
    "DEFAULT_K",
    "RETRIEVAL_MODES",
    "DemonstrationRetriever",
    "build_index",
    "format_demonstrations",
]
