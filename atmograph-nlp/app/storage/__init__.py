"""Local persistence backends for normalized articles."""

from app.storage.json_storage import (
    ArticleStorage,
    JsonArticleStorage,
    StorageError,
    StorageWriteResult,
)

__all__ = [
    "ArticleStorage",
    "JsonArticleStorage",
    "StorageError",
    "StorageWriteResult",
]
