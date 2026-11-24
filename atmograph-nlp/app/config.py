"""Central configuration for the AtmoGraph NLP module.

All settings are read from environment variables (prefix ``ATMOGRAPH_``) or
from a local ``.env`` file.  Values are validated by Pydantic so a bad port
number or log level fails fast at start-up instead of at request time.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# ``app/config.py`` -> ``app`` -> module root (atmograph-nlp/)
MODULE_ROOT = Path(__file__).resolve().parent.parent

LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
Environment = Literal["development", "testing", "production"]


class Settings(BaseSettings):
    """Runtime configuration for the NLP & News Intelligence service."""

    model_config = SettingsConfigDict(
        env_file=MODULE_ROOT / ".env",
        env_file_encoding="utf-8",
        env_prefix="ATMOGRAPH_",
        case_sensitive=False,
        extra="ignore",
    )

    # --- Application -------------------------------------------------------
    app_name: str = "AtmoGraph NLP & News Intelligence"
    app_version: str = "0.1.0"
    environment: Environment = "development"
    debug: bool = False
    api_prefix: str = "/api"
    host: str = "127.0.0.1"
    port: int = Field(default=8000, ge=1, le=65535)

    # Comma-separated in .env; exposed as a list via ``cors_origin_list``.
    cors_origins: str = "http://localhost:5173,http://localhost:3000"

    # --- Logging -----------------------------------------------------------
    log_level: LogLevel = "INFO"
    log_dir: Path = MODULE_ROOT / "logs"
    log_file_name: str = "atmograph-nlp.log"
    log_max_bytes: int = Field(default=5_000_000, ge=100_000)
    log_backup_count: int = Field(default=3, ge=1, le=20)

    # --- Data / storage ----------------------------------------------------
    data_dir: Path = MODULE_ROOT / "data"
    sample_news_path: Path = MODULE_ROOT / "data" / "sample_news.json"
    mock_news_path: Path = MODULE_ROOT / "data" / "mock_news.json"
    articles_store_path: Path = MODULE_ROOT / "data" / "articles.json"
    processed_store_path: Path = MODULE_ROOT / "data" / "processed_articles.json"

    # --- NLP models --------------------------------------------------------
    spacy_model: str = "en_core_web_sm"
    transformer_model: str = "distilbert-base-uncased"
    enable_transformer: bool = False
    transformer_device: str = "cpu"
    classifier_min_confidence: float = Field(default=0.50, ge=0.0, le=1.0)

    # --- News ingestion ----------------------------------------------------
    # Comma-separated list of public RSS URLs (``Name|URL`` is also accepted);
    # exposed as a list via ``rss_feed_list``. No paid APIs or keys are used.
    rss_feeds: str = (
        "https://feeds.bbci.co.uk/news/business/rss.xml,"
        "https://feeds.bbci.co.uk/news/world/rss.xml,"
        "https://www.supplychaindive.com/feeds/news/,"
        "https://www.manufacturingdive.com/feeds/news/,"
        "https://gcaptain.com/feed/,"
        "https://theloadstar.com/feed/,"
        "https://splash247.com/feed/,"
        "https://container-news.com/feed/,"
        "https://www.freightwaves.com/feed"
    )
    rss_timeout_seconds: float = Field(default=10.0, gt=0, le=120)
    rss_max_items_per_feed: int = Field(default=25, ge=1, le=200)
    news_request_timeout_seconds: float = Field(default=15.0, gt=0, le=120)
    news_user_agent: str = "AtmoGraph-NLP/0.1 (+college project)"
    article_max_chars: int = Field(default=20_000, ge=500, le=200_000)

    # --- Member 1 backend / graph sync -------------------------------------
    # TEMPORARY contract: the NLP module calls Member 1's FastAPI backend.
    # Update these values (and the paths in integration/backend_client.py)
    # once the team agrees on the final API contract.
    backend_base_url: str = "http://127.0.0.1:8001"
    backend_timeout_seconds: float = Field(default=10.0, gt=0, le=120)
    backend_max_retries: int = Field(default=3, ge=1, le=10)
    backend_retry_backoff_seconds: float = Field(default=1.0, ge=0, le=30)
    backend_api_key: SecretStr | None = None
    graph_sync_enabled: bool = False

    # Optional direct Neo4j access (only used if Member 1 exposes it).
    neo4j_uri: str = "bolt://127.0.0.1:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: SecretStr | None = None
    neo4j_database: str = "neo4j"

    # --- Limits ------------------------------------------------------------
    max_batch_size: int = Field(default=50, ge=1, le=500)

    # --- Derived helpers ---------------------------------------------------
    @field_validator(
        "log_dir",
        "data_dir",
        "sample_news_path",
        "mock_news_path",
        "articles_store_path",
        "processed_store_path",
        mode="after",
    )
    @classmethod
    def _resolve_relative_paths(cls, value: Path) -> Path:
        """Resolve relative paths against the module root, not the CWD."""
        return value if value.is_absolute() else (MODULE_ROOT / value).resolve()

    @property
    def cors_origin_list(self) -> list[str]:
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]

    @property
    def rss_feed_list(self) -> list[str]:
        return [feed.strip() for feed in self.rss_feeds.split(",") if feed.strip()]

    @property
    def is_testing(self) -> bool:
        return self.environment == "testing"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide settings singleton (cached)."""
    return Settings()
