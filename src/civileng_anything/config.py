"""
Configuration management for CivilEng-Anything.
"""

from dataclasses import dataclass, field
from pathlib import Path
import os


@dataclass
class Settings:
    """Application settings loaded from environment variables with defaults."""

    # Base data directory
    data_dir: Path = field(default_factory=lambda: Path("./data").resolve())

    # Embedding model for semantic search
    embed_model: str = "BAAI/bge-m3"

    # Retrieval defaults
    top_k: int = 8
    token_budget: int = 6000

    # API server
    host: str = "127.0.0.1"
    api_port: int = 8765

    # Derived paths (computed from data_dir)
    originals_dir: Path = field(init=False)
    parsed_dir: Path = field(init=False)
    assets_dir: Path = field(init=False)
    db_path: Path = field(init=False)
    indexes_dir: Path = field(init=False)

    def __post_init__(self):
        self.data_dir = self.data_dir.resolve()
        self.originals_dir = self.data_dir / "originals"
        self.parsed_dir = self.data_dir / "parsed"
        self.assets_dir = self.data_dir / "assets"
        self.db_path = self.data_dir / "civileng.db"
        self.indexes_dir = self.data_dir / "indexes"

    def ensure_dirs(self) -> None:
        """Create all required directories if they don't exist."""
        for d in (
            self.data_dir,
            self.originals_dir,
            self.parsed_dir,
            self.assets_dir,
            self.indexes_dir,
        ):
            d.mkdir(parents=True, exist_ok=True)


def get_settings() -> Settings:
    """Load settings from environment variables."""
    return Settings(
        data_dir=Path(os.getenv("CIVILENG_DATA_DIR", "./data")).resolve(),
        embed_model=os.getenv("CIVILENG_EMBED_MODEL", "BAAI/bge-m3"),
        top_k=int(os.getenv("CIVILENG_TOP_K", "8")),
        token_budget=int(os.getenv("CIVILENG_TOKEN_BUDGET", "6000")),
        host=os.getenv("CIVILENG_HOST", "127.0.0.1"),
        api_port=int(os.getenv("CIVILENG_API_PORT", "8765")),
    )