"""
Embedder protocol and implementations for CivilEng-Anything.

Provides a unified interface for text embedding with multiple backends.
"""

from __future__ import annotations

import hashlib
import logging
from abc import ABC, abstractmethod
from typing import Protocol, runtime_checkable

logger = logging.getLogger(__name__)


@runtime_checkable
class Embedder(Protocol):
    """Protocol for text embedders."""

    name: str
    dim: int

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of texts, returning list of vectors."""
        ...


class HashingEmbedder:
    """
    Deterministic, dependency-free embedder for TESTS ONLY.

    Uses a simple hashing scheme to generate fixed-dimension vectors.
    Name starts with 'test-hashing' to prevent confusion with real embedders.
    """

    name: str = "test-hashing-v1"
    dim: int = 1024  # Match BGE-M3 dimension

    def __init__(self, dim: int = 1024):
        self.dim = dim
        self.name = f"test-hashing-v1-dim{dim}"

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Generate deterministic pseudo-vectors from text hashes."""
        vectors = []
        for text in texts:
            # Create a deterministic vector from the text
            vec = [0.0] * self.dim
            # Use multiple hash seeds to fill the vector
            for i in range(min(len(text), 100)):
                h = hashlib.md5(f"{text}{i}".encode()).digest()
                for j, b in enumerate(h):
                    idx = (i * 16 + j) % self.dim
                    # Normalize to [-1, 1]
                    vec[idx] += (b - 127.5) / 127.5
            # L2 normalize
            norm = sum(x * x for x in vec) ** 0.5
            if norm > 0:
                vec = [x / norm for x in vec]
            vectors.append(vec)
        return vectors


class SentenceTransformerEmbedder:
    """Sentence-Transformers based embedder (lazy import)."""

    def __init__(self, model_name: str = "BAAI/bge-m3"):
        self._model_name = model_name
        self._model = None
        self._dim = None
        self._name = None

    @property
    def name(self) -> str:
        if self._name is None:
            self._ensure_loaded()
        return self._name

    @property
    def dim(self) -> int:
        if self._dim is None:
            self._ensure_loaded()
        return self._dim

    def _ensure_loaded(self):
        if self._model is not None:
            return
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:
            raise RuntimeError(
                "sentence-transformers not installed. "
                "Install with: pip install sentence-transformers"
            ) from e

        self._model = SentenceTransformer(self._model_name)
        self._dim = self._model.get_sentence_embedding_dimension()
        # Create a fingerprint including model name and revision
        self._name = f"sentence-transformers:{self._model_name}:dim{self._dim}"
        logger.info("Loaded SentenceTransformer model: %s (dim=%d)", self._model_name, self._dim)

    def embed(self, texts: list[str]) -> list[list[float]]:
        self._ensure_loaded()
        embeddings = self._model.encode(texts, normalize_embeddings=True)
        return embeddings.tolist()


class FastEmbedEmbedder:
    """FastEmbed (ONNX) based embedder (lazy import)."""

    def __init__(self, model_name: str = "BAAI/bge-m3"):
        self._model_name = model_name
        self._model = None
        self._dim = None
        self._name = None

    @property
    def name(self) -> str:
        if self._name is None:
            self._ensure_loaded()
        return self._name

    @property
    def dim(self) -> int:
        if self._dim is None:
            self._ensure_loaded()
        return self._dim

    def _ensure_loaded(self):
        if self._model is not None:
            return
        try:
            from fastembed import TextEmbedding
        except ImportError as e:
            raise RuntimeError(
                "fastembed not installed. Install with: pip install fastembed"
            ) from e

        self._model = TextEmbedding(model_name=self._model_name)
        # Get dimension by embedding a test string
        test_vec = list(self._model.embed(["test"]))[0]
        self._dim = len(test_vec)
        self._name = f"fastembed:{self._model_name}:dim{self._dim}"
        logger.info("Loaded FastEmbed model: %s (dim=%d)", self._model_name, self._dim)

    def embed(self, texts: list[str]) -> list[list[float]]:
        self._ensure_loaded()
        return list(self._model.embed(texts))


def get_embedder(settings) -> Embedder:
    """
    Get the first available real embedder backend.

    Priority: fastembed > sentence-transformers
    Raises clear error if neither is installed.
    Never silently falls back to HashingEmbedder.
    """
    model_name = getattr(settings, "embed_model", "BAAI/bge-m3")

    # Try fastembed first (preferred on CPU-only machines)
    try:
        embedder = FastEmbedEmbedder(model_name)
        # Test it works
        _ = embedder.dim
        logger.info("Using FastEmbed embedder: %s", embedder.name)
        return embedder
    except RuntimeError:
        logger.debug("FastEmbed not available, trying sentence-transformers")

    # Try sentence-transformers
    try:
        embedder = SentenceTransformerEmbedder(model_name)
        _ = embedder.dim
        logger.info("Using SentenceTransformer embedder: %s", embedder.name)
        return embedder
    except RuntimeError:
        logger.debug("SentenceTransformer not available")

    raise RuntimeError(
        "No embedding backend available. Install one of:\n"
        "  pip install fastembed\n"
        "  pip install sentence-transformers"
    )