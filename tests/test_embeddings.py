"""
Tests for embeddings module (Brief 03).
"""

from __future__ import annotations

import pytest
from unittest.mock import patch, MagicMock

from civileng_anything.embeddings import (
    Embedder,
    HashingEmbedder,
    SentenceTransformerEmbedder,
    FastEmbedEmbedder,
    get_embedder,
)
from civileng_anything.config import Settings


class TestHashingEmbedder:
    """Tests for the deterministic test embedder."""

    def test_name_starts_with_test_hashing(self):
        """HashingEmbedder name must start with 'test-hashing' for test isolation."""
        embedder = HashingEmbedder()
        assert embedder.name.startswith("test-hashing")

    def test_dim_default(self):
        """Default dimension should match BGE-M3 (1024)."""
        embedder = HashingEmbedder()
        assert embedder.dim == 1024

    def test_custom_dim(self):
        """Custom dimension should be respected."""
        embedder = HashingEmbedder(dim=512)
        assert embedder.dim == 512
        assert embedder.name == "test-hashing-v1-dim512"

    def test_embed_returns_correct_shape(self):
        """embed() should return list of vectors with correct dimensions."""
        embedder = HashingEmbedder(dim=128)
        texts = ["hello world", "test text", "another example"]
        vectors = embedder.embed(texts)

        assert len(vectors) == 3
        assert all(len(v) == 128 for v in vectors)

    def test_embed_deterministic(self):
        """Same input should produce same output."""
        embedder = HashingEmbedder(dim=64)
        texts = ["deterministic test"]
        v1 = embedder.embed(texts)
        v2 = embedder.embed(texts)
        assert v1 == v2

    def test_embed_normalized(self):
        """Vectors should be L2 normalized."""
        embedder = HashingEmbedder(dim=32)
        vectors = embedder.embed(["test normalization"])
        vec = vectors[0]
        norm = sum(x * x for x in vec) ** 0.5
        assert abs(norm - 1.0) < 1e-5

    def test_embed_empty_list(self):
        """Empty input should return empty list."""
        embedder = HashingEmbedder()
        vectors = embedder.embed([])
        assert vectors == []

    def test_protocol_compliance(self):
        """HashingEmbedder should satisfy Embedder protocol."""
        embedder: Embedder = HashingEmbedder()
        assert hasattr(embedder, "name")
        assert hasattr(embedder, "dim")
        assert callable(getattr(embedder, "embed"))


class TestSentenceTransformerEmbedder:
    """Tests for SentenceTransformerEmbedder (lazy import)."""

    def test_lazy_import(self):
        """Model should not load until accessed."""
        embedder = SentenceTransformerEmbedder("BAAI/bge-base-en-v1.5")
        assert embedder._model is None
        assert embedder._dim is None
        assert embedder._name is None

    def test_name_property_triggers_load(self):
        """Accessing name should trigger model load."""
        embedder = SentenceTransformerEmbedder("BAAI/bge-base-en-v1.5")
        try:
            name = embedder.name
            assert "sentence-transformers" in name
            assert "BAAI/bge-base-en-v1.5" in name
            assert "dim" in name
        except RuntimeError as e:
            # Expected if sentence-transformers not installed
            assert "sentence-transformers not installed" in str(e)

    def test_dim_property_triggers_load(self):
        """Accessing dim should trigger model load."""
        embedder = SentenceTransformerEmbedder("BAAI/bge-base-en-v1.5")
        try:
            dim = embedder.dim
            assert isinstance(dim, int)
            assert dim > 0
        except RuntimeError as e:
            assert "sentence-transformers not installed" in str(e)

    def test_embed_triggers_load(self):
        """Calling embed should trigger model load."""
        embedder = SentenceTransformerEmbedder("BAAI/bge-base-en-v1.5")
        try:
            vectors = embedder.embed(["test"])
            assert len(vectors) == 1
            assert len(vectors[0]) == embedder.dim
        except RuntimeError as e:
            assert "sentence-transformers not installed" in str(e)


class TestFastEmbedEmbedder:
    """Tests for FastEmbedEmbedder (lazy import)."""

    def test_lazy_import(self):
        """Model should not load until accessed."""
        embedder = FastEmbedEmbedder("BAAI/bge-base-en-v1.5")
        assert embedder._model is None
        assert embedder._dim is None
        assert embedder._name is None

    def _make_fastembed_mock(self, mock_text_embedding, dim=768):
        """Helper to create a mock that returns fresh iterators on each embed call."""
        mock_model = MagicMock()
        # Use side_effect to return a fresh iterator each time embed is called
        def embed_side_effect(texts):
            return iter([[0.1] * dim for _ in texts])
        mock_model.embed.side_effect = embed_side_effect
        mock_text_embedding.return_value = mock_model
        return mock_model

    @patch("fastembed.TextEmbedding")
    def test_name_property_triggers_load(self, mock_text_embedding):
        """Accessing name should trigger model load."""
        self._make_fastembed_mock(mock_text_embedding)

        embedder = FastEmbedEmbedder("BAAI/bge-base-en-v1.5")
        name = embedder.name

        assert "fastembed" in name
        assert "BAAI/bge-base-en-v1.5" in name
        assert "dim768" in name
        mock_text_embedding.assert_called_once_with(model_name="BAAI/bge-base-en-v1.5")

    @patch("fastembed.TextEmbedding")
    def test_dim_property_triggers_load(self, mock_text_embedding):
        """Accessing dim should trigger model load."""
        self._make_fastembed_mock(mock_text_embedding)

        embedder = FastEmbedEmbedder("BAAI/bge-base-en-v1.5")
        dim = embedder.dim

        assert dim == 768
        mock_text_embedding.assert_called_once_with(model_name="BAAI/bge-base-en-v1.5")

    @patch("fastembed.TextEmbedding")
    def test_embed_triggers_load(self, mock_text_embedding):
        """Calling embed should trigger model load."""
        self._make_fastembed_mock(mock_text_embedding)

        embedder = FastEmbedEmbedder("BAAI/bge-base-en-v1.5")
        vectors = embedder.embed(["test1", "test2"])

        assert len(vectors) == 2
        assert len(vectors[0]) == 768
        assert len(vectors[1]) == 768
        mock_text_embedding.assert_called_once_with(model_name="BAAI/bge-base-en-v1.5")


class TestGetEmbedder:
    """Tests for get_embedder factory function."""

    @patch("fastembed.TextEmbedding", side_effect=RuntimeError("fastembed not installed"))
    def test_returns_hashing_when_no_backend(self, mock_fe, tmp_path):
        """Should raise RuntimeError when no real backend available (not silently fall back).
        
        sentence_transformers is not installed in test environment, so it naturally fails.
        """
        settings = Settings(data_dir=tmp_path)
        settings.embed_model = "unsupported-model-xyz"

        with pytest.raises(RuntimeError) as exc:
            get_embedder(settings)
        assert "No embedding backend available" in str(exc.value)
        assert "fastembed" in str(exc.value)
        assert "sentence-transformers" in str(exc.value)

    @patch("fastembed.TextEmbedding")
    def test_prefers_fastembed(self, mock_text_embedding, tmp_path):
        """Should prefer fastembed when available."""
        settings = Settings(data_dir=tmp_path)

        # Setup mock
        mock_model = MagicMock()
        def embed_side_effect(texts):
            return iter([[0.1] * 768 for _ in texts])
        mock_model.embed.side_effect = embed_side_effect
        mock_text_embedding.return_value = mock_model

        embedder = get_embedder(settings)
        assert "fastembed" in embedder.name
        assert "BAAI/bge-base-en-v1.5" in embedder.name
        mock_text_embedding.assert_called_once_with(model_name="BAAI/bge-base-en-v1.5")


class TestEmbedderProtocol:
    """Tests for Embedder protocol compliance."""

    def test_hashing_embeds_list_of_lists(self):
        """embed() returns list[list[float]]."""
        embedder = HashingEmbedder(dim=16)
        result = embedder.embed(["a", "b", "c"])
        assert isinstance(result, list)
        assert all(isinstance(v, list) for v in result)
        assert all(isinstance(x, float) for v in result for x in v)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])