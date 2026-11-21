"""
Tests for vector index module (Brief 03).
"""

from __future__ import annotations

import pytest
import numpy as np

from civileng_anything.index import VectorIndex
from civileng_anything.store import Store
from civileng_anything.embeddings import HashingEmbedder
from civileng_anything.config import Settings
from civileng_anything.schema import (
    ItemType, DocType, DocumentMeta, ContentItem, ContentOrigin, ItemStatus
)
from datetime import date


@pytest.fixture
def temp_db(tmp_path):
    """Create a temporary database with test data."""
    settings = Settings(data_dir=tmp_path)
    db_path = settings.db_path
    store = Store(db_path)

    # Create a test document
    doc = DocumentMeta(
        document_id="TEST-001",
        document_family_id="TEST-001",
        edition_id="2024",
        title="Test Document",
        publisher="Test Publisher",
        jurisdiction="US",
        doc_type=DocType.GOVERNMENT_CODE,
        language="en",
        publication_date=date(2024, 1, 1),
        revision_date=date(2024, 1, 1),
        source_url="http://example.com",
        retrieved_at=date(2024, 1, 1),
        original_sha256="abc123",
        total_pdf_pages=10,
        parser_name="test",
        parser_version="1.0",
        parser_config={},
        parse_run_id="run1",
        completeness=1.0,
        qc_status="done",
        amendment_relations=[],
        version_relations=[],
    )
    store.upsert_document(doc)

    # Add some test items
    items = [
        ContentItem(
            item_id="TEST-001:2024:section:1",
            document_id="TEST-001",
            edition_id="2024",
            type=ItemType.SECTION,
            number="1",
            text="This section covers general requirements for structural design.",
            section_path=["1"],
            pdf_page=1,
            printed_page="1",
            parent_id=None,
            asset_ids=[],
            cross_refs=[],
            footnote_ids=[],
            amendment_links=[],
            content_origin=ContentOrigin.ORIGINAL,
            status=ItemStatus.CHECKED,
        ),
        ContentItem(
            item_id="TEST-001:2024:clause:1.1",
            document_id="TEST-001",
            edition_id="2024",
            type=ItemType.CLAUSE,
            number="1.1",
            text="This clause defines the scope of the standard for steel structures.",
            section_path=["1"],
            pdf_page=1,
            printed_page="1",
            parent_id="TEST-001:2024:section:1",
            asset_ids=[],
            cross_refs=[],
            footnote_ids=[],
            amendment_links=[],
            content_origin=ContentOrigin.ORIGINAL,
            status=ItemStatus.CHECKED,
        ),
        ContentItem(
            item_id="TEST-001:2024:clause:1.2",
            document_id="TEST-001",
            edition_id="2024",
            type=ItemType.CLAUSE,
            number="1.2",
            text="Steel materials shall conform to ASTM A36 or equivalent.",
            section_path=["1"],
            pdf_page=1,
            printed_page="1",
            parent_id="TEST-001:2024:section:1",
            asset_ids=[],
            cross_refs=[],
            footnote_ids=[],
            amendment_links=[],
            content_origin=ContentOrigin.ORIGINAL,
            status=ItemStatus.CHECKED,
        ),
        ContentItem(
            item_id="TEST-001:2024:table:1",
            document_id="TEST-001",
            edition_id="2024",
            type=ItemType.TABLE,
            number="1",
            text="Table 1 shows material properties for structural steel.",
            section_path=["1"],
            pdf_page=2,
            printed_page="2",
            parent_id="TEST-001:2024:section:1",
            asset_ids=[],
            cross_refs=[],
            footnote_ids=[],
            amendment_links=[],
            content_origin=ContentOrigin.ORIGINAL,
            status=ItemStatus.CHECKED,
        ),
    ]

    store.add_items(items)

    store.close()
    return db_path


class TestVectorIndex:
    """Tests for VectorIndex class."""

    def test_init_creates_tables(self, temp_db):
        """Initializing VectorIndex should create tables."""
        index = VectorIndex(temp_db)
        # Tables should exist - verify by checking meta
        meta = index.get_index_meta("test-embedder")
        assert meta is None  # No vectors yet

    def test_has_vectors_false_initially(self, temp_db):
        """has_vectors should return False for unknown embedder."""
        index = VectorIndex(temp_db)
        assert index.has_vectors("unknown-embedder") is False

    def test_build_vectors_creates_embeddings(self, temp_db):
        """build_vectors should create embeddings for all items."""
        store = Store(temp_db)
        index = VectorIndex(temp_db)
        embedder = HashingEmbedder(dim=64)

        result = index.build_vectors(store, embedder)

        assert result["built_count"] == 4
        assert result["total_items"] == 4
        assert result["skipped_count"] == 0
        assert index.has_vectors(embedder.name)

        meta = index.get_index_meta(embedder.name)
        assert meta is not None
        assert meta["item_count"] == 4
        assert meta["dim"] == 64

        store.close()

    def test_build_vectors_resumable(self, temp_db):
        """build_vectors should be resumable (only fill gaps)."""
        store = Store(temp_db)
        index = VectorIndex(temp_db)
        embedder = HashingEmbedder(dim=64)

        # First build
        result1 = index.build_vectors(store, embedder)
        assert result1["built_count"] == 4

        # Second build should skip all
        result2 = index.build_vectors(store, embedder)
        assert result2["built_count"] == 0
        assert result2["skipped_count"] == 0

        store.close()

    def test_build_vectors_with_document_filter(self, temp_db):
        """build_vectors should support document_id filter."""
        store = Store(temp_db)
        index = VectorIndex(temp_db)
        embedder = HashingEmbedder(dim=64)

        result = index.build_vectors(store, embedder, document_id="TEST-001")
        assert result["built_count"] == 4

        # Build with non-existent document
        result2 = index.build_vectors(store, embedder, document_id="NONEXISTENT")
        assert result2["built_count"] == 0

        store.close()

    def test_vector_search_returns_results(self, temp_db):
        """vector_search should return ranked results."""
        store = Store(temp_db)
        index = VectorIndex(temp_db)
        embedder = HashingEmbedder(dim=64)

        index.build_vectors(store, embedder)

        results = index.vector_search(store, embedder, "steel structures", top_k=2)

        assert len(results) <= 2
        assert all("item_id" in r and "score" in r for r in results)
        # Scores should be in descending order
        scores = [r["score"] for r in results]
        assert scores == sorted(scores, reverse=True)

        store.close()

    def test_vector_search_empty_without_vectors(self, temp_db):
        """vector_search should return empty list if no vectors built."""
        store = Store(temp_db)
        index = VectorIndex(temp_db)
        embedder = HashingEmbedder(dim=64)

        results = index.vector_search(store, embedder, "steel structures", top_k=5)
        assert results == []

        store.close()

    def test_vector_search_with_filters(self, temp_db):
        """vector_search should respect document_id filter."""
        store = Store(temp_db)
        index = VectorIndex(temp_db)
        embedder = HashingEmbedder(dim=64)

        index.build_vectors(store, embedder)

        # Search with matching document_id
        results = index.vector_search(store, embedder, "steel", document_id="TEST-001", top_k=5)
        assert len(results) > 0

        # Search with non-matching document_id
        results = index.vector_search(store, embedder, "steel", document_id="NONEXISTENT", top_k=5)
        assert results == []

        store.close()

    def test_delete_vectors(self, temp_db):
        """delete_vectors should remove all vectors for an embedder."""
        store = Store(temp_db)
        index = VectorIndex(temp_db)
        embedder = HashingEmbedder(dim=64)

        index.build_vectors(store, embedder)
        assert index.has_vectors(embedder.name)

        index.delete_vectors(embedder.name)
        assert not index.has_vectors(embedder.name)
        assert index.get_index_meta(embedder.name) is None

        store.close()

    def test_namespace_isolation(self, temp_db):
        """Different embedders should have separate namespaces."""
        store = Store(temp_db)
        index = VectorIndex(temp_db)
        embedder1 = HashingEmbedder(dim=64)
        embedder2 = HashingEmbedder(dim=128)

        index.build_vectors(store, embedder1)
        index.build_vectors(store, embedder2)

        assert index.has_vectors(embedder1.name)
        assert index.has_vectors(embedder2.name)

        meta1 = index.get_index_meta(embedder1.name)
        meta2 = index.get_index_meta(embedder2.name)

        assert meta1["dim"] == 64
        assert meta2["dim"] == 128

        # Vectors for embedder1 should not be searchable with embedder2
        results1 = index.vector_search(store, embedder1, "steel", top_k=5)
        results2 = index.vector_search(store, embedder2, "steel", top_k=5)

        # Both should work but return results from their own namespace
        assert len(results1) > 0
        assert len(results2) > 0

        store.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])