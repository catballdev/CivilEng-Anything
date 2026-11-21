"""
Tests for retrieval module (Brief 03).
"""

from __future__ import annotations

import pytest
import json

from civileng_anything.retrieval import (
    list_documents,
    get_document_manifest,
    search,
    get_section,
    get_clause,
    get_equation,
    get_table,
    get_figure,
    get_page,
    get_related,
    SearchMode,
)
from civileng_anything.store import Store
from civileng_anything.embeddings import HashingEmbedder
from civileng_anything.config import Settings
from civileng_anything.schema import (
    DocumentMeta,
    ItemType,
    DocType,
    ItemStatus,
    ContentOrigin,
    ContentItem,
)
from datetime import date


@pytest.fixture
def populated_store(tmp_path):
    """Create a store with test documents and items."""
    settings = Settings(data_dir=tmp_path)
    store = Store(settings.db_path)

    # Create baseline document
    doc = DocumentMeta(
        document_id="STD-001",
        document_family_id="STD-001",
        edition_id="2024",
        title="Steel Design Standard",
        publisher="Test Standards Org",
        jurisdiction="US",
        doc_type=DocType.GOVERNMENT_CODE,
        language="en",
        publication_date=date(2024, 1, 1),
        revision_date=date(2024, 1, 1),
        source_url="http://example.com/std-001",
        retrieved_at=date(2024, 1, 1),
        original_sha256="abc123",
        total_pdf_pages=10,
        parser_name="test",
        parser_version="1.0",
        parser_config={},
        parse_run_id="run1",
        completeness=1.0,
        qc_status="done",
        amendment_relations=[{"document_id": "AMD-001", "relation_type": "amends"}],
        version_relations=[],
    )
    store.upsert_document(doc)

    # Create amendment document
    am_doc = DocumentMeta(
        document_id="AMD-001",
        document_family_id="STD-001",
        edition_id="2025",
        title="Amendment 1 to Steel Design Standard",
        publisher="Test Standards Org",
        jurisdiction="US",
        doc_type=DocType.AMENDMENT,
        language="en",
        publication_date=date(2025, 1, 1),
        revision_date=date(2025, 1, 1),
        source_url="http://example.com/amd-001",
        retrieved_at=date(2025, 1, 1),
        original_sha256="def456",
        total_pdf_pages=5,
        parser_name="test",
        parser_version="1.0",
        parser_config={},
        parse_run_id="run2",
        completeness=1.0,
        qc_status="done",
        amendment_relations=[],
        version_relations=[],
    )
    store.upsert_document(am_doc)

    # Add items to baseline document
    items = [
        # Section 1
        ContentItem(
            item_id="STD-001:2024:section:1",
            document_id="STD-001",
            edition_id="2024",
            type=ItemType.SECTION,
            number="1",
            text="This section covers general requirements for structural steel design.",
            section_path=["1"],
            pdf_page=1,
            printed_page="1",
            parent_id=None,
            asset_ids=[],
            cross_refs=["Clause 1.1", "Table 1"],
            footnote_ids=[],
            amendment_links=[
                {"document_id": "AMD-001", "item_id": "AMD-001:2025:clause:1.1", "status": "verified"}
            ],
            content_origin=ContentOrigin.ORIGINAL,
            status=ItemStatus.CHECKED,
        ),
        # Clause 1.1
        ContentItem(
            item_id="STD-001:2024:clause:1.1",
            document_id="STD-001",
            edition_id="2024",
            type=ItemType.CLAUSE,
            number="1.1",
            text="This clause defines the scope of the standard for steel structures.",
            section_path=["1"],
            pdf_page=1,
            printed_page="1",
            parent_id="STD-001:2024:section:1",
            asset_ids=[],
            cross_refs=[],
            footnote_ids=["STD-001:2024:footnote:1"],
            amendment_links=[
                {"document_id": "AMD-001", "item_id": "AMD-001:2025:clause:1.1", "status": "verified"}
            ],
            content_origin=ContentOrigin.ORIGINAL,
            status=ItemStatus.CHECKED,
        ),
        # Clause 1.2
        ContentItem(
            item_id="STD-001:2024:clause:1.2",
            document_id="STD-001",
            edition_id="2024",
            type=ItemType.CLAUSE,
            number="1.2",
            text="Steel materials shall conform to ASTM A36 or equivalent grade 50 steel.",
            section_path=["1"],
            pdf_page=1,
            printed_page="1",
            parent_id="STD-001:2024:section:1",
            asset_ids=[],
            cross_refs=["Table 1"],
            footnote_ids=[],
            amendment_links=[],
            content_origin=ContentOrigin.ORIGINAL,
            status=ItemStatus.CHECKED,
        ),
        # Table 1
        ContentItem(
            item_id="STD-001:2024:table:1",
            document_id="STD-001",
            edition_id="2024",
            type=ItemType.TABLE,
            number="1",
            text="Table 1 shows material properties for structural steel grades.",
            section_path=["1"],
            pdf_page=2,
            printed_page="2",
            parent_id="STD-001:2024:section:1",
            asset_ids=[],
            cross_refs=[],
            footnote_ids=["STD-001:2024:footnote:2"],
            amendment_links=[],
            content_origin=ContentOrigin.ORIGINAL,
            status=ItemStatus.CHECKED,
        ),
        # Footnote 1
        ContentItem(
            item_id="STD-001:2024:footnote:1",
            document_id="STD-001",
            edition_id="2024",
            type=ItemType.FOOTNOTE,
            number="1",
            text="Footnote 1: Refer to ASTM A6 for general requirements.",
            section_path=["1"],
            pdf_page=1,
            printed_page="1",
            parent_id="STD-001:2024:clause:1.1",
            asset_ids=[],
            cross_refs=[],
            footnote_ids=[],
            amendment_links=[],
            content_origin=ContentOrigin.ORIGINAL,
            status=ItemStatus.CHECKED,
        ),
        # Footnote 2
        ContentItem(
            item_id="STD-001:2024:footnote:2",
            document_id="STD-001",
            edition_id="2024",
            type=ItemType.FOOTNOTE,
            number="2",
            text="Footnote 2: Yield strength values are minimums.",
            section_path=["1"],
            pdf_page=2,
            printed_page="2",
            parent_id="STD-001:2024:table:1",
            asset_ids=[],
            cross_refs=[],
            footnote_ids=[],
            amendment_links=[],
            content_origin=ContentOrigin.ORIGINAL,
            status=ItemStatus.CHECKED,
        ),
        # Figure 1
        ContentItem(
            item_id="STD-001:2024:figure:1",
            document_id="STD-001",
            edition_id="2024",
            type=ItemType.FIGURE,
            number="1",
            text="Figure 1 shows the typical stress-strain curve for structural steel.",
            section_path=["1"],
            pdf_page=3,
            printed_page="3",
            parent_id="STD-001:2024:section:1",
            asset_ids=["asset-fig1"],
            cross_refs=[],
            footnote_ids=[],
            amendment_links=[],
            content_origin=ContentOrigin.ORIGINAL,
            status=ItemStatus.CHECKED,
        ),
        # Equation 1.1
        ContentItem(
            item_id="STD-001:2024:equation:1.1",
            document_id="STD-001",
            edition_id="2024",
            type=ItemType.EQUATION,
            number="1.1",
            text="φRn = φ × Fy × Zx",
            section_path=["1"],
            pdf_page=2,
            printed_page="2",
            parent_id="STD-001:2024:section:1",
            asset_ids=[],
            cross_refs=[],
            footnote_ids=[],
            amendment_links=[],
            content_origin=ContentOrigin.ORIGINAL,
            status=ItemStatus.CHECKED,
        ),
    ]

    store.add_items(items)

    # Add items to amendment document
    am_items = [
        ContentItem(
            item_id="AMD-001:2025:clause:1.1",
            document_id="AMD-001",
            edition_id="2025",
            type=ItemType.CLAUSE,
            number="1.1",
            text="This clause defines the scope for steel and composite structures.",
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
    ]

    store.add_items(am_items)

    store.close()
    return store


class TestListDocuments:
    """Tests for list_documents."""

    def test_list_documents(self, populated_store):
        """Should list all documents."""
        docs = list_documents(populated_store)
        assert len(docs) == 2
        doc_ids = {d.document_id for d in docs}
        assert doc_ids == {"STD-001", "AMD-001"}


class TestGetDocumentManifest:
    """Tests for get_document_manifest."""

    def test_manifest_exists(self, populated_store):
        """Should return manifest for existing document."""
        manifest = get_document_manifest(populated_store, "STD-001")
        assert manifest is not None
        assert manifest["document"].document_id == "STD-001"
        assert manifest["document"].edition_id == "2024"

    def test_manifest_page_counts(self, populated_store):
        """Manifest should include page status counts."""
        manifest = get_document_manifest(populated_store, "STD-001")
        counts = manifest["page_counts"]
        assert counts["total"] == 10
        assert "done" in counts
        assert "failed" in counts
        assert "retry_pending" in counts
        assert "needs_review" in counts
        assert "pending" in counts

    def test_manifest_amendments(self, populated_store):
        """Manifest should include amendment relations."""
        manifest = get_document_manifest(populated_store, "STD-001")
        amendments = manifest["amendments"]
        assert len(amendments) == 1
        assert amendments[0]["document_id"] == "AMD-001"

    def test_manifest_not_found(self, populated_store):
        """Should return None for non-existent document."""
        manifest = get_document_manifest(populated_store, "NONEXISTENT")
        assert manifest is None


class TestSearch:
    """Tests for unified search function."""

    def test_search_exact_mode(self, populated_store):
        """Exact mode should find items by number."""
        embedder = HashingEmbedder(dim=64)
        response = search(
            store=populated_store,
            query="1.1",
            mode=SearchMode.EXACT.value,
            top_k=5,
            embedder=embedder,
        )
        assert len(response.results) > 0
        # Should find clause 1.1
        item_ids = {r.item_id for r in response.results}
        assert "STD-001:2024:clause:1.1" in item_ids

    def test_search_exact_partial_match(self, populated_store):
        """Exact mode should support partial number matches (e.g., '9.2' matches '9.2.1')."""
        embedder = HashingEmbedder(dim=64)
        response = search(
            store=populated_store,
            query="1",
            mode=SearchMode.EXACT.value,
            top_k=10,
            embedder=embedder,
        )
        # Should find items starting with "1"
        assert len(response.results) > 0

    def test_search_keyword_mode(self, populated_store):
        """Keyword mode should use FTS5."""
        embedder = HashingEmbedder(dim=64)
        response = search(
            store=populated_store,
            query="steel",
            mode=SearchMode.KEYWORD.value,
            top_k=5,
            embedder=embedder,
        )
        assert len(response.results) > 0
        # All results should have match_type keyword
        for r in response.results:
            assert r.match_type == "keyword"

    def test_search_semantic_mode_requires_semantic_enabled(self, populated_store):
        """Semantic mode should raise ValueError when semantic_enabled=False."""
        embedder = HashingEmbedder(dim=64)
        # Default settings has semantic_enabled=False
        settings = Settings()
        with pytest.raises(ValueError, match="semantic_enabled=False"):
            search(
                store=populated_store,
                query="steel structures",
                mode=SearchMode.SEMANTIC.value,
                top_k=5,
                embedder=embedder,
                settings=settings,
            )

    def test_search_semantic_mode_requires_index(self, populated_store):
        """Semantic mode should raise ValueError when index not built."""
        embedder = HashingEmbedder(dim=64)
        # Enable semantic but don't build index
        settings = Settings(semantic_enabled=True)
        with pytest.raises(ValueError, match="built semantic index"):
            search(
                store=populated_store,
                query="steel structures",
                mode=SearchMode.SEMANTIC.value,
                top_k=5,
                embedder=embedder,
                settings=settings,
            )

    def test_search_hybrid_mode_requires_semantic_enabled(self, populated_store):
        """Hybrid mode should raise ValueError when semantic_enabled=False."""
        embedder = HashingEmbedder(dim=64)
        settings = Settings()
        with pytest.raises(ValueError, match="semantic_enabled=False"):
            search(
                store=populated_store,
                query="steel structures",
                mode=SearchMode.HYBRID.value,
                top_k=5,
                embedder=embedder,
                settings=settings,
            )

    def test_search_hybrid_mode_requires_index(self, populated_store):
        """Hybrid mode should raise ValueError when index not built."""
        embedder = HashingEmbedder(dim=64)
        settings = Settings(semantic_enabled=True)
        with pytest.raises(ValueError, match="built semantic index"):
            search(
                store=populated_store,
                query="steel structures",
                mode=SearchMode.HYBRID.value,
                top_k=5,
                embedder=embedder,
                settings=settings,
            )

    def test_search_hybrid_lexical_mode(self, populated_store):
        """Hybrid-lexical mode should work without semantic enabled (vectorless-first)."""
        embedder = HashingEmbedder(dim=64)
        settings = Settings(semantic_enabled=False)  # Default
        response = search(
            store=populated_store,
            query="steel structures",
            mode=SearchMode.HYBRID_LEXICAL.value,
            top_k=5,
            embedder=embedder,
            settings=settings,
        )
        assert response.degraded is False
        assert len(response.results) > 0
        # Results should have match_type exact, keyword, or hybrid-lexical
        for r in response.results:
            assert r.match_type in ("exact", "keyword", "hybrid-lexical")

    def test_search_with_document_filter(self, populated_store):
        """Search should respect document_id filter."""
        embedder = HashingEmbedder(dim=64)
        response = search(
            store=populated_store,
            query="steel",
            document_id="STD-001",
            mode=SearchMode.KEYWORD.value,
            top_k=5,
            embedder=embedder,
        )
        for r in response.results:
            assert r.document_id == "STD-001"

    def test_search_with_jurisdiction_filter(self, populated_store):
        """Search should respect jurisdiction filter."""
        embedder = HashingEmbedder(dim=64)
        response = search(
            store=populated_store,
            query="steel",
            jurisdiction="US",
            mode=SearchMode.KEYWORD.value,
            top_k=5,
            embedder=embedder,
        )
        for r in response.results:
            doc = populated_store.get_document(r.document_id)
            assert doc and doc.jurisdiction == "US"

    def test_search_with_doc_type_filter(self, populated_store):
        """Search should respect doc_type filter."""
        embedder = HashingEmbedder(dim=64)
        response = search(
            store=populated_store,
            query="scope",
            doc_type="amendment",
            mode=SearchMode.KEYWORD.value,
            top_k=5,
            embedder=embedder,
        )
        for r in response.results:
            doc = populated_store.get_document(r.document_id)
            assert doc and doc.doc_type.value == "amendment"

    def test_search_token_budget_trimming(self, populated_store):
        """Results should be trimmed to token budget."""
        embedder = HashingEmbedder(dim=64)
        response = search(
            store=populated_store,
            query="steel",
            mode=SearchMode.KEYWORD.value,
            top_k=10,
            token_budget=100,  # Very small budget
            embedder=embedder,
        )
        # Should have has_more_context on last item if trimmed
        if len(response.results) > 1:
            assert response.has_more_context is True
            assert response.results[-1].has_more_context is True

    def test_search_amendment_awareness(self, populated_store):
        """Search should include amendment documents when relevant."""
        embedder = HashingEmbedder(dim=64)
        response = search(
            store=populated_store,
            query="scope",
            mode=SearchMode.KEYWORD.value,
            top_k=10,
            embedder=embedder,
        )
        # Should find both baseline clause and amendment clause
        doc_ids = {r.document_id for r in response.results}
        assert "AMD-001" in doc_ids or response.amendments_included


class TestGetSection:
    """Tests for get_section."""

    def test_get_section(self, populated_store):
        """Should return section with descendants."""
        result = get_section(populated_store, "STD-001", "1")
        assert result is not None
        assert result.number == "1"
        assert result.item_id == "STD-001:2024:section:1"
        assert result.has_more_context is True  # Has children

    def test_get_section_not_found(self, populated_store):
        """Should return None for non-existent section."""
        result = get_section(populated_store, "STD-001", "99")
        assert result is None


class TestGetClause:
    """Tests for get_clause."""

    def test_get_clause(self, populated_store):
        """Should return clause with children, footnotes, amendment links."""
        result = get_clause(populated_store, "STD-001", "1.1")
        assert result is not None
        assert result.number == "1.1"
        assert result.item_id == "STD-001:2024:clause:1.1"
        assert result.has_more_context is True

    def test_get_clause_not_found(self, populated_store):
        """Should return None for non-existent clause."""
        result = get_clause(populated_store, "STD-001", "9.9")
        assert result is None


class TestGetEquation:
    """Tests for get_equation."""

    def test_get_equation(self, populated_store):
        """Should return equation."""
        result = get_equation(populated_store, "STD-001", "1.1")
        assert result is not None
        assert result.number == "1.1"
        assert result.item_id == "STD-001:2024:equation:1.1"

    def test_get_equation_not_found(self, populated_store):
        """Should return None for non-existent equation."""
        result = get_equation(populated_store, "STD-001", "9.9")
        assert result is None


class TestGetTable:
    """Tests for get_table."""

    def test_get_table(self, populated_store):
        """Should return table with footnotes."""
        result = get_table(populated_store, "STD-001", "1")
        assert result is not None
        assert result.number == "1"
        assert result.item_id == "STD-001:2024:table:1"
        assert result.has_more_context is True  # Has footnotes

    def test_get_table_not_found(self, populated_store):
        """Should return None for non-existent table."""
        result = get_table(populated_store, "STD-001", "99")
        assert result is None


class TestGetFigure:
    """Tests for get_figure."""

    def test_get_figure(self, populated_store):
        """Should return figure with asset info."""
        result = get_figure(populated_store, "STD-001", "1")
        assert result is not None
        assert result.number == "1"
        assert result.item_id == "STD-001:2024:figure:1"
        assert result.has_more_context is True  # Has assets

    def test_get_figure_not_found(self, populated_store):
        """Should return None for non-existent figure."""
        result = get_figure(populated_store, "STD-001", "99")
        assert result is None


class TestGetPage:
    """Tests for get_page."""

    def test_get_page(self, populated_store):
        """Should return all items on a page."""
        results = get_page(populated_store, "STD-001", 1)
        assert len(results) > 0
        for r in results:
            assert r.document_id == "STD-001"
            assert r.pdf_page == 1


class TestGetRelated:
    """Tests for get_related."""

    def test_get_related(self, populated_store):
        """Should return cross-refs, parent, footnotes, amendment links."""
        result = get_related(populated_store, "STD-001:2024:clause:1.1")
        assert result is not None
        assert result.item_id == "STD-001:2024:clause:1.1"
        assert result.has_more_context is True

    def test_get_related_not_found(self, populated_store):
        """Should return None for non-existent item."""
        result = get_related(populated_store, "NONEXISTENT")
        assert result is None


class TestSemanticSearchWithIndex:
    """Tests for semantic search after building index."""

    def test_semantic_search_after_build(self, populated_store):
        """Semantic search should work after building vectors."""
        from civileng_anything.index import VectorIndex
        embedder = HashingEmbedder(dim=64)
        index = VectorIndex(populated_store.db_path)
        index.build_vectors(populated_store, embedder)

        response = search(
            store=populated_store,
            query="steel structures",
            mode=SearchMode.SEMANTIC.value,
            top_k=5,
            embedder=embedder,
        )
        assert response.degraded is False
        assert len(response.results) > 0
        for r in response.results:
            assert r.match_type == "semantic"

    def test_hybrid_search_with_index(self, populated_store):
        """Hybrid search should fuse keyword and semantic when index built."""
        from civileng_anything.index import VectorIndex
        embedder = HashingEmbedder(dim=64)
        index = VectorIndex(populated_store.db_path)
        index.build_vectors(populated_store, embedder)

        response = search(
            store=populated_store,
            query="steel",
            mode=SearchMode.HYBRID.value,
            top_k=5,
            embedder=embedder,
        )
        assert response.degraded is False
        assert len(response.results) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])