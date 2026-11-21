"""
Tests for MCP server (Brief 03).
"""

from __future__ import annotations

import pytest
import json
import base64

from civileng_anything.mcp_server import (
    initialize_globals,
    get_store,
    get_embedder_dep,
    HANDLERS,
    TOOLS,
)
from civileng_anything.config import Settings
from civileng_anything.store import Store
from civileng_anything.schema import DocumentMeta, DocType, ItemType, ContentItem, ContentOrigin, ItemStatus
from civileng_anything.embeddings import HashingEmbedder
from civileng_anything.index import VectorIndex
from datetime import date


@pytest.fixture
def mcp_setup(tmp_path):
    """Set up MCP globals with test data."""
    settings = Settings(data_dir=tmp_path)
    initialize_globals(settings)

    store = get_store()
    embedder = get_embedder_dep()

    # Create test document
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

    # Add test items
    items = [
        {
            "item_id": "TEST-001:2024:section:1",
            "document_id": "TEST-001",
            "edition_id": "2024",
            "type": ItemType.SECTION,
            "number": "1",
            "title": "General",
            "text": "General requirements for steel structures.",
            "section_path": json.dumps(["1"]),
            "page_start": 1,
            "page_end": 2,
            "parent_id": None,
            "asset_ids": [],
            "cross_refs": ["Clause 1.1"],
            "footnote_ids": [],
            "amendment_links": [],
            "status": "verified",
            "content_origin": "parsed",
        },
        {
            "item_id": "TEST-001:2024:clause:1.1",
            "document_id": "TEST-001",
            "edition_id": "2024",
            "type": ItemType.CLAUSE,
            "number": "1.1",
            "title": "Scope",
            "text": "This clause defines the scope for steel structures.",
            "section_path": json.dumps(["1"]),
            "page_start": 1,
            "page_end": 1,
            "parent_id": "TEST-001:2024:section:1",
            "asset_ids": [],
            "cross_refs": [],
            "footnote_ids": ["TEST-001:2024:footnote:1"],
            "amendment_links": [],
            "status": "verified",
            "content_origin": "parsed",
        },
        {
            "item_id": "TEST-001:2024:footnote:1",
            "document_id": "TEST-001",
            "edition_id": "2024",
            "type": ItemType.FOOTNOTE,
            "number": "1",
            "title": None,
            "text": "Refer to ASTM A6.",
            "section_path": json.dumps(["1"]),
            "page_start": 1,
            "page_end": 1,
            "parent_id": "TEST-001:2024:clause:1.1",
            "asset_ids": [],
            "cross_refs": [],
            "footnote_ids": [],
            "amendment_links": [],
            "status": "verified",
            "content_origin": "parsed",
        },
        {
            "item_id": "TEST-001:2024:figure:1",
            "document_id": "TEST-001",
            "edition_id": "2024",
            "type": ItemType.FIGURE,
            "number": "1",
            "title": "Stress-Strain Curve",
            "text": "Figure 1 shows stress-strain curve.",
            "section_path": json.dumps(["1"]),
            "page_start": 2,
            "page_end": 2,
            "parent_id": "TEST-001:2024:section:1",
            "asset_ids": ["asset-fig1"],
            "cross_refs": [],
            "footnote_ids": [],
            "amendment_links": [],
            "status": "verified",
            "content_origin": "parsed",
        },
    ]

    content_items = [
        ContentItem(
            item_id=item["item_id"],
            document_id=item["document_id"],
            edition_id=item["edition_id"],
            type=item["type"],
            number=item["number"],
            text=item["text"],
            section_path=json.loads(item["section_path"]),
            pdf_page=item["page_start"],
            printed_page=str(item["page_start"]),
            parent_id=item["parent_id"],
            asset_ids=item["asset_ids"],
            cross_refs=item["cross_refs"],
            footnote_ids=item["footnote_ids"],
            amendment_links=item["amendment_links"],
            content_origin=ContentOrigin.ORIGINAL,
            status=ItemStatus.CHECKED,
        )
        for item in items
    ]
    store.add_items(content_items)

    # Add asset
    from civileng_anything.schema import AssetMeta
    import os
    os.makedirs(tmp_path / "assets", exist_ok=True)
    asset_path = tmp_path / "assets" / "fig1.png"
    with open(asset_path, "wb") as f:
        f.write(b"fake png data")

    asset = AssetMeta(
        asset_id="asset-fig1",
        document_id="TEST-001",
        item_id="TEST-001:2024:figure:1",
        kind="figure",
        file_path=str(asset_path),
        mime_type="image/png",
        width=800,
        height=600,
        sha256="asset123",
        description="Stress-strain curve",
    )
    store.upsert_asset(asset)

    # Build vector index
    index = VectorIndex(settings.db_path)
    index.build_vectors(store, embedder)

    yield settings

    store.close()


class TestMCPTools:
    """Tests for MCP tool definitions."""

    def test_tools_list(self):
        """Should have all required tools."""
        tool_names = {t["name"] for t in TOOLS}
        expected = {
            "list_documents",
            "get_document_manifest",
            "get_document_structure",
            "search",
            "get_section",
            "get_clause",
            "get_equation",
            "get_table",
            "get_figure",
            "get_page",
            "get_related",
            "build_vectors",
        }
        assert tool_names == expected

    def test_tool_schemas(self):
        """Each tool should have proper schema."""
        for tool in TOOLS:
            assert "name" in tool
            assert "description" in tool
            assert "inputSchema" in tool
            assert tool["inputSchema"]["type"] == "object"
            assert "properties" in tool["inputSchema"]


class TestMCPHandlers:
    """Tests for MCP tool handlers."""

    def test_list_documents(self, mcp_setup):
        """list_documents should return all documents."""
        result = HANDLERS["list_documents"]({})
        assert "documents" in result
        assert len(result["documents"]) == 1
        assert result["documents"][0]["document_id"] == "TEST-001"

    def test_get_document_manifest(self, mcp_setup):
        """get_document_manifest should return manifest."""
        result = HANDLERS["get_document_manifest"]({"document_id": "TEST-001"})
        assert "document" in result
        assert result["document"]["document_id"] == "TEST-001"
        assert "page_counts" in result
        assert "amendments" in result

    def test_get_document_manifest_not_found(self, mcp_setup):
        """get_document_manifest should raise for non-existent doc."""
        with pytest.raises(ValueError, match="Document not found"):
            HANDLERS["get_document_manifest"]({"document_id": "NONEXISTENT"})

    def test_search_keyword(self, mcp_setup):
        """search should work in keyword mode."""
        result = HANDLERS["search"]({
            "query": "steel",
            "mode": "keyword",
            "top_k": 5,
        })
        assert "results" in result
        assert result["mode"] == "keyword"
        assert len(result["results"]) > 0

    def test_search_semantic(self, mcp_setup):
        """search should work in semantic mode after index build (requires semantic_enabled)."""
        # Enable semantic and build index
        from civileng_anything.config import Settings
        from civileng_anything.mcp_server import _settings, _store
        _settings.semantic_enabled = True
        HANDLERS["build_vectors"]({"document_id": "TEST-001"})
        
        result = HANDLERS["search"]({
            "query": "steel structures",
            "mode": "semantic",
            "top_k": 5,
        })
        assert "results" in result
        assert result["mode"] == "semantic"
        assert result["degraded"] is False
        assert len(result["results"]) > 0

    def test_search_hybrid(self, mcp_setup):
        """search should work in hybrid mode (requires semantic_enabled)."""
        # Enable semantic and build index
        from civileng_anything.mcp_server import _settings
        _settings.semantic_enabled = True
        HANDLERS["build_vectors"]({"document_id": "TEST-001"})
        
        result = HANDLERS["search"]({
            "query": "steel",
            "mode": "hybrid",
            "top_k": 5,
        })
        assert "results" in result
        assert result["mode"] == "hybrid"
        assert result["degraded"] is False
        assert len(result["results"]) > 0

    def test_search_exact(self, mcp_setup):
        """search should work in exact mode."""
        result = HANDLERS["search"]({
            "query": "1.1",
            "mode": "exact",
            "top_k": 5,
        })
        assert "results" in result
        assert result["mode"] == "exact"
        assert len(result["results"]) > 0

    def test_search_with_filters(self, mcp_setup):
        """search should respect filters."""
        result = HANDLERS["search"]({
            "query": "steel",
            "document_id": "TEST-001",
            "mode": "keyword",
            "top_k": 5,
        })
        assert len(result["results"]) > 0
        for r in result["results"]:
            assert r["document_id"] == "TEST-001"

    def test_get_section(self, mcp_setup):
        """get_section should return section with descendants."""
        result = HANDLERS["get_section"]({
            "document_id": "TEST-001",
            "section_number": "1",
        })
        assert result["number"] == "1"
        assert "section" in result["item_id"]
        assert result["has_more_context"] is True

    def test_get_section_not_found(self, mcp_setup):
        """get_section should raise for non-existent section."""
        with pytest.raises(ValueError, match="Section not found"):
            HANDLERS["get_section"]({
                "document_id": "TEST-001",
                "section_number": "99",
            })

    def test_get_clause(self, mcp_setup):
        """get_clause should return clause with footnotes."""
        result = HANDLERS["get_clause"]({
            "document_id": "TEST-001",
            "number": "1.1",
        })
        assert result["number"] == "1.1"
        assert "clause" in result["item_id"]
        assert result["has_more_context"] is True

    def test_get_clause_not_found(self, mcp_setup):
        """get_clause should raise for non-existent clause."""
        with pytest.raises(ValueError, match="Clause not found"):
            HANDLERS["get_clause"]({
                "document_id": "TEST-001",
                "number": "9.9",
            })

    def test_get_equation(self, mcp_setup):
        """get_equation should return equation."""
        store = get_store()
        # Add equation
        items = [
            ContentItem(
                item_id="TEST-001:2024:equation:1.1",
                document_id="TEST-001",
                edition_id="2024",
                type=ItemType.EQUATION,
                number="1.1",
                text="E = mc^2",
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
            )
        ]
        store.add_items(items)

        result = HANDLERS["get_equation"]({
            "document_id": "TEST-001",
            "number": "1.1",
        })
        assert result["number"] == "1.1"
        assert "equation" in result["item_id"]

    def test_get_table(self, mcp_setup):
        """get_table should return table."""
        store = get_store()
        items = [
            ContentItem(
                item_id="TEST-001:2024:table:1",
                document_id="TEST-001",
                edition_id="2024",
                type=ItemType.TABLE,
                number="1",
                text="Table data",
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
            )
        ]
        store.add_items(items)

        result = HANDLERS["get_table"]({
            "document_id": "TEST-001",
            "number": "1",
        })
        assert result["number"] == "1"
        assert "table" in result["item_id"]

    def test_get_figure(self, mcp_setup):
        """get_figure should return figure with asset resources."""
        result = HANDLERS["get_figure"]({
            "document_id": "TEST-001",
            "number": "1",
        })
        assert result["number"] == "1"
        assert "figure" in result["item_id"]
        assert "resources" in result
        assert len(result["resources"]) == 1
        assert result["resources"][0]["uri"] == "asset://asset-fig1"
        assert result["resources"][0]["mimeType"] == "image/png"
        # Verify base64 blob
        blob = result["resources"][0]["blob"]
        decoded = base64.b64decode(blob)
        assert decoded == b"fake png data"

    def test_get_figure_not_found(self, mcp_setup):
        """get_figure should raise for non-existent figure."""
        with pytest.raises(ValueError, match="Figure not found"):
            HANDLERS["get_figure"]({
                "document_id": "TEST-001",
                "number": "99",
            })

    def test_get_page(self, mcp_setup):
        """get_page should return all items on a page."""
        result = HANDLERS["get_page"]({
            "document_id": "TEST-001",
            "pdf_page": 1,
        })
        assert "results" in result
        assert len(result["results"]) > 0

    def test_get_related(self, mcp_setup):
        """get_related should return cross-refs, parent, footnotes, amendment links."""
        result = HANDLERS["get_related"]({
            "item_id": "TEST-001:2024:clause:1.1",
        })
        assert result["item_id"] == "TEST-001:2024:clause:1.1"
        assert result["has_more_context"] is True

    def test_get_related_not_found(self, mcp_setup):
        """get_related should raise for non-existent item."""
        with pytest.raises(ValueError, match="Item not found"):
            HANDLERS["get_related"]({"item_id": "NONEXISTENT"})

    def test_build_vectors(self, mcp_setup):
        """build_vectors should rebuild index."""
        result = HANDLERS["build_vectors"]({"document_id": "TEST-001"})
        assert "built_count" in result
        # Second build should skip
        result2 = HANDLERS["build_vectors"]({"document_id": "TEST-001"})
        assert result2["built_count"] == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])