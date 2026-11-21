"""
Tests for FastAPI application (Brief 03).
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
import pytest
from fastapi.testclient import TestClient

# Set test data dir env var BEFORE importing app so get_settings() uses it
_test_data_dir = Path(tempfile.mkdtemp(prefix="civileng_test_"))
os.environ["CIVILENG_DATA_DIR"] = str(_test_data_dir)

from civileng_anything.api import app
from civileng_anything.config import Settings, get_settings
from civileng_anything.store import Store
from civileng_anything.schema import DocumentMeta, DocType, ItemType, ContentItem, ContentOrigin, ItemStatus
from civileng_anything.embeddings import HashingEmbedder
from civileng_anything.index import VectorIndex
from datetime import date
import json
import shutil


@pytest.fixture(scope="session", autouse=True)
def cleanup_test_dir():
    """Clean up test data directory after all tests."""
    yield
    shutil.rmtree(_test_data_dir, ignore_errors=True)


@pytest.fixture
def test_settings():
    """Create test settings using the test data dir."""
    return get_settings()


@pytest.fixture
def populated_store(test_settings):
    """Create a store with test data."""
    store = Store(test_settings.db_path)

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

    # Add test items
    items = [
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
            "parent_id": None,
            "asset_ids": [],
            "cross_refs": [],
            "footnote_ids": [],
            "amendment_links": [],
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

    store.close()
    return test_settings


def test_health_endpoint(populated_store):
    """Test health check endpoint."""
    with TestClient(app) as client:
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}


def test_list_documents_endpoint(populated_store):
    """Test list documents endpoint."""
    with TestClient(app) as client:
        response = client.get("/documents")
        assert response.status_code == 200
        docs = response.json()
        assert len(docs) == 1
        assert docs[0]["document_id"] == "TEST-001"


def test_get_manifest_endpoint(populated_store):
    """Test get manifest endpoint."""
    with TestClient(app) as client:
        response = client.get("/documents/TEST-001/manifest")
        assert response.status_code == 200
        manifest = response.json()
        assert manifest["document"]["document_id"] == "TEST-001"
        assert "page_counts" in manifest
        assert "amendments" in manifest


def test_get_manifest_not_found(populated_store):
    """Test get manifest for non-existent document."""
    with TestClient(app) as client:
        response = client.get("/documents/NONEXISTENT/manifest")
        assert response.status_code == 404


def test_search_endpoint(populated_store):
    """Test search endpoint."""
    with TestClient(app) as client:
        response = client.post("/search", json={
            "query": "steel",
            "mode": "keyword",
            "top_k": 5,
        })
        assert response.status_code == 200
        data = response.json()
        assert "results" in data
        assert "mode" in data
        assert data["mode"] == "keyword"


def test_search_endpoint_invalid_mode(populated_store):
    """Test search endpoint with invalid mode."""
    with TestClient(app) as client:
        response = client.post("/search", json={
            "query": "steel",
            "mode": "invalid",
            "top_k": 5,
        })
        assert response.status_code == 400


def test_get_clause_endpoint(populated_store):
    """Test get clause endpoint."""
    with TestClient(app) as client:
        response = client.get("/documents/TEST-001/clauses/1.1")
        assert response.status_code == 200
        data = response.json()
        assert data["number"] == "1.1"
        # type is not in SearchResult, but item_id contains it
        assert "clause" in data["item_id"]


def test_get_clause_not_found(populated_store):
    """Test get clause for non-existent clause."""
    with TestClient(app) as client:
        response = client.get("/documents/TEST-001/clauses/9.9")
        assert response.status_code == 404


def test_get_section_endpoint(populated_store):
    """Test get section endpoint."""
    # Need to add a section first
    store = Store(populated_store.db_path)
    content_items = [
        ContentItem(
            item_id="TEST-001:2024:section:1",
            document_id="TEST-001",
            edition_id="2024",
            type=ItemType.SECTION,
            number="1",
            text="General requirements.",
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
    store.add_items(content_items)
    store.close()

    with TestClient(app) as client:
        response = client.get("/documents/TEST-001/sections/1")
        assert response.status_code == 200
        data = response.json()
        assert data["number"] == "1"
        assert "section" in data["item_id"]


def test_get_equation_endpoint(populated_store):
    """Test get equation endpoint."""
    store = Store(populated_store.db_path)
    content_items = [
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
        ),
    ]
    store.add_items(content_items)
    store.close()

    with TestClient(app) as client:
        response = client.get("/documents/TEST-001/equations/1.1")
        assert response.status_code == 200
        data = response.json()
        assert data["number"] == "1.1"
        assert "equation" in data["item_id"]


def test_get_table_endpoint(populated_store):
    """Test get table endpoint."""
    store = Store(populated_store.db_path)
    content_items = [
        ContentItem(
            item_id="TEST-001:2024:table:1",
            document_id="TEST-001",
            edition_id="2024",
            type=ItemType.TABLE,
            number="1",
            text="Table data.",
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
    store.add_items(content_items)
    store.close()

    with TestClient(app) as client:
        response = client.get("/documents/TEST-001/tables/1")
        assert response.status_code == 200
        data = response.json()
        assert data["number"] == "1"
        assert "table" in data["item_id"]


def test_get_figure_endpoint(populated_store):
    """Test get figure endpoint."""
    store = Store(populated_store.db_path)
    content_items = [
        ContentItem(
            item_id="TEST-001:2024:figure:1",
            document_id="TEST-001",
            edition_id="2024",
            type=ItemType.FIGURE,
            number="1",
            text="A figure.",
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
    store.add_items(content_items)
    store.close()

    with TestClient(app) as client:
        response = client.get("/documents/TEST-001/figures/1")
        assert response.status_code == 200
        data = response.json()
        assert data["number"] == "1"
        assert "figure" in data["item_id"]


def test_get_page_endpoint(populated_store):
    """Test get page endpoint."""
    with TestClient(app) as client:
        response = client.get("/documents/TEST-001/pages/1")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)


def test_get_related_endpoint(populated_store):
    """Test get related endpoint."""
    with TestClient(app) as client:
        response = client.get("/items/TEST-001:2024:clause:1.1/related")
        assert response.status_code == 200
        data = response.json()
        assert data["item_id"] == "TEST-001:2024:clause:1.1"


def test_get_related_not_found(populated_store):
    """Test get related for non-existent item."""
    with TestClient(app) as client:
        response = client.get("/items/NONEXISTENT/related")
        assert response.status_code == 404


def test_build_index_endpoint(populated_store):
    """Test build index endpoint."""
    with TestClient(app) as client:
        response = client.post("/index/build", params={"document_id": "TEST-001"})
        assert response.status_code == 200
        data = response.json()
        assert "built_count" in data
        assert data["built_count"] >= 0


def test_get_asset_endpoint(populated_store):
    """Test get asset endpoint."""
    # Need to add an asset
    from civileng_anything.schema import AssetMeta
    import os

    store = Store(populated_store.db_path)
    os.makedirs(populated_store.data_dir / "assets", exist_ok=True)
    asset_path = populated_store.data_dir / "assets" / "test.png"
    with open(asset_path, "wb") as f:
        f.write(b"fake png")

    asset = AssetMeta(
        asset_id="asset-test",
        document_id="TEST-001",
        item_id="TEST-001:2024:figure:1",
        kind="figure",
        file_path=str(asset_path),
        mime_type="image/png",
        width=100,
        height=100,
        sha256="test123",
        description="Test asset",
    )
    store.upsert_asset(asset)
    store.close()

    with TestClient(app) as client:
        response = client.get("/assets/asset-test")
        assert response.status_code == 200
        assert response.headers["content-type"] == "image/png"
        assert response.content == b"fake png"


def test_get_asset_not_found(populated_store):
    """Test get asset for non-existent asset."""
    with TestClient(app) as client:
        response = client.get("/assets/NONEXISTENT")
        assert response.status_code == 404


if __name__ == "__main__":
    pytest.main([__file__, "-v"])