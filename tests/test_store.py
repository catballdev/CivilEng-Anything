"""
Tests for the SQLite store.
"""

import json
import tempfile
from pathlib import Path
from datetime import date

import pytest

from civileng_anything.store import Store
from civileng_anything.schema import (
    DocumentMeta,
    ContentItem,
    PageStatus,
    DocType,
    ItemType,
    PageStatusEnum,
    ContentOrigin,
    ItemStatus,
)
from civileng_anything.ids import make_item_id, content_hash


# --- Fixtures ---

@pytest.fixture
def temp_db():
    """Create a temporary database file."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = Path(f.name)
    yield db_path
    # Cleanup
    if db_path.exists():
        db_path.unlink()


@pytest.fixture
def store(temp_db):
    """Create a Store instance with temp database."""
    s = Store(temp_db)
    yield s
    s.close()


@pytest.fixture
def sample_document():
    """Create a sample document for testing."""
    return DocumentMeta(
        document_id="test-code-001",
        document_family_id="hk-cc",
        edition_id="2021",
        title="Code of Practice for Structural Use of Concrete 2021",
        publisher="Buildings Department",
        jurisdiction="Hong Kong",
        doc_type=DocType.GOVERNMENT_CODE,
        language="en",
        publication_date=date(2021, 12, 31),
        revision_date=None,
        source_url="https://example.com/concrete-2021.pdf",
        retrieved_at=date(2026, 1, 15),
        original_sha256="a" * 64,
        total_pdf_pages=2,
        parser_name="mineru",
        parser_version="3.4.5",
        parser_config={"backend": "cpu", "lang": "en"},
        parse_run_id="run-test-001",
        completeness=1.0,
        qc_status="checked",
        amendment_relations=[],
        version_relations=[],
    )


@pytest.fixture
def sample_items(sample_document):
    """Create sample content items (2 pages: clause+equation, table+footnote)."""
    doc_id = sample_document.document_id
    ed_id = sample_document.edition_id

    items = [
        # Page 1: Section header
        ContentItem(
            item_id=make_item_id(doc_id, 1, 1),
            document_id=doc_id,
            edition_id=ed_id,
            type=ItemType.SECTION,
            parent_id=None,
            section_path=["Chapter 6"],
            number="6",
            text="Chapter 6: Design of Concrete Structures",
            pdf_page=1,
            printed_page="13",
            status=ItemStatus.CHECKED,
        ),
        # Page 1: Clause with fcu reference
        ContentItem(
            item_id=make_item_id(doc_id, 1, 2),
            document_id=doc_id,
            edition_id=ed_id,
            type=ItemType.CLAUSE,
            parent_id=make_item_id(doc_id, 1, 1),
            section_path=["Chapter 6", "Section 6.2"],
            number="6.2.3",
            text="The design strength of concrete fcu shall be at least 25 MPa for Grade C25.",
            pdf_page=1,
            printed_page="15",
            cross_refs=["Table 6.1"],
            footnote_ids=[make_item_id(doc_id, 1, 4)],
            status=ItemStatus.CHECKED,
        ),
        # Page 1: Equation
        ContentItem(
            item_id=make_item_id(doc_id, 1, 3),
            document_id=doc_id,
            edition_id=ed_id,
            type=ItemType.EQUATION,
            parent_id=make_item_id(doc_id, 1, 2),
            section_path=["Chapter 6", "Section 6.2"],
            number="Eq. 6.1",
            text="fc = fcu / γm",
            latex=r"f_c = \frac{f_{cu}}{\gamma_m}",
            pdf_page=1,
            printed_page="15",
            status=ItemStatus.CHECKED,
        ),
        # Page 1: Footnote
        ContentItem(
            item_id=make_item_id(doc_id, 1, 4),
            document_id=doc_id,
            edition_id=ed_id,
            type=ItemType.FOOTNOTE,
            parent_id=make_item_id(doc_id, 1, 2),
            section_path=["Chapter 6", "Section 6.2"],
            number="1",
            text="fcu refers to the characteristic cube compressive strength of concrete.",
            pdf_page=1,
            printed_page="15",
            status=ItemStatus.CHECKED,
        ),
        # Page 2: Table
        ContentItem(
            item_id=make_item_id(doc_id, 2, 1),
            document_id=doc_id,
            edition_id=ed_id,
            type=ItemType.TABLE,
            parent_id=make_item_id(doc_id, 2, 0),
            section_path=["Chapter 6", "Section 6.3"],
            number="Table 6.1",
            text="Concrete Grade | fcu (MPa) | γm\nC25 | 25 | 1.5\nC30 | 30 | 1.5\nC35 | 35 | 1.5",
            table_html="<table><thead><tr><th>Concrete Grade</th><th>fcu (MPa)</th><th>γm</th></tr></thead><tbody><tr><td>C25</td><td>25</td><td>1.5</td></tr><tr><td>C30</td><td>30</td><td>1.5</td></tr><tr><td>C35</td><td>35</td><td>1.5</td></tr></tbody></table>",
            pdf_page=2,
            printed_page="16",
            asset_ids=["asset-table-6-1"],
            cross_refs=["Cl. 6.2.3"],
            status=ItemStatus.CHECKED,
        ),
    ]
    return items


@pytest.fixture
def sample_page_statuses(sample_document):
    """Create sample page status records."""
    doc_id = sample_document.document_id
    return [
        PageStatus(document_id=doc_id, pdf_page=1, status=PageStatusEnum.DONE, parser="mineru", detail="OK"),
        PageStatus(document_id=doc_id, pdf_page=2, status=PageStatusEnum.DONE, parser="mineru", detail="OK"),
    ]


# --- Tests ---

def test_init_db(store):
    """Test database initialization creates all tables."""
    conn = store._get_conn()
    tables = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
    ).fetchall()
    table_names = [t[0] for t in tables]
    assert "documents" in table_names
    assert "items" in table_names
    assert "pages" in table_names
    assert "assets" in table_names
    assert "items_fts" in table_names


def test_upsert_and_get_document(store, sample_document):
    """Test document upsert and retrieval."""
    store.upsert_document(sample_document)
    retrieved = store.get_document(sample_document.document_id)
    assert retrieved is not None
    assert retrieved.document_id == sample_document.document_id
    assert retrieved.title == sample_document.title
    assert retrieved.doc_type == sample_document.doc_type
    assert retrieved.original_sha256 == sample_document.original_sha256
    assert retrieved.publication_date == sample_document.publication_date


def test_upsert_document_idempotent(store, sample_document):
    """Test that upserting the same document_id updates in place."""
    store.upsert_document(sample_document)
    # Modify and upsert again
    sample_document.title = "Updated Title"
    sample_document.completeness = 0.5
    store.upsert_document(sample_document)

    retrieved = store.get_document(sample_document.document_id)
    assert retrieved.title == "Updated Title"
    assert retrieved.completeness == 0.5

    # Should still be only one document
    docs = store.list_documents()
    assert len(docs) == 1


def test_list_documents(store, sample_document):
    """Test listing documents."""
    store.upsert_document(sample_document)
    docs = store.list_documents()
    assert len(docs) == 1
    assert docs[0].document_id == sample_document.document_id


def test_add_items_and_get(store, sample_document, sample_items):
    """Test bulk add items and retrieval."""
    store.upsert_document(sample_document)
    store.add_items(sample_items)

    # Retrieve each item
    for item in sample_items:
        retrieved = store.get_item(item.item_id)
        assert retrieved is not None
        assert retrieved.item_id == item.item_id
        assert retrieved.text == item.text
        assert retrieved.type == item.type
        assert retrieved.pdf_page == item.pdf_page


def test_add_items_idempotent(store, sample_document, sample_items):
    """Test that re-adding same item_id updates in place (no dupes)."""
    store.upsert_document(sample_document)
    store.add_items(sample_items)

    # Modify first item and re-add
    modified = sample_items[0].model_copy()
    modified.text = "Modified text"
    modified.status = ItemStatus.PENDING_VERIFICATION
    store.add_items([modified])

    retrieved = store.get_item(modified.item_id)
    assert retrieved.text == "Modified text"
    assert retrieved.status == ItemStatus.PENDING_VERIFICATION

    # Count should still be 5
    conn = store._get_conn()
    count = conn.execute("SELECT COUNT(*) FROM items").fetchone()[0]
    assert count == 5


def test_items_by_number_exact_lookup(store, sample_document, sample_items):
    """Test exact lookup by type + number within document/edition."""
    store.upsert_document(sample_document)
    store.add_items(sample_items)

    # Look up clause 6.2.3
    results = store.items_by_number(
        sample_document.document_id,
        sample_document.edition_id,
        ItemType.CLAUSE.value,
        "6.2.3",
    )
    assert len(results) == 1
    assert results[0].number == "6.2.3"
    assert results[0].type == ItemType.CLAUSE
    assert "fcu" in results[0].text

    # Look up table 6.1
    results = store.items_by_number(
        sample_document.document_id,
        sample_document.edition_id,
        ItemType.TABLE.value,
        "Table 6.1",
    )
    assert len(results) == 1
    assert results[0].number == "Table 6.1"
    assert results[0].type == ItemType.TABLE
    assert results[0].table_html is not None

    # Non-existent
    results = store.items_by_number(
        sample_document.document_id,
        sample_document.edition_id,
        ItemType.CLAUSE.value,
        "9.9.9",
    )
    assert len(results) == 0


def test_fts_search_finds_fcu(store, sample_document, sample_items):
    """Test FTS search finds clause by engineering symbol token 'fcu'."""
    store.upsert_document(sample_document)
    store.add_items(sample_items)

    # Search for fcu - should find at least one item containing it
    results = store.fts_search("fcu", limit=10)
    assert len(results) >= 1
    # At least one result should contain "fcu" (could be clause or equation)
    found = any("fcu" in r.snippet.lower() for r in results)
    assert found, "FTS search should find 'fcu' in some item text"

    # Search for equation symbol γm
    results = store.fts_search("γm", limit=10)
    assert len(results) >= 1
    found = any("γm" in r.snippet for r in results)
    assert found, "FTS search should find 'γm' (engineering symbol preserved)"

    # Search for clause number
    results = store.fts_search("6.2.3", limit=10)
    assert len(results) >= 1
    assert any(r.number == "6.2.3" for r in results)


def test_fts_search_filters(store, sample_document, sample_items):
    """Test FTS search with document/edition filters."""
    store.upsert_document(sample_document)
    store.add_items(sample_items)

    # Filter by document_id
    results = store.fts_search("concrete", document_id=sample_document.document_id, limit=10)
    assert len(results) >= 1
    assert all(r.document_id == sample_document.document_id for r in results)

    # Filter by edition_id
    results = store.fts_search("concrete", edition_id=sample_document.edition_id, limit=10)
    assert len(results) >= 1
    assert all(r.edition_id == sample_document.edition_id for r in results)

    # Filter by non-matching document_id
    results = store.fts_search("concrete", document_id="non-existent", limit=10)
    assert len(results) == 0


def test_add_page_status(store, sample_document, sample_page_statuses):
    """Test page status insert and update."""
    store.upsert_document(sample_document)
    for ps in sample_page_statuses:
        store.add_page_status(ps)

    conn = store._get_conn()
    rows = conn.execute(
        "SELECT * FROM pages WHERE document_id = ? ORDER BY pdf_page",
        (sample_document.document_id,),
    ).fetchall()
    assert len(rows) == 2
    assert rows[0]["pdf_page"] == 1
    assert rows[0]["status"] == "done"
    assert rows[1]["pdf_page"] == 2
    assert rows[1]["status"] == "done"

    # Update page 1 status
    updated = PageStatus(
        document_id=sample_document.document_id,
        pdf_page=1,
        status=PageStatusEnum.NEEDS_REVIEW,
        parser="mineru",
        detail="Low confidence on table",
    )
    store.add_page_status(updated)

    row = conn.execute(
        "SELECT * FROM pages WHERE document_id = ? AND pdf_page = 1",
        (sample_document.document_id,),
    ).fetchone()
    assert row["status"] == "needs_review"
    assert row["detail"] == "Low confidence on table"


def test_export_jsonl(store, sample_document, sample_items):
    """Test JSONL export produces correct line count and format."""
    store.upsert_document(sample_document)
    store.add_items(sample_items)

    with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
        output_path = Path(f.name)

    try:
        line_count = store.export_jsonl(output_path)
        # 1 document + 5 items = 6 lines
        assert line_count == 6

        # Verify content
        with open(output_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        assert len(lines) == 6

        # First line should be document
        doc_line = json.loads(lines[0])
        assert doc_line["_kind"] == "document"
        assert doc_line["document_id"] == sample_document.document_id
        assert doc_line["title"] == sample_document.title

        # Remaining lines should be items
        item_lines = [json.loads(l) for l in lines[1:]]
        assert len(item_lines) == 5
        assert all(l["_kind"] == "item" for l in item_lines)
        item_ids = {l["item_id"] for l in item_lines}
        expected_ids = {item.item_id for item in sample_items}
        assert item_ids == expected_ids

        # Check that item has all fields
        clause_item = next(l for l in item_lines if l["number"] == "6.2.3")
        assert clause_item["type"] == "clause"
        assert "fcu" in clause_item["text"]
        assert clause_item["cross_refs"] == ["Table 6.1"]
        assert clause_item["footnote_ids"] == [sample_items[3].item_id]

    finally:
        if output_path.exists():
            output_path.unlink()


def test_content_hash():
    """Test content_hash produces consistent SHA-256."""
    text = "The design strength of concrete fcu shall be at least 25 MPa."
    h1 = content_hash(text)
    h2 = content_hash(text)
    assert h1 == h2
    assert len(h1) == 64  # SHA-256 hex length
    # Different text gives different hash
    assert content_hash("different") != h1


def test_make_item_id_format():
    """Test make_item_id produces stable, readable IDs."""
    id1 = make_item_id("doc-001", 1, 1)
    id2 = make_item_id("doc-001", 1, 1)
    assert id1 == id2
    assert id1 == "doc-001-p0001-0001"

    # Different page/seq gives different ID
    assert make_item_id("doc-001", 1, 2) == "doc-001-p0001-0002"
    assert make_item_id("doc-001", 2, 1) == "doc-001-p0002-0001"

    # Special chars in document_id are sanitized
    assert make_item_id("doc/with:special*chars", 1, 1) == "doc_with_special_chars-p0001-0001"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])