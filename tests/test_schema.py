"""
Tests for the canonical schema models.
"""

import pytest
from datetime import date
from civileng_anything.schema import (
    DocumentMeta,
    ContentItem,
    PageStatus,
    SearchResult,
    DocType,
    ItemType,
    PageStatusEnum,
    ContentOrigin,
    ItemStatus,
)


def test_document_meta_minimal():
    """Test DocumentMeta with minimal required fields."""
    doc = DocumentMeta(
        document_id="test-doc-001",
        document_family_id="test-family",
        edition_id="ed-1",
        title="Test Code of Practice",
        doc_type=DocType.GOVERNMENT_CODE,
        original_sha256="a" * 64,
        total_pdf_pages=10,
        parser_name="mineru",
        parser_version="3.4.5",
        parse_run_id="run-001",
    )
    assert doc.document_id == "test-doc-001"
    assert doc.doc_type == DocType.GOVERNMENT_CODE
    assert doc.publication_date is None
    assert doc.revision_date is None
    assert doc.parser_config == {}
    assert doc.amendment_relations == []


def test_document_meta_full():
    """Test DocumentMeta with all fields."""
    doc = DocumentMeta(
        document_id="test-doc-002",
        document_family_id="hk-cc",
        edition_id="2021",
        title="Code of Practice for Structural Use of Concrete 2021",
        publisher="Buildings Department",
        jurisdiction="Hong Kong",
        doc_type=DocType.GOVERNMENT_CODE,
        language="en",
        publication_date=date(2021, 12, 31),
        revision_date=date(2022, 6, 15),
        source_url="https://example.com/code.pdf",
        retrieved_at=date(2026, 1, 1),
        original_sha256="b" * 64,
        total_pdf_pages=200,
        parser_name="mineru",
        parser_version="3.4.5",
        parser_config={"backend": "cpu", "lang": "en"},
        parse_run_id="run-002",
        completeness=0.95,
        qc_status="checked",
        amendment_relations=[{"amendment_id": "amd-001", "target_clause": "6.2.3"}],
        version_relations=[{"previous_edition": "2013"}],
    )
    assert doc.publication_date == date(2021, 12, 31)
    assert doc.revision_date == date(2022, 6, 15)
    assert doc.parser_config == {"backend": "cpu", "lang": "en"}
    assert len(doc.amendment_relations) == 1


def test_content_item_clause():
    """Test ContentItem for a clause with equation and footnote."""
    item = ContentItem(
        item_id="test-doc-001-p0001-0001",
        document_id="test-doc-001",
        edition_id="ed-1",
        type=ItemType.CLAUSE,
        parent_id="test-doc-001-p0001-0000",
        section_path=["Chapter 6", "Section 6.2"],
        number="6.2.3",
        text="The design strength of concrete fcu shall be at least 25 MPa.",
        latex=None,
        table_html=None,
        pdf_page=1,
        printed_page="15",
        bbox=[100, 200, 400, 250],
        asset_ids=[],
        cross_refs=["Cl. 6.2.1", "Table 6.1"],
        footnote_ids=["fn-001"],
        amendment_links=[],
        content_origin=ContentOrigin.ORIGINAL,
        derived_info=None,
        status=ItemStatus.PARSED,
    )
    assert item.type == ItemType.CLAUSE
    assert item.number == "6.2.3"
    assert "fcu" in item.text
    assert item.pdf_page == 1
    assert item.printed_page == "15"
    assert item.cross_refs == ["Cl. 6.2.1", "Table 6.1"]
    assert item.footnote_ids == ["fn-001"]


def test_content_item_equation():
    """Test ContentItem for an equation."""
    item = ContentItem(
        item_id="test-doc-001-p0001-0002",
        document_id="test-doc-001",
        edition_id="ed-1",
        type=ItemType.EQUATION,
        parent_id="test-doc-001-p0001-0001",
        section_path=["Chapter 6", "Section 6.2"],
        number="Eq. 6.2",
        text="fc = fcu / γm",
        latex=r"f_c = \frac{f_{cu}}{\gamma_m}",
        table_html=None,
        pdf_page=1,
        printed_page="15",
        bbox=None,
        asset_ids=[],
        cross_refs=[],
        footnote_ids=[],
        amendment_links=[],
        content_origin=ContentOrigin.ORIGINAL,
        derived_info=None,
        status=ItemStatus.PARSED,
    )
    assert item.type == ItemType.EQUATION
    assert item.latex == r"f_c = \frac{f_{cu}}{\gamma_m}"
    assert item.number == "Eq. 6.2"


def test_content_item_table():
    """Test ContentItem for a table."""
    item = ContentItem(
        item_id="test-doc-001-p0002-0001",
        document_id="test-doc-001",
        edition_id="ed-1",
        type=ItemType.TABLE,
        parent_id="test-doc-001-p0002-0000",
        section_path=["Chapter 6", "Section 6.3"],
        number="Table 6.1",
        text="Grade | fcu (MPa) | γm\nC25 | 25 | 1.5\nC30 | 30 | 1.5",
        latex=None,
        table_html="<table><tr><th>Grade</th><th>fcu (MPa)</th><th>γm</th></tr><tr><td>C25</td><td>25</td><td>1.5</td></tr><tr><td>C30</td><td>30</td><td>1.5</td></tr></table>",
        pdf_page=2,
        printed_page="16",
        bbox=[50, 100, 500, 300],
        asset_ids=["asset-table-001"],
        cross_refs=["Cl. 6.2.3"],
        footnote_ids=["fn-002"],
        amendment_links=[],
        content_origin=ContentOrigin.ORIGINAL,
        derived_info=None,
        status=ItemStatus.PARSED,
    )
    assert item.type == ItemType.TABLE
    assert item.table_html is not None
    assert "<table>" in item.table_html
    assert item.asset_ids == ["asset-table-001"]


def test_content_item_footnote():
    """Test ContentItem for a footnote."""
    item = ContentItem(
        item_id="test-doc-001-p0001-0003",
        document_id="test-doc-001",
        edition_id="ed-1",
        type=ItemType.FOOTNOTE,
        parent_id="test-doc-001-p0001-0001",
        section_path=["Chapter 6", "Section 6.2"],
        number="1",
        text="fcu refers to the characteristic cube compressive strength.",
        latex=None,
        table_html=None,
        pdf_page=1,
        printed_page="15",
        bbox=None,
        asset_ids=[],
        cross_refs=[],
        footnote_ids=[],
        amendment_links=[],
        content_origin=ContentOrigin.ORIGINAL,
        derived_info=None,
        status=ItemStatus.PARSED,
    )
    assert item.type == ItemType.FOOTNOTE
    assert item.number == "1"


def test_page_status():
    """Test PageStatus model."""
    page = PageStatus(
        document_id="test-doc-001",
        pdf_page=1,
        status=PageStatusEnum.DONE,
        parser="mineru",
        detail="Successfully parsed",
    )
    assert page.status == PageStatusEnum.DONE
    assert page.parser == "mineru"


def test_search_result():
    """Test SearchResult model."""
    result = SearchResult(
        item_id="test-doc-001-p0001-0001",
        document_id="test-doc-001",
        edition_id="ed-1",
        doc_title="Test Code of Practice",
        number="6.2.3",
        section_path=["Chapter 6", "Section 6.2"],
        snippet="The design strength of concrete fcu shall be at least 25 MPa…",
        pdf_page=1,
        printed_page="15",
        content_origin=ContentOrigin.ORIGINAL,
        status=ItemStatus.PARSED,
        score=0.95,
        has_more_context=True,
        related=["test-doc-001-p0001-0002", "test-doc-001-p0001-0003"],
    )
    assert result.score == 0.95
    assert result.has_more_context is True
    assert len(result.related) == 2


def test_enums():
    """Test all enum values are accessible."""
    assert DocType.GOVERNMENT_CODE.value == "government_code"
    assert DocType.AMENDMENT.value == "amendment"
    assert DocType.GUIDELINE.value == "guideline"
    assert DocType.TEXTBOOK.value == "textbook"
    assert DocType.OTHER.value == "other"

    assert ItemType.SECTION.value == "section"
    assert ItemType.CLAUSE.value == "clause"
    assert ItemType.PARAGRAPH.value == "paragraph"
    assert ItemType.EQUATION.value == "equation"
    assert ItemType.TABLE.value == "table"
    assert ItemType.FIGURE.value == "figure"
    assert ItemType.FOOTNOTE.value == "footnote"
    assert ItemType.CROSS_REFERENCE.value == "cross_reference"
    assert ItemType.AMENDMENT.value == "amendment"

    assert PageStatusEnum.DONE.value == "done"
    assert PageStatusEnum.FAILED.value == "failed"
    assert PageStatusEnum.RETRY_PENDING.value == "retry_pending"
    assert PageStatusEnum.NEEDS_REVIEW.value == "needs_review"

    assert ContentOrigin.ORIGINAL.value == "original"
    assert ContentOrigin.OCR.value == "ocr"
    assert ContentOrigin.DERIVED.value == "derived"

    assert ItemStatus.PARSED.value == "parsed"
    assert ItemStatus.CHECKED.value == "checked"
    assert ItemStatus.PENDING_VERIFICATION.value == "pending_verification"
    assert ItemStatus.FAILED.value == "failed"


def test_optional_fields_are_none():
    """Verify optional fields default to None, not invented values."""
    doc = DocumentMeta(
        document_id="test",
        document_family_id="fam",
        edition_id="ed",
        title="Test",
        doc_type=DocType.OTHER,
        original_sha256="x" * 64,
        total_pdf_pages=1,
        parser_name="p",
        parser_version="1",
        parse_run_id="run",
    )
    # These should be None, not empty strings or fake dates
    assert doc.publisher is None
    assert doc.jurisdiction is None
    assert doc.language is None
    assert doc.publication_date is None
    assert doc.revision_date is None
    assert doc.source_url is None
    assert doc.retrieved_at is None

    item = ContentItem(
        item_id="item-1",
        document_id="doc-1",
        edition_id="ed-1",
        type=ItemType.PARAGRAPH,
        text="Test text",
        pdf_page=1,
    )
    assert item.parent_id is None
    assert item.number is None
    assert item.latex is None
    assert item.table_html is None
    assert item.printed_page is None
    assert item.bbox is None
    assert item.derived_info is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])