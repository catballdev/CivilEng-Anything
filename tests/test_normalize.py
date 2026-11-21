"""
Tests for normalize module (MinerU content_list -> ContentItem).
"""

import pytest
from civileng_anything.normalize import normalize_mineru, extract_cross_refs
from civileng_anything.schema import ItemType, ContentOrigin, ItemStatus


# Hand-written MinerU content_list fixture (2 pages)
MINERU_CONTENT_LIST = [
    # Page 1: Heading
    {
        "type": "title",
        "text": "Chapter 9: Structural Analysis",
        "level": 1,
        "page_idx": 0,
        "bbox": [100, 100, 400, 150],
    },
    # Page 1: Section heading
    {
        "type": "heading",
        "text": "9.2 Load Combinations",
        "level": 2,
        "page_idx": 0,
        "bbox": [100, 180, 350, 220],
    },
    # Page 1: Clause with cross-reference
    {
        "type": "text",
        "text": "9.2.1 The design shall consider all load combinations per Table 9.2 and clause 9.1.3.",
        "page_idx": 0,
        "bbox": [100, 250, 500, 290],
    },
    # Page 1: Equation
    {
        "type": "equation",
        "text": "U = 1.2D + 1.6L (Eq. 9.2)",
        "latex": "U = 1.2D + 1.6L",
        "page_idx": 0,
        "bbox": [150, 320, 450, 360],
    },
    # Page 2: Table
    {
        "type": "table",
        "html": "<table><thead><tr><th>Combo</th><th>Factor</th></tr></thead><tbody><tr><td>1.2D+1.6L</td><td>1.0</td></tr></tbody></table>",
        "caption": "Table 9.2: Load Combination Factors",
        "page_idx": 1,
        "bbox": [100, 150, 500, 300],
    },
    # Page 2: Image/Figure
    {
        "type": "image",
        "img_path": "/tmp/mineru_output/images/figure_9_1.png",
        "caption": "Figure 9.1: Load Combination Diagram",
        "page_idx": 1,
        "bbox": [100, 350, 500, 550],
    },
]


def test_normalize_mineru_basic():
    """Test basic normalization of MinerU content_list."""
    items = normalize_mineru(
        content_list=MINERU_CONTENT_LIST,
        document_id="test-doc-001",
        edition_id="ed-1",
        page_offset=0,
    )

    # Should have 6 items
    assert len(items) == 6

    # Check item 0: section (title)
    assert items[0].type == ItemType.SECTION
    assert items[0].text == "Chapter 9: Structural Analysis"
    assert items[0].section_path == ["Chapter 9: Structural Analysis"]
    assert items[0].number == "9"  # Extracted from text
    assert items[0].pdf_page == 1
    assert items[0].content_origin == ContentOrigin.ORIGINAL
    assert items[0].status == ItemStatus.PARSED

    # Check item 1: section (heading)
    assert items[1].type == ItemType.SECTION
    assert items[1].text == "9.2 Load Combinations"
    assert items[1].section_path == ["Chapter 9: Structural Analysis", "9.2 Load Combinations"]
    assert items[1].number == "9.2"
    assert items[1].pdf_page == 1

    # Check item 2: clause (starts with clause number pattern)
    assert items[2].type == ItemType.CLAUSE
    assert items[2].number == "9.2.1"
    assert "Table 9.2" in items[2].text
    assert "clause 9.1.3" in items[2].text
    assert items[2].pdf_page == 1
    # Cross-refs should be extracted
    assert "Table 9.2" in items[2].cross_refs
    assert "Clause 9.1.3" in items[2].cross_refs

    # Check item 3: equation
    assert items[3].type == ItemType.EQUATION
    assert items[3].latex == "U = 1.2D + 1.6L"
    assert items[3].number == "9.2"  # Extracted from (Eq. 9.2)
    assert items[3].pdf_page == 1

    # Check item 4: table
    assert items[4].type == ItemType.TABLE
    assert items[4].table_html is not None
    assert "<table>" in items[4].table_html
    assert items[4].number == "9.2"  # From caption "Table 9.2"
    assert items[4].pdf_page == 2

    # Check item 5: figure
    assert items[5].type == ItemType.FIGURE
    assert items[5].number == "9.1"  # From caption "Figure 9.1"
    assert items[5].asset_ids == ["/tmp/mineru_output/images/figure_9_1.png"]
    assert items[5].pdf_page == 2


def test_normalize_mineru_page_offset():
    """Test page_offset is correctly applied."""
    items = normalize_mineru(
        content_list=MINERU_CONTENT_LIST,
        document_id="test-doc-001",
        edition_id="ed-1",
        page_offset=10,  # This batch starts at page 11
    )

    # First item should be on page 11 (0 + 1 + 10)
    assert items[0].pdf_page == 11
    # Table should be on page 12 (1 + 1 + 10)
    assert items[4].pdf_page == 12


def test_extract_cross_refs():
    """Test cross-reference extraction."""
    text = "As per clause 9.2.1 and Table 3.2, see Figure 9.5 or equation 6.33."
    refs = extract_cross_refs(text)

    # Should find all four reference types
    assert "Clause 9.2.1" in refs
    assert "Table 3.2" in refs
    assert "Figure 9.5" in refs
    assert "Equation 6.33" in refs

    # Case insensitive
    text2 = "CLAUSE 1.1 and table 2.2"
    refs2 = extract_cross_refs(text2)
    assert "Clause 1.1" in refs2
    assert "Table 2.2" in refs2

    # Abbreviated forms
    text3 = "Cl. 5.3 and Fig. 2.1, Eq. 4.2"
    refs3 = extract_cross_refs(text3)
    assert "Clause 5.3" in refs3
    assert "Figure 2.1" in refs3
    assert "Equation 4.2" in refs3

    # Deduplication
    text4 = "clause 1.1 and Clause 1.1"
    refs4 = extract_cross_refs(text4)
    assert refs4.count("Clause 1.1") == 1


def test_unknown_fields_are_none():
    """Verify unknown/optional fields are None, not invented."""
    items = normalize_mineru(
        content_list=MINERU_CONTENT_LIST,
        document_id="test-doc-001",
        edition_id="ed-1",
        page_offset=0,
    )

    for item in items:
        assert item.parent_id is None  # Not tracked in this simple version
        assert item.printed_page is None
        assert item.derived_info is None
        assert item.footnote_ids == []
        assert item.amendment_links == []


def test_item_ids_are_stable():
    """Test that item IDs are stable and follow the expected format."""
    items = normalize_mineru(
        content_list=MINERU_CONTENT_LIST,
        document_id="test-doc-001",
        edition_id="ed-1",
        page_offset=0,
    )

    # All items should have proper IDs
    for i, item in enumerate(items):
        expected_id = f"test-doc-001-p{item.pdf_page:04d}-{i+1:04d}"
        assert item.item_id == expected_id, f"Item {i}: got {item.item_id}, expected {expected_id}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])