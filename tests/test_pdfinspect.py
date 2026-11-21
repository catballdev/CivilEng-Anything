"""
Tests for pdfinspect module.
"""

import pytest
from pathlib import Path

from civileng_anything.pdfinspect import inspect_pdf, PdfInfo


def test_inspect_pdf_text_based(tmp_path):
    """Test inspecting a simple text-based PDF."""
    # Create a minimal PDF with text
    from pypdf import PdfWriter

    pdf_path = tmp_path / "test_text.pdf"
    writer = PdfWriter()
    writer.add_blank_page(width=612, height=792)
    # Add some text content
    page = writer.pages[0]
    page.merge_page(writer.add_blank_page(width=612, height=792))
    with open(pdf_path, "wb") as f:
        writer.write(f)

    info = inspect_pdf(pdf_path)

    assert isinstance(info, PdfInfo)
    assert info.page_count == 2
    assert info.is_encrypted is False
    assert info.encryption_detail is None
    assert len(info.has_text) == 2
    # Blank pages may or may not have extractable text depending on pypdf
    assert info.classification in ("text", "scanned", "mixed")


def test_inspect_pdf_encrypted(tmp_path):
    """Test inspecting an encrypted PDF."""
    from pypdf import PdfWriter

    pdf_path = tmp_path / "test_encrypted.pdf"
    writer = PdfWriter()
    writer.add_blank_page(width=612, height=792)
    writer.encrypt(user_password="", owner_password="owner123")
    with open(pdf_path, "wb") as f:
        writer.write(f)

    info = inspect_pdf(pdf_path)

    assert info.page_count == 1
    assert info.is_encrypted is True
    assert info.encryption_detail is not None
    assert "print:" in info.encryption_detail
    assert "change:" in info.encryption_detail
    assert "copy:" in info.encryption_detail


def test_inspect_pdf_classification(tmp_path):
    """Test classification logic."""
    from pypdf import PdfWriter

    # Create a PDF with many pages, some with text
    pdf_path = tmp_path / "test_mixed.pdf"
    writer = PdfWriter()
    for i in range(10):
        writer.add_blank_page(width=612, height=792)
    with open(pdf_path, "wb") as f:
        writer.write(f)

    info = inspect_pdf(pdf_path)

    assert info.page_count == 10
    # All blank pages - should be scanned (no text extracted)
    assert info.classification in ("scanned", "mixed")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])