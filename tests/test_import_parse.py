"""
Tests for import-parse command (consuming existing MinerU output).
"""

import json
import tempfile
from pathlib import Path
from datetime import date
from unittest.mock import patch, MagicMock

import pytest

from civileng_anything.config import Settings
from civileng_anything.store import Store
from civileng_anything.schema import DocumentMeta, DocType, PageStatusEnum, ItemType, ContentOrigin, ItemStatus
from civileng_anything.import_parse import ImportParser, register_document_if_needed
from civileng_anything.ids import make_item_id, content_hash


# Minimal fake MinerU content_list for testing (3 pages)
FAKE_MINERU_CONTENT_LIST = [
    # Page 1
    {
        "type": "title",
        "text": "Test Document",
        "level": 1,
        "page_idx": 0,
        "bbox": [100, 100, 400, 150],
    },
    {
        "type": "text",
        "text": "9.2.1 Longitudinal reinforcement shall be designed per Table 9.2.",
        "page_idx": 0,
        "bbox": [100, 200, 500, 240],
    },
    # Page 2
    {
        "type": "heading",
        "text": "9.5 Shear Design",
        "level": 2,
        "page_idx": 1,
        "bbox": [100, 100, 350, 140],
    },
    {
        "type": "text",
        "text": "9.5.1 Longitudinal reinforcement requirements for shear.",
        "page_idx": 1,
        "bbox": [100, 180, 450, 220],
    },
    # Page 3
    {
        "type": "table",
        "text": "Table 9.2: Design Parameters",
        "page_idx": 2,
        "bbox": [100, 100, 500, 300],
    },
]


def create_fake_pdf(tmp_dir: Path, content: str = "") -> Path:
    """Create a minimal fake PDF for testing."""
    from pypdf import PdfWriter
    pdf_path = tmp_dir / "test.pdf"
    writer = PdfWriter()
    page = writer.add_blank_page(width=612, height=792)  # 1 page
    # Add some text to make PDFs different if content provided
    if content:
        # Note: pypdf doesn't easily support adding text to blank pages
        # We'll just use different page sizes to differentiate
        pass
    with open(pdf_path, "wb") as f:
        writer.write(f)
    return pdf_path


def create_fake_pdf_with_size(tmp_dir: Path, width: int, height: int) -> Path:
    """Create a fake PDF with specific page size."""
    from pypdf import PdfWriter
    pdf_path = tmp_dir / "test.pdf"
    writer = PdfWriter()
    writer.add_blank_page(width=width, height=height)
    with open(pdf_path, "wb") as f:
        writer.write(f)
    return pdf_path


def create_fake_mineru_dir(tmp_dir: Path) -> Path:
    """Create a fake MinerU output directory."""
    mineru_dir = tmp_dir / "mineru_out"
    mineru_dir.mkdir()
    
    content_list_path = mineru_dir / "CoP_content_list.json"
    with open(content_list_path, "w") as f:
        json.dump(FAKE_MINERU_CONTENT_LIST, f)
    
    # Create dummy markdown file
    (mineru_dir / "CoP.md").write_text("# Test Document\n\nContent here.")
    
    # Create dummy images dir
    images_dir = mineru_dir / "images"
    images_dir.mkdir()
    
    return mineru_dir


class TestImportParse:
    """Tests for ImportParser.import_parse()."""

    def test_import_parse_synthetic(self):
        """Test importing a synthetic MinerU output."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            
            # Create fake PDF and MinerU output
            pdf_path = create_fake_pdf(tmp_path)
            mineru_dir = create_fake_mineru_dir(tmp_path)
            
            # Use a separate test database
            db_path = tmp_path / "test.db"
            settings = Settings(data_dir=tmp_path)
            store = Store(settings.db_path)
            
            try:
                # Register document
                doc = register_document_if_needed(
                    settings=settings,
                    store=store,
                    document_id="test-doc",
                    edition_id="ed1",
                    title="Test Document",
                    pdf_path=pdf_path,
                    publisher="Test Publisher",
                    jurisdiction="Test Jurisdiction",
                    doc_type=DocType.GOVERNMENT_CODE,
                    language="en",
                )
                assert doc.document_id == "test-doc"
                assert doc.total_pdf_pages == 1  # Our fake PDF has 1 page
                
                # Import parse
                importer = ImportParser(settings, store)
                report = importer.import_parse(
                    document_id="test-doc",
                    edition_id="ed1",
                    mineru_dir=mineru_dir,
                    parser_name="mineru",
                    parser_version="3.4.5",
                    parse_location="Test Location",
                    parse_date="2026-11-21",
                    expected_sha256=None,  # Skip hash check for synthetic test
                )
                
                # Verify report
                assert report.document_id == "test-doc"
                assert report.sha256 == doc.original_sha256
                assert report.items_created > 0
                assert len(report.pages_done) > 0
                
                # Verify items were stored
                conn = store._get_conn()
                cursor = conn.execute(
                    "SELECT COUNT(*) as cnt FROM items WHERE document_id = ? AND edition_id = ?",
                    ("test-doc", "ed1")
                )
                count = cursor.fetchone()["cnt"]
                assert count > 0
                
                # Verify page statuses
                page_statuses = store.get_page_statuses("test-doc")
                done_pages = [ps for ps in page_statuses if ps.status == PageStatusEnum.DONE]
                assert len(done_pages) > 0
                
                # Verify archive was created
                archive_dir = settings.parsed_dir / "test-doc" / report.run_id / "full_book"
                assert archive_dir.exists()
                assert (archive_dir / "import_manifest.json").exists()
                assert (archive_dir / "CoP_content_list.json").exists()
                
                # Verify manifest content
                with open(archive_dir / "import_manifest.json") as f:
                    manifest = json.load(f)
                assert manifest["document_id"] == "test-doc"
                assert manifest["parser_name"] == "mineru"
                assert manifest["parser_version"] == "3.4.5"
                assert manifest["parse_location"] == "Test Location"
                
            finally:
                store.close()

    def test_import_parse_deduplication(self):
        """Test that re-importing the same MinerU output is idempotent."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            
            pdf_path = create_fake_pdf(tmp_path)
            mineru_dir = create_fake_mineru_dir(tmp_path)
            
            db_path = tmp_path / "test.db"
            settings = Settings(data_dir=tmp_path)
            store = Store(settings.db_path)
            
            try:
                # Register document
                register_document_if_needed(
                    settings=settings,
                    store=store,
                    document_id="test-doc-2",
                    edition_id="ed1",
                    title="Test Document 2",
                    pdf_path=pdf_path,
                )
                
                importer = ImportParser(settings, store)
                
                # First import
                report1 = importer.import_parse(
                    document_id="test-doc-2",
                    edition_id="ed1",
                    mineru_dir=mineru_dir,
                    parse_date="2026-11-21",
                    expected_sha256=None,  # Skip hash check for synthetic test
                )
                
                # Second import (should be idempotent)
                report2 = importer.import_parse(
                    document_id="test-doc-2",
                    edition_id="ed1",
                    mineru_dir=mineru_dir,
                    parse_date="2026-11-21",
                    expected_sha256=None,
                )
                
                # Items should not be duplicated
                conn = store._get_conn()
                cursor = conn.execute(
                    "SELECT COUNT(*) as cnt FROM items WHERE document_id = ? AND edition_id = ?",
                    ("test-doc-2", "ed1")
                )
                count = cursor.fetchone()["cnt"]
                # The number of items should be the same after second import
                # (exact count depends on normalization, but should not double)
                assert count == report1.items_created
                
            finally:
                store.close()

    def test_import_parse_hash_mismatch(self):
        """Test that import refuses when source hash doesn't match."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            
            pdf_path = create_fake_pdf(tmp_path)
            mineru_dir = create_fake_mineru_dir(tmp_path)
            
            db_path = tmp_path / "test.db"
            settings = Settings(data_dir=tmp_path)
            store = Store(settings.db_path)
            
            try:
                # Register document with a different PDF (different hash)
                # First create a different PDF in a subdirectory
                diff_dir = tmp_path / "different"
                diff_dir.mkdir()
                pdf_path2 = create_fake_pdf(diff_dir)
                
                register_document_if_needed(
                    settings=settings,
                    store=store,
                    document_id="test-doc-3",
                    edition_id="ed1",
                    title="Test Document 3",
                    pdf_path=pdf_path2,  # Different PDF
                )
                
                importer = ImportParser(settings, store)
                
                # Import should fail because the MinerU output was from a different PDF
                with pytest.raises(ValueError, match="Source hash mismatch"):
                    importer.import_parse(
                        document_id="test-doc-3",
                        edition_id="ed1",
                        mineru_dir=mineru_dir,
                    )
                
            finally:
                store.close()

    def test_import_parse_unregistered_document(self):
        """Test that import fails for unregistered document without --register-if-missing."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            
            mineru_dir = create_fake_mineru_dir(tmp_path)
            
            settings = Settings(data_dir=tmp_path)
            store = Store(settings.db_path)
            
            importer = ImportParser(settings, store)
            
            # Import should fail because document is not registered
            with pytest.raises(ValueError, match="Document not registered"):
                importer.import_parse(
                    document_id="unregistered-doc",
                    edition_id="ed1",
                    mineru_dir=mineru_dir,
                )
            
            store.close()

    def test_register_document_if_needed_creates_new(self):
        """Test register_document_if_needed creates new document."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            
            pdf_path = create_fake_pdf(tmp_path)
            
            settings = Settings(data_dir=tmp_path)
            store = Store(settings.db_path)
            
            doc = register_document_if_needed(
                settings=settings,
                store=store,
                document_id="new-doc",
                edition_id="ed1",
                title="New Document",
                pdf_path=pdf_path,
                publisher="Test Publisher",
                jurisdiction="Test Jurisdiction",
                doc_type=DocType.GOVERNMENT_CODE,
                language="en",
            )
            
            assert doc.document_id == "new-doc"
            assert doc.edition_id == "ed1"
            assert doc.title == "New Document"
            assert doc.publisher == "Test Publisher"
            assert doc.jurisdiction == "Test Jurisdiction"
            assert doc.doc_type == DocType.GOVERNMENT_CODE
            assert doc.language == "en"
            assert doc.total_pdf_pages == 1
            assert doc.original_sha256 is not None
            assert len(doc.original_sha256) == 64  # SHA-256 hex
            
            # Verify original was stored
            original_path = settings.originals_dir / f"{doc.original_sha256}.pdf"
            assert original_path.exists()
            
            store.close()

    def test_register_document_if_needed_existing_matches(self):
        """Test register_document_if_needed returns existing when hash matches."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            
            pdf_path = create_fake_pdf(tmp_path)
            
            settings = Settings(data_dir=tmp_path)
            store = Store(settings.db_path)
            
            # Register first time
            doc1 = register_document_if_needed(
                settings=settings,
                store=store,
                document_id="existing-doc",
                edition_id="ed1",
                title="Existing Document",
                pdf_path=pdf_path,
            )
            
            # Register again with same PDF - should return same document
            doc2 = register_document_if_needed(
                settings=settings,
                store=store,
                document_id="existing-doc",
                edition_id="ed1",
                title="Existing Document",
                pdf_path=pdf_path,
            )
            
            assert doc1.document_id == doc2.document_id
            assert doc1.original_sha256 == doc2.original_sha256
            
            store.close()

    def test_register_document_if_needed_hash_mismatch(self):
        """Test register_document_if_needed fails when hash differs."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            
            # Create two different PDFs in separate subdirectories with different sizes
            pdf1_dir = tmp_path / "pdf1"
            pdf1_dir.mkdir()
            pdf2_dir = tmp_path / "pdf2"
            pdf2_dir.mkdir()
            
            pdf_path1 = create_fake_pdf_with_size(pdf1_dir, 612, 792)
            pdf_path2 = create_fake_pdf_with_size(pdf2_dir, 595, 842)  # A4 size
            
            settings = Settings(data_dir=tmp_path)
            store = Store(settings.db_path)
            
            # Register with first PDF
            register_document_if_needed(
                settings=settings,
                store=store,
                document_id="mismatch-doc",
                edition_id="ed1",
                title="Mismatch Document",
                pdf_path=pdf_path1,
            )
            
            # Try to register with different PDF - should fail
            with pytest.raises(ValueError, match="different SHA-256"):
                register_document_if_needed(
                    settings=settings,
                    store=store,
                    document_id="mismatch-doc",
                    edition_id="ed1",
                    title="Mismatch Document",
                    pdf_path=pdf_path2,
                )
            
            store.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])