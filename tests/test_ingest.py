"""
Tests for ingest pipeline (with monkeypatched fake parser).
"""

import json
import tempfile
from pathlib import Path
from datetime import date
from unittest.mock import patch, MagicMock

import pytest

from civileng_anything.config import Settings
from civileng_anything.store import Store
from civileng_anything.schema import DocumentMeta, DocType, PageStatusEnum
from civileng_anything.ingest import Ingestor, IngestReport
from civileng_anything.parsers.mineru_adapter import MineruResult


# Fake MinerU content_list for testing
FAKE_CONTENT_LIST = [
    {
        "type": "title",
        "text": "Test Document",
        "level": 1,
        "page_idx": 0,
    },
    {
        "type": "text",
        "text": "1.1 This is a test clause with fcu reference.",
        "page_idx": 0,
    },
    {
        "type": "text",
        "text": "1.2 Another clause.",
        "page_idx": 1,
    },
]


def create_fake_pdf(tmp_path: Path) -> Path:
    """Create a minimal fake PDF for testing."""
    from pypdf import PdfWriter

    pdf_path = tmp_path / "test.pdf"
    writer = PdfWriter()
    writer.add_blank_page(width=612, height=792)  # Page 1
    writer.add_blank_page(width=612, height=792)  # Page 2
    with open(pdf_path, "wb") as f:
        writer.write(f)
    return pdf_path


def create_fake_mineru_output(out_dir: Path, content_list: list, pages_done: list[int]):
    """Create a fake MinerU output structure."""
    # MinerU creates: out_dir/<pdf_stem>/auto/content_list.json
    pdf_stem = "test"
    method_dir = out_dir / pdf_stem / "auto"
    method_dir.mkdir(parents=True, exist_ok=True)

    # Write content_list.json
    with open(method_dir / "content_list.json", "w", encoding="utf-8") as f:
        json.dump(content_list, f)

    # Create a dummy markdown file
    with open(method_dir / "test.md", "w") as f:
        f.write("# Test\n")

    # Create images dir
    images_dir = method_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    return MineruResult(
        content_list_path=method_dir / "content_list.json",
        markdown_path=method_dir / "test.md",
        images_dir=images_dir,
        raw_dir=method_dir,
        pages_done=pages_done,
    )


def test_ingest_first_time(tmp_path):
    """Test first-time ingest of a document."""
    # Setup
    settings = Settings(data_dir=tmp_path / "data")
    settings.ensure_dirs()
    store = Store(settings.db_path)

    pdf_path = create_fake_pdf(tmp_path)

    meta = DocumentMeta(
        document_id="test-doc-001",
        document_family_id="test-family",
        edition_id="ed-1",
        title="Test Document",
        doc_type=DocType.GOVERNMENT_CODE,
        original_sha256="",
        total_pdf_pages=0,
        parser_name="mineru",
        parser_version="3.4.5",
        parser_config={},
        parse_run_id="",
        completeness=0.0,
        qc_status="pending",
        amendment_relations=[],
        version_relations=[],
    )

    ingestor = Ingestor(settings, store)

    # Mock run_mineru to return fake output
    with patch("civileng_anything.ingest.run_mineru") as mock_mineru:
        mock_mineru.return_value = create_fake_mineru_output(
            settings.parsed_dir / "test-doc-001" / "run-xxx" / "batch_0",
            FAKE_CONTENT_LIST,
            pages_done=[1, 2],
        )

        report = ingestor.ingest_file(pdf_path, meta)

    # Assertions
    assert report.deduplicated is False
    assert report.document_id == "test-doc-001"
    assert report.total_pages == 2
    assert set(report.pages_done) == {1, 2}
    assert report.pages_failed == []
    assert report.items_created == 3  # title + 2 clauses
    assert report.run_id.startswith("run-")

    # Check document stored
    doc = store.get_document("test-doc-001")
    assert doc is not None
    assert doc.original_sha256 == report.sha256
    assert doc.total_pdf_pages == 2
    assert doc.parse_run_id == report.run_id

    # Check items stored
    items = store.items_by_number("test-doc-001", "ed-1", "clause", "1.1")
    assert len(items) == 1
    assert "fcu" in items[0].text

    # Check page statuses
    page_statuses = store.get_page_statuses("test-doc-001")
    assert len(page_statuses) == 2
    assert all(ps.status == PageStatusEnum.DONE for ps in page_statuses)

    # Check run recorded
    runs = store.get_runs_for_document("test-doc-001")
    assert len(runs) == 1
    assert runs[0]["run_id"] == report.run_id
    assert runs[0]["status"] == "completed"

    store.close()


def test_ingest_deduplication(tmp_path):
    """Test that re-ingesting the same file deduplicates."""
    settings = Settings(data_dir=tmp_path / "data")
    settings.ensure_dirs()
    store = Store(settings.db_path)

    pdf_path = create_fake_pdf(tmp_path)

    meta = DocumentMeta(
        document_id="test-doc-002",
        document_family_id="test-family",
        edition_id="ed-1",
        title="Test Document",
        doc_type=DocType.GOVERNMENT_CODE,
        original_sha256="",
        total_pdf_pages=0,
        parser_name="mineru",
        parser_version="3.4.5",
        parser_config={},
        parse_run_id="",
        completeness=0.0,
        qc_status="pending",
        amendment_relations=[],
        version_relations=[],
    )

    ingestor = Ingestor(settings, store)

    with patch("civileng_anything.ingest.run_mineru") as mock_mineru:
        mock_mineru.return_value = create_fake_mineru_output(
            settings.parsed_dir / "test-doc-002" / "run-xxx" / "batch_0",
            FAKE_CONTENT_LIST,
            pages_done=[1, 2],
        )

        # First ingest
        report1 = ingestor.ingest_file(pdf_path, meta)
        assert report1.deduplicated is False
        assert report1.items_created == 3

        # Second ingest (same file, same sha256)
        report2 = ingestor.ingest_file(pdf_path, meta)
        assert report2.deduplicated is True
        assert report2.items_created == 0
        assert report2.document_id == report1.document_id
        assert report2.run_id == report1.run_id

    store.close()


def test_ingest_resume_missing_page(tmp_path):
    """Test resume behavior when one page is missing from previous run."""
    settings = Settings(data_dir=tmp_path / "data")
    settings.ensure_dirs()
    store = Store(settings.db_path)

    pdf_path = create_fake_pdf(tmp_path)

    meta = DocumentMeta(
        document_id="test-doc-003",
        document_family_id="test-family",
        edition_id="ed-1",
        title="Test Document",
        doc_type=DocType.GOVERNMENT_CODE,
        original_sha256="",
        total_pdf_pages=0,
        parser_name="mineru",
        parser_version="3.4.5",
        parser_config={},
        parse_run_id="",
        completeness=0.0,
        qc_status="pending",
        amendment_relations=[],
        version_relations=[],
    )

    ingestor = Ingestor(settings, store)

    with patch("civileng_anything.ingest.run_mineru") as mock_mineru:
        # First run: MinerU processes pages 1-2 as a batch, but only returns page 1
        def side_effect_first(pdf_path, out_dir, start_page_1based, end_page_1based, **kwargs):
            # Batch is pages 1-2, but only page 1 succeeds
            return create_fake_mineru_output(out_dir, FAKE_CONTENT_LIST[:2], pages_done=[1])

        mock_mineru.side_effect = side_effect_first
        report1 = ingestor.ingest_file(pdf_path, meta)

        # Page 1 done, page 2 failed (no output)
        assert report1.pages_done == [1]
        assert report1.pages_failed == [2]
        assert report1.items_created == 2

        # Second run: should only parse page 2 (page 1 already done)
        mock_mineru.side_effect = None
        mock_mineru.return_value = create_fake_mineru_output(
            settings.parsed_dir / "test-doc-003" / "run-yyy" / "batch_0",
            FAKE_CONTENT_LIST[2:],
            pages_done=[2],
        )

        report2 = ingestor.ingest_file(pdf_path, meta)

        # Should have parsed only page 2
        assert 1 in report2.pages_done  # Already done
        assert 2 in report2.pages_done  # Now done
        assert report2.pages_failed == []
        assert report2.items_created == 1  # Only the new clause

    store.close()


def test_ingest_partial_failure(tmp_path):
    """Test that partial failure is recorded correctly."""
    settings = Settings(data_dir=tmp_path / "data")
    settings.ensure_dirs()
    store = Store(settings.db_path)

    pdf_path = create_fake_pdf(tmp_path)

    meta = DocumentMeta(
        document_id="test-doc-004",
        document_family_id="test-family",
        edition_id="ed-1",
        title="Test Document",
        doc_type=DocType.GOVERNMENT_CODE,
        original_sha256="",
        total_pdf_pages=0,
        parser_name="mineru",
        parser_version="3.4.5",
        parser_config={},
        parse_run_id="",
        completeness=0.0,
        qc_status="pending",
        amendment_relations=[],
        version_relations=[],
    )

    ingestor = Ingestor(settings, store)

    with patch("civileng_anything.ingest.run_mineru") as mock_mineru:
        # First batch succeeds, second fails
        def side_effect(pdf_path, out_dir, start_page_1based, end_page_1based, **kwargs):
            if start_page_1based == 1:
                return create_fake_mineru_output(out_dir, FAKE_CONTENT_LIST[:2], pages_done=[1])
            else:
                raise RuntimeError("MinerU crashed")

        mock_mineru.side_effect = side_effect

        report = ingestor.ingest_file(pdf_path, meta)

        assert report.pages_done == [1]
        assert report.pages_failed == [2]
        assert report.items_created == 2

        # Check page statuses
        page_statuses = store.get_page_statuses("test-doc-004")
        assert page_statuses[0].status == PageStatusEnum.DONE
        assert page_statuses[1].status == PageStatusEnum.FAILED

    store.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])