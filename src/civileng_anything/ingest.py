"""
Ingest pipeline for CivilEng-Anything.

Implements the ARCHITECTURE.md state machine:
register -> store original -> inspect PDF -> parse (MinerU) -> archive raw ->
normalize -> per-page QC -> chunk & link -> build indexes -> publish dataset version
"""

import hashlib
import json
import logging
import shutil
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

from .config import Settings
from .schema import DocumentMeta, ContentItem, PageStatus, DocType, PageStatusEnum
from .store import Store
from .pdfinspect import inspect_pdf, PdfInfo
from .parsers.mineru_adapter import run_mineru, MineruResult
from .parsers.docling_adapter import run_docling, DoclingResult, ParserNotAvailable
from .normalize import normalize_mineru

logger = logging.getLogger(__name__)


@dataclass
class IngestReport:
    """Report from an ingest operation."""
    document_id: str
    sha256: str
    total_pages: int
    pages_done: list[int]
    pages_failed: list[int]
    items_created: int
    deduplicated: bool
    parser: str
    run_id: str

    def to_dict(self) -> dict:
        return {
            "document_id": self.document_id,
            "sha256": self.sha256,
            "total_pages": self.total_pages,
            "pages_done": self.pages_done,
            "pages_failed": self.pages_failed,
            "items_created": self.items_created,
            "deduplicated": self.deduplicated,
            "parser": self.parser,
            "run_id": self.run_id,
        }


class Ingestor:
    """
    Ingest pipeline for engineering documents.

    Handles deduplication, PDF inspection, MinerU parsing in batches,
    normalization, asset management, and resumability.
    """

    def __init__(self, settings: Settings, store: Optional[Store] = None):
        self.settings = settings
        self.store = store or Store(settings.db_path)
        self.settings.ensure_dirs()

    def ingest_file(
        self,
        pdf_path: Path,
        meta: DocumentMeta,
        source_url: Optional[str] = None,
    ) -> IngestReport:
        """
        Ingest a PDF file into the canonical store.

        Args:
            pdf_path: Path to the PDF file.
            meta: Document metadata (must have document_id, edition_id, etc.)
            source_url: Optional source URL for the document.

        Returns:
            IngestReport with results.
        """
        pdf_path = Path(pdf_path)

        # Compute SHA-256 of the file
        sha256 = self._compute_sha256(pdf_path)
        logger.info("File SHA-256: %s", sha256)

        # Check for deduplication - only skip if document is already complete
        existing = self.store.get_document_by_sha256(sha256)
        if existing and existing.completeness >= 1.0:
            logger.info("Document with SHA-256 %s already complete as %s", sha256, existing.document_id)
            return IngestReport(
                document_id=existing.document_id,
                sha256=sha256,
                total_pages=existing.total_pdf_pages,
                pages_done=[],
                pages_failed=[],
                items_created=0,
                deduplicated=True,
                parser=existing.parser_name,
                run_id=existing.parse_run_id,
            )
        elif existing:
            # Document exists but incomplete - resume ingest
            logger.info("Document with SHA-256 %s exists but incomplete (%s), resuming...",
                        sha256, existing.document_id)
            # Use the existing document's ID and edition
            meta.document_id = existing.document_id
            meta.edition_id = existing.edition_id

        # Store original (write-once)
        original_path = self.settings.originals_dir / f"{sha256}.pdf"
        if original_path.exists():
            # Verify it's the same file
            existing_sha = self._compute_sha256(original_path)
            if existing_sha != sha256:
                raise RuntimeError(
                    f"Collision: different file already exists at {original_path}"
                )
        else:
            shutil.copy2(pdf_path, original_path)
            logger.info("Stored original at %s", original_path)

        # Inspect PDF
        pdf_info = inspect_pdf(pdf_path)
        total_pages = pdf_info.page_count
        logger.info("PDF: %d pages, classification: %s", total_pages, pdf_info.classification)

        # Upsert document FIRST (before creating run, due to FK constraint)
        meta.original_sha256 = sha256
        meta.total_pdf_pages = total_pages
        meta.parse_run_id = ""  # Will be updated after run creation
        meta.completeness = 0.0
        meta.qc_status = "pending"
        self.store.upsert_document(meta)

        # Generate run ID
        run_id = f"run-{uuid.uuid4().hex[:8]}"

        # Create run record
        parser_name = meta.parser_name
        parser_version = meta.parser_version
        parser_config = meta.parser_config
        self.store.create_run(run_id, meta.document_id, parser_name, parser_version, parser_config)

        # Determine which pages need parsing (resumability)
        done_pages = self.store.get_done_pages(meta.document_id)
        failed_pages = self.store.get_failed_pages(meta.document_id)

        # Pages to parse: all pages minus done pages
        pages_to_parse = [p for p in range(1, total_pages + 1) if p not in done_pages]

        if not pages_to_parse:
            logger.info("All pages already parsed for %s", meta.document_id)
            self.store.update_run(run_id, status="completed", pages_done=sorted(done_pages))
            return IngestReport(
                document_id=meta.document_id,
                sha256=sha256,
                total_pages=total_pages,
                pages_done=sorted(done_pages),
                pages_failed=sorted(failed_pages),
                items_created=0,
                deduplicated=False,
                parser=parser_name,
                run_id=run_id,
            )

        # Parse in batches of max 40 pages
        batch_size = 40
        all_pages_done = list(done_pages)
        all_pages_failed = list(failed_pages)
        all_items = []
        items_created = 0

        for i in range(0, len(pages_to_parse), batch_size):
            batch = pages_to_parse[i:i + batch_size]
            start_page = batch[0]
            end_page = batch[-1]

            logger.info("Parsing pages %d-%d of %s", start_page, end_page, meta.document_id)

            # Run MinerU for this batch
            batch_out_dir = self.settings.parsed_dir / meta.document_id / run_id / f"batch_{i // batch_size}"
            try:
                result = run_mineru(
                    pdf_path=original_path,
                    out_dir=batch_out_dir,
                    start_page_1based=start_page,
                    end_page_1based=end_page,
                    lang="en",
                    method="auto",
                )
            except Exception as e:
                logger.error("MinerU failed for pages %d-%d: %s", start_page, end_page, e)
                # Mark all pages in batch as failed
                for p in batch:
                    self.store.add_page_status(PageStatus(
                        document_id=meta.document_id,
                        pdf_page=p,
                        status=PageStatusEnum.FAILED,
                        parser=parser_name,
                        detail=str(e),
                    ))
                    all_pages_failed.append(p)
                continue

            # Archive raw output (already in batch_out_dir)

            # Normalize
            page_offset = start_page - 1  # MinerU page_idx 0-based corresponds to start_page
            try:
                with open(result.content_list_path, "r", encoding="utf-8") as f:
                    content_list = json.load(f)
            except Exception as e:
                logger.error("Failed to read content_list: %s", e)
                for p in batch:
                    self.store.add_page_status(PageStatus(
                        document_id=meta.document_id,
                        pdf_page=p,
                        status=PageStatusEnum.FAILED,
                        parser=parser_name,
                        detail=f"Failed to read output: {e}",
                    ))
                    all_pages_failed.append(p)
                continue

            items = normalize_mineru(
                content_list=content_list,
                document_id=meta.document_id,
                edition_id=meta.edition_id,
                page_offset=page_offset,
            )

            # Copy figure assets
            for item in items:
                if item.type.value == "figure" and item.asset_ids:
                    # asset_ids contains source paths; copy to assets dir
                    for src_path_str in item.asset_ids:
                        src_path = Path(src_path_str)
                        if src_path.exists():
                            dst_dir = self.settings.assets_dir / meta.document_id
                            dst_dir.mkdir(parents=True, exist_ok=True)
                            dst_path = dst_dir / src_path.name
                            shutil.copy2(src_path, dst_path)
                            # Update asset_id to the new path
                            # (In a full implementation, we'd register in assets table)

            # Store items
            self.store.add_items(items)
            items_created += len(items)
            all_items.extend(items)

            # Record page statuses
            pages_in_output = set(result.pages_done)
            for p in batch:
                if p in pages_in_output:
                    self.store.add_page_status(PageStatus(
                        document_id=meta.document_id,
                        pdf_page=p,
                        status=PageStatusEnum.DONE,
                        parser=parser_name,
                        detail="OK",
                    ))
                    all_pages_done.append(p)
                    # Remove from failed if it was previously failed
                    if p in all_pages_failed:
                        all_pages_failed.remove(p)
                else:
                    self.store.add_page_status(PageStatus(
                        document_id=meta.document_id,
                        pdf_page=p,
                        status=PageStatusEnum.FAILED,
                        parser=parser_name,
                        detail="No output from parser",
                    ))
                    if p not in all_pages_failed:
                        all_pages_failed.append(p)

            # Update run record
            self.store.update_run(
                run_id,
                pages_done=sorted(all_pages_done),
                pages_failed=sorted(all_pages_failed),
                items_created=items_created,
            )

        # Final run status
        final_status = "completed" if not all_pages_failed else "completed_with_failures"
        self.store.update_run(
            run_id,
            pages_done=sorted(all_pages_done),
            pages_failed=sorted(all_pages_failed),
            items_created=items_created,
            status=final_status,
        )

        # Update document metadata (parse_run_id, completeness, qc_status)
        meta.parse_run_id = run_id
        meta.completeness = len(all_pages_done) / total_pages if total_pages > 0 else 0.0
        meta.qc_status = "needs_review" if all_pages_failed else "parsed"
        self.store.upsert_document(meta)

        return IngestReport(
            document_id=meta.document_id,
            sha256=sha256,
            total_pages=total_pages,
            pages_done=sorted(all_pages_done),
            pages_failed=sorted(all_pages_failed),
            items_created=items_created,
            deduplicated=False,
            parser=parser_name,
            run_id=run_id,
        )

    def _compute_sha256(self, path: Path) -> str:
        """Compute SHA-256 hash of a file."""
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
        return h.hexdigest()