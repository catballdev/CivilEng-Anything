"""
Import existing MinerU parse output for CivilEng-Anything.

Consumes an existing MinerU output directory (full-book output) and imports it
into the canonical store, superseding any previous failed runs.
"""

import hashlib
import json
import logging
import shutil
import uuid
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Optional

from .config import Settings
from .schema import DocumentMeta, ContentItem, PageStatus, PageStatusEnum, DocType, ItemStatus, ContentOrigin
from .store import Store
from .normalize import normalize_mineru
from .ids import make_item_id, content_hash

logger = logging.getLogger(__name__)

# Sentinel for optional expected_sha256 parameter
_SKIP_HASH_CHECK = object()


@dataclass
class ImportParseReport:
    """Report from an import-parse operation."""
    document_id: str
    sha256: str
    total_pages: int
    pages_done: list[int]
    pages_failed: list[int]
    items_created: int
    items_updated: int
    items_skipped: int
    deduplicated: bool
    parser: str
    parser_version: str
    parse_location: str
    parse_date: str
    run_id: str

    def to_dict(self) -> dict:
        return {
            "document_id": self.document_id,
            "sha256": self.sha256,
            "total_pages": self.total_pages,
            "pages_done": self.pages_done,
            "pages_failed": self.pages_failed,
            "items_created": self.items_created,
            "items_updated": self.items_updated,
            "items_skipped": self.items_skipped,
            "deduplicated": self.deduplicated,
            "parser": self.parser,
            "parser_version": self.parser_version,
            "parse_location": self.parse_location,
            "parse_date": self.parse_date,
            "run_id": self.run_id,
        }


class ImportParser:
    """
    Import an existing MinerU full-book parse output.
    
    This supersedes any previous failed runs by:
    - Verifying document registration and source hash match
    - Clearing previous page statuses and items for the edition
    - Archiving the raw MinerU output with provenance
    - Running the normalize pipeline on the full content_list
    - Building FTS index
    - Setting all 212 pages to 'done'
    """

    def __init__(self, settings: Settings, store: Optional[Store] = None):
        self.settings = settings
        self.store = store or Store(settings.db_path)
        self.settings.ensure_dirs()

    def import_parse(
        self,
        document_id: str,
        edition_id: str,
        mineru_dir: Path,
        parser_name: str = "mineru",
        parser_version: str = "3.4.5",
        parse_location: str = "Kaggle (GPU T4 x2)",
        parse_date: Optional[str] = None,
        expected_sha256: object = _SKIP_HASH_CHECK,
    ) -> ImportParseReport:
        """
        Import an existing MinerU parse output directory.
        
        Args:
            document_id: Document ID (must be already registered)
            edition_id: Edition ID
            mineru_dir: Path to MinerU output directory (e.g., .../auto/)
            parser_name: Parser name (default: mineru)
            parser_version: Parser version (default: 3.4.5)
            parse_location: Where the parse was run (default: Kaggle GPU)
            parse_date: Parse date in ISO format (default: today)
            expected_sha256: Expected SHA256 of source PDF (default: CoP SUC 2020 official;
                             pass None to skip hash verification for testing)
            
        Returns:
            ImportParseReport with results
            
        Raises:
            ValueError: If document not registered or hash mismatch
            RuntimeError: If import fails
        """
        if parse_date is None:
            parse_date = date.today().isoformat()
        
        # Default expected SHA256 for the specific CoP SUC 2020 document
        if expected_sha256 is _SKIP_HASH_CHECK:
            expected_sha256 = "2af81851e1db0a7ce9bdd96d6714885d7d6fc408861060aeec247935da1e5664"
        
        # 1. Verify document is registered
        doc = self.store.get_document(document_id)
        if doc is None:
            raise ValueError(
                f"Document not registered: {document_id}. "
                f"Register the document first using 'civileng ingest' or 'civileng register'."
            )
        
        # 2. Verify source hash matches the PDF the parse came from
        # Skip check if expected_sha256 is explicitly None (for testing)
        if expected_sha256 is not None and doc.original_sha256 != expected_sha256:
            raise ValueError(
                f"Source hash mismatch for document {document_id}.\n"
                f"  Registered: {doc.original_sha256}\n"
                f"  Expected:   {expected_sha256}\n"
                f"The MinerU output was generated from a different PDF. Import refused."
            )
        
        # 3. Verify the MinerU directory exists and has content_list.json
        mineru_dir = Path(mineru_dir)
        content_list_path = mineru_dir / "CoP_content_list.json"
        if not content_list_path.exists():
            # Try alternative naming
            content_list_paths = list(mineru_dir.glob("*_content_list.json"))
            if content_list_paths:
                content_list_path = content_list_paths[0]
            else:
                raise ValueError(f"content_list.json not found in {mineru_dir}")
        
        # 4. Load and validate content_list
        logger.info("Loading content_list from %s", content_list_path)
        with open(content_list_path, "r", encoding="utf-8") as f:
            content_list = json.load(f)
        
        # Extract page count from content_list
        pages_in_output = set()
        for entry in content_list:
            page_idx = entry.get("page_idx")
            if isinstance(page_idx, int):
                pages_in_output.add(page_idx + 1)  # Convert to 1-based
        
        total_pages_in_parse = len(pages_in_output)
        logger.info("MinerU output covers %d pages (1-based): %s", 
                    total_pages_in_parse, sorted(pages_in_output)[:10])
        
        if total_pages_in_parse != doc.total_pdf_pages:
            logger.warning(
                "Page count mismatch: document has %d pages, MinerU output has %d pages",
                doc.total_pdf_pages, total_pages_in_parse
            )
        
        # 5. Generate run ID for this import
        run_id = f"import-{uuid.uuid4().hex[:8]}"
        
        # 6. Create run record
        self.store.create_run(
            run_id, document_id, parser_name, parser_version,
            {
                "source": "import_parse",
                "mineru_dir": str(mineru_dir),
                "parse_location": parse_location,
                "parse_date": parse_date,
            }
        )
        
        # 7. Archive raw MinerU output under data/parsed/<document_id>/
        archive_dir = self.settings.parsed_dir / document_id / run_id / "full_book"
        archive_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy the entire mineru_dir to archive
        logger.info("Archiving MinerU output to %s", archive_dir)
        for item in mineru_dir.iterdir():
            if item.is_file():
                shutil.copy2(item, archive_dir / item.name)
            else:
                shutil.copytree(item, archive_dir / item.name, dirs_exist_ok=True)
        
        # Write provenance manifest
        manifest = {
            "document_id": document_id,
            "edition_id": edition_id,
            "parser_name": parser_name,
            "parser_version": parser_version,
            "parser_config": {"method": "pipeline", "backend": "pipeline"},
            "parse_location": parse_location,
            "parse_date": parse_date,
            "run_id": run_id,
            "source_sha256": expected_sha256,
            "mineru_dir": str(mineru_dir),
            "content_list_sha256": self._compute_sha256(content_list_path),
            "total_pages_in_parse": total_pages_in_parse,
            "total_items_in_content_list": len(content_list),
            "archived_at": datetime.now().isoformat(),
        }
        with open(archive_dir / "import_manifest.json", "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)
        
        # 8. Supersede failed local run: clear previous items and page statuses for this edition
        logger.info("Clearing previous items and page statuses for %s/%s", document_id, edition_id)
        self._clear_edition_data(document_id, edition_id)
        
        # 9. Normalize the full content_list (page_offset=0 for full book)
        logger.info("Normalizing MinerU content_list (%d entries)...", len(content_list))
        items = normalize_mineru(
            content_list=content_list,
            document_id=document_id,
            edition_id=edition_id,
            page_offset=0,
        )
        logger.info("Normalized to %d canonical items", len(items))
        
        # 10. Copy figure assets
        self._copy_figure_assets(items, mineru_dir, document_id)
        
        # 11. Store items (idempotent via ON CONFLICT)
        items_before = self._count_items(document_id, edition_id)
        self.store.add_items(items)
        items_after = self._count_items(document_id, edition_id)
        items_created = items_after - items_before
        items_updated = len(items) - items_created  # Simplified: items that already existed
        
        # 12. Set page statuses to DONE for all pages in the parse
        pages_done = sorted(pages_in_output)
        for page_num in pages_done:
            self.store.add_page_status(PageStatus(
                document_id=document_id,
                pdf_page=page_num,
                status=PageStatusEnum.DONE,
                parser=parser_name,
                detail=f"Imported from {parse_location} ({parse_date})",
            ))
        
        # 13. Update run record
        self.store.update_run(
            run_id,
            pages_done=pages_done,
            pages_failed=[],
            items_created=items_created,
            status="completed",
        )
        
        # 14. Update document metadata
        doc.parse_run_id = run_id
        doc.completeness = 1.0
        doc.qc_status = "parsed"
        doc.parser_name = parser_name
        doc.parser_version = parser_version
        self.store.upsert_document(doc)
        
        # 15. Build FTS index (already done by add_items via store)
        # The FTS is updated incrementally in add_items
        
        logger.info("Import complete: %d items, %d pages done", len(items), len(pages_done))
        
        return ImportParseReport(
            document_id=document_id,
            sha256=doc.original_sha256,
            total_pages=doc.total_pdf_pages,
            pages_done=pages_done,
            pages_failed=[],
            items_created=items_created,
            items_updated=items_updated,
            items_skipped=0,
            deduplicated=False,
            parser=parser_name,
            parser_version=parser_version,
            parse_location=parse_location,
            parse_date=parse_date,
            run_id=run_id,
        )
    
    def _clear_edition_data(self, document_id: str, edition_id: str) -> None:
        """Clear all items and page statuses for a document/edition."""
        with self.store._transaction() as conn:
            # Delete items for this document/edition
            conn.execute(
                "DELETE FROM items WHERE document_id = ? AND edition_id = ?",
                (document_id, edition_id)
            )
            # Delete FTS entries for this document/edition
            conn.execute(
                "DELETE FROM items_fts WHERE document_id = ? AND edition_id = ?",
                (document_id, edition_id)
            )
            # Delete page statuses for this document
            conn.execute(
                "DELETE FROM pages WHERE document_id = ?",
                (document_id,)
            )
            # Delete runs for this document (except we keep them for history)
            # We don't delete runs - they're historical records
    
    def _copy_figure_assets(self, items: list[ContentItem], mineru_dir: Path, document_id: str) -> None:
        """Copy figure assets from MinerU output to assets directory."""
        assets_dir = self.settings.assets_dir / document_id
        assets_dir.mkdir(parents=True, exist_ok=True)
        
        for item in items:
            if item.type.value == "figure" and item.asset_ids:
                for src_path_str in item.asset_ids:
                    src_path = Path(src_path_str)
                    if not src_path.is_absolute():
                        # Relative to mineru_dir
                        src_path = mineru_dir / src_path_str
                    
                    if src_path.exists():
                        dst_path = assets_dir / src_path.name
                        if not dst_path.exists():
                            shutil.copy2(src_path, dst_path)
                            logger.debug("Copied asset: %s -> %s", src_path, dst_path)
                        # Update asset_id to the new path (relative to assets dir)
                        # In a full implementation, we'd register in assets table
    
    def _count_items(self, document_id: str, edition_id: str) -> int:
        """Count items for a document/edition."""
        conn = self.store._get_conn()
        row = conn.execute(
            "SELECT COUNT(*) as cnt FROM items WHERE document_id = ? AND edition_id = ?",
            (document_id, edition_id)
        ).fetchone()
        return row["cnt"] if row else 0
    
    def _compute_sha256(self, path: Path) -> str:
        """Compute SHA-256 hash of a file."""
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
        return h.hexdigest()


def register_document_if_needed(
    settings: Settings,
    store: Store,
    document_id: str,
    edition_id: str,
    title: str,
    pdf_path: Path,
    document_family_id: Optional[str] = None,
    publisher: Optional[str] = None,
    jurisdiction: Optional[str] = None,
    doc_type: DocType = DocType.GOVERNMENT_CODE,
    language: str = "en",
    parser_name: str = "mineru",
    parser_version: str = "3.4.5",
) -> DocumentMeta:
    """
    Register a document if it doesn't exist, or verify it matches.
    
    This is a helper for the CLI to ensure the document is registered
    before import-parse.
    """
    # Compute SHA-256
    h = hashlib.sha256()
    with open(pdf_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    sha256 = h.hexdigest()
    
    # Check if document exists
    existing = store.get_document(document_id)
    if existing:
        if existing.original_sha256 != sha256:
            raise ValueError(
                f"Document {document_id} exists but has different SHA-256.\n"
                f"  Registered: {existing.original_sha256}\n"
                f"  File:       {sha256}"
            )
        return existing
    
    # Check if another document has this SHA-256
    by_sha = store.get_document_by_sha256(sha256)
    if by_sha:
        raise ValueError(
            f"Another document ({by_sha.document_id}) already has this SHA-256: {sha256}"
        )
    
    # Inspect PDF for page count
    from .pdfinspect import inspect_pdf
    pdf_info = inspect_pdf(pdf_path)
    
    # Create document metadata
    doc_family_id = document_family_id or document_id
    meta = DocumentMeta(
        document_id=document_id,
        document_family_id=doc_family_id,
        edition_id=edition_id,
        title=title,
        publisher=publisher,
        jurisdiction=jurisdiction,
        doc_type=doc_type,
        language=language,
        publication_date=None,
        revision_date=None,
        source_url=None,
        retrieved_at=date.today(),
        original_sha256=sha256,
        total_pdf_pages=pdf_info.page_count,
        parser_name=parser_name,
        parser_version=parser_version,
        parser_config={},
        parse_run_id="",
        completeness=0.0,
        qc_status="pending",
        amendment_relations=[],
        version_relations=[],
    )
    
    # Store original (write-once)
    original_path = settings.originals_dir / f"{sha256}.pdf"
    if original_path.exists():
        existing_sha = hashlib.sha256(original_path.read_bytes()).hexdigest()
        if existing_sha != sha256:
            raise RuntimeError(f"Collision: different file already exists at {original_path}")
    else:
        original_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(pdf_path, original_path)
        logger.info("Stored original at %s", original_path)
    
    # Upsert document
    store.upsert_document(meta)
    logger.info("Registered document: %s (%s)", document_id, edition_id)
    
    return meta