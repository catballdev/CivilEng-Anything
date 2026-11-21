"""
SQLite store for CivilEng-Anything canonical data.

Uses stdlib sqlite3 with FTS5 for full-text search.
"""

import json
import re
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Optional

from .schema import DocumentMeta, ContentItem, PageStatus, SearchResult
from .ids import content_hash

# jieba for Traditional Chinese tokenization (CJK pre-tokenization for FTS5 BM25)
try:
    import jieba
    import jieba.posseg as pseg
    JIEBA_AVAILABLE = True
except ImportError:
    JIEBA_AVAILABLE = False
    jieba = None
    pseg = None


def cjk_tokenize(text: str) -> str:
    """
    Pre-tokenize text for FTS5 with jieba Traditional Chinese segmentation.
    
    CJK characters are split into tokens; Latin letters, digits, and common
    engineering symbols are preserved as single tokens. This ensures BM25
    works correctly for mixed Chinese/English technical text.
    
    Applied identically at index time and query time.
    """
    if not text:
        return ""
    
    if not JIEBA_AVAILABLE:
        # Fallback: basic whitespace/punctuation split (less accurate for CJK)
        return text
    
    # Use jieba with Traditional Chinese dictionary if available
    # jieba automatically uses dict.txt.big if set
    tokens = []
    for word in jieba.cut(text, cut_all=False):
        # Preserve Latin/digit sequences and common engineering symbols as single tokens
        # Check if the word is mostly Latin/digits/symbols
        if re.match(r'^[\w\d\.\-\+\(\)\/\%\=\<\>\:\,\;\_\@\#\$\&\*]+$', word):
            tokens.append(word)
        else:
            # For CJK text, jieba already split appropriately
            tokens.append(word)
    
    return " ".join(tokens)


SCHEMA_SQL = """
-- Core tables
CREATE TABLE IF NOT EXISTS documents (
    document_id TEXT PRIMARY KEY,
    document_family_id TEXT NOT NULL,
    edition_id TEXT NOT NULL,
    title TEXT NOT NULL,
    publisher TEXT,
    jurisdiction TEXT,
    doc_type TEXT NOT NULL,
    language TEXT,
    publication_date TEXT,
    revision_date TEXT,
    source_url TEXT,
    retrieved_at TEXT,
    original_sha256 TEXT NOT NULL UNIQUE,
    total_pdf_pages INTEGER NOT NULL,
    parser_name TEXT NOT NULL,
    parser_version TEXT NOT NULL,
    parser_config TEXT NOT NULL,  -- JSON
    parse_run_id TEXT NOT NULL,
    completeness REAL NOT NULL DEFAULT 0.0,
    qc_status TEXT NOT NULL DEFAULT 'pending',
    amendment_relations TEXT NOT NULL,  -- JSON array
    version_relations TEXT NOT NULL,    -- JSON array
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS items (
    item_id TEXT PRIMARY KEY,
    document_id TEXT NOT NULL,
    edition_id TEXT NOT NULL,
    type TEXT NOT NULL,
    parent_id TEXT,
    section_path TEXT NOT NULL,  -- JSON array
    number TEXT,
    text TEXT NOT NULL,
    latex TEXT,
    table_html TEXT,
    pdf_page INTEGER NOT NULL,
    printed_page TEXT,
    bbox TEXT,  -- JSON array
    asset_ids TEXT NOT NULL,  -- JSON array
    cross_refs TEXT NOT NULL,  -- JSON array
    footnote_ids TEXT NOT NULL,  -- JSON array
    amendment_links TEXT NOT NULL,  -- JSON array
    content_origin TEXT NOT NULL,
    derived_info TEXT,  -- JSON object
    status TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    summary_en TEXT,
    summary_tc TEXT,
    generated_questions TEXT,  -- JSON array
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at TEXT NOT NULL DEFAULT (datetime('now')),
    FOREIGN KEY (document_id) REFERENCES documents(document_id)
);

CREATE TABLE IF NOT EXISTS pages (
    document_id TEXT NOT NULL,
    pdf_page INTEGER NOT NULL,
    status TEXT NOT NULL,
    parser TEXT NOT NULL,
    detail TEXT,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at TEXT NOT NULL DEFAULT (datetime('now')),
    PRIMARY KEY (document_id, pdf_page),
    FOREIGN KEY (document_id) REFERENCES documents(document_id)
);

CREATE TABLE IF NOT EXISTS assets (
    asset_id TEXT PRIMARY KEY,
    document_id TEXT NOT NULL,
    item_id TEXT,
    kind TEXT NOT NULL,  -- figure, page_render, etc.
    mime_type TEXT NOT NULL,
    path TEXT NOT NULL,
    width INTEGER,
    height INTEGER,
    description TEXT,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    FOREIGN KEY (document_id) REFERENCES documents(document_id),
    FOREIGN KEY (item_id) REFERENCES items(item_id)
);

-- Parse runs table for tracking ingest runs
CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY,
    document_id TEXT NOT NULL,
    parser TEXT NOT NULL,
    parser_version TEXT NOT NULL,
    config_json TEXT NOT NULL,  -- JSON
    started_at TEXT NOT NULL,
    finished_at TEXT,
    pages_done TEXT NOT NULL,  -- JSON array of 1-based page numbers
    pages_failed TEXT NOT NULL,  -- JSON array of 1-based page numbers
    items_created INTEGER NOT NULL DEFAULT 0,
    status TEXT NOT NULL DEFAULT 'running',  -- running, completed, failed
    error_detail TEXT,
    FOREIGN KEY (document_id) REFERENCES documents(document_id)
);

-- FTS5 virtual table for full-text search
-- Uses pre-tokenized text (CJK split by jieba, Latin/digits preserved)
-- tokenize = 'unicode61' with pre-tokenized input (space-separated tokens)
CREATE VIRTUAL TABLE IF NOT EXISTS items_fts USING fts5(
    item_id UNINDEXED,
    document_id UNINDEXED,
    edition_id UNINDEXED,
    type UNINDEXED,
    number,
    section_path,
    text,
    tokenize = 'unicode61'
);

-- FTS5 is kept in sync by application code (add_items, delete_item), not triggers
-- This allows CJK pre-tokenization at index time

-- Indexes for common lookups
CREATE INDEX IF NOT EXISTS idx_items_document_edition ON items(document_id, edition_id);
CREATE INDEX IF NOT EXISTS idx_items_type_number ON items(document_id, edition_id, type, number);
CREATE INDEX IF NOT EXISTS idx_pages_document ON pages(document_id);
CREATE INDEX IF NOT EXISTS idx_runs_document ON runs(document_id);
"""


class Store:
    """SQLite-backed canonical store with FTS5 search."""

    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn: Optional[sqlite3.Connection] = None
        self.init_db()

    @contextmanager
    def _transaction(self):
        """Context manager for database transactions."""
        conn = self._get_conn()
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise

    def _get_conn(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(self.db_path)
            self._conn.row_factory = sqlite3.Row
            # Enable foreign keys
            self._conn.execute("PRAGMA foreign_keys = ON")
        return self._conn

    def close(self) -> None:
        """Close the database connection."""
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def init_db(self) -> None:
        """Initialize database schema."""
        conn = self._get_conn()
        conn.executescript(SCHEMA_SQL)
        self._migrate_schema(conn)
        conn.commit()

    def _migrate_schema(self, conn: sqlite3.Connection) -> None:
        """Run idempotent migrations for schema additions."""
        # Migration: Add summary_en, summary_tc, generated_questions to items table (Brief 04)
        self._ensure_columns_exist(
            conn,
            "items",
            {
                "summary_en": "TEXT",
                "summary_tc": "TEXT",
                "generated_questions": "TEXT",  # JSON array
            },
        )

    def _ensure_columns_exist(
        self, conn: sqlite3.Connection, table: str, columns: dict[str, str]
    ) -> None:
        """Add missing columns to a table idempotently."""
        # Get existing columns
        cursor = conn.execute(f"PRAGMA table_info({table})")
        existing = {row["name"] for row in cursor.fetchall()}

        for col_name, col_type in columns.items():
            if col_name not in existing:
                conn.execute(
                    f"ALTER TABLE {table} ADD COLUMN {col_name} {col_type}"
                )

    # --- Document operations ---

    def upsert_document(self, doc: DocumentMeta) -> None:
        """Insert or update a document (by document_id)."""
        with self._transaction() as conn:
            conn.execute(
                """
                INSERT INTO documents (
                    document_id, document_family_id, edition_id, title, publisher,
                    jurisdiction, doc_type, language, publication_date, revision_date,
                    source_url, retrieved_at, original_sha256, total_pdf_pages,
                    parser_name, parser_version, parser_config, parse_run_id,
                    completeness, qc_status, amendment_relations, version_relations,
                    updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, datetime('now'))
                ON CONFLICT(document_id) DO UPDATE SET
                    document_family_id = excluded.document_family_id,
                    edition_id = excluded.edition_id,
                    title = excluded.title,
                    publisher = excluded.publisher,
                    jurisdiction = excluded.jurisdiction,
                    doc_type = excluded.doc_type,
                    language = excluded.language,
                    publication_date = excluded.publication_date,
                    revision_date = excluded.revision_date,
                    source_url = excluded.source_url,
                    retrieved_at = excluded.retrieved_at,
                    original_sha256 = excluded.original_sha256,
                    total_pdf_pages = excluded.total_pdf_pages,
                    parser_name = excluded.parser_name,
                    parser_version = excluded.parser_version,
                    parser_config = excluded.parser_config,
                    parse_run_id = excluded.parse_run_id,
                    completeness = excluded.completeness,
                    qc_status = excluded.qc_status,
                    amendment_relations = excluded.amendment_relations,
                    version_relations = excluded.version_relations,
                    updated_at = datetime('now')
                """,
                (
                    doc.document_id,
                    doc.document_family_id,
                    doc.edition_id,
                    doc.title,
                    doc.publisher,
                    doc.jurisdiction,
                    doc.doc_type.value,
                    doc.language,
                    doc.publication_date.isoformat() if doc.publication_date else None,
                    doc.revision_date.isoformat() if doc.revision_date else None,
                    doc.source_url,
                    doc.retrieved_at.isoformat() if doc.retrieved_at else None,
                    doc.original_sha256,
                    doc.total_pdf_pages,
                    doc.parser_name,
                    doc.parser_version,
                    json.dumps(doc.parser_config),
                    doc.parse_run_id,
                    doc.completeness,
                    doc.qc_status,
                    json.dumps(doc.amendment_relations),
                    json.dumps(doc.version_relations),
                ),
            )

    def get_document(self, document_id: str) -> Optional[DocumentMeta]:
        """Retrieve a document by ID."""
        conn = self._get_conn()
        row = conn.execute(
            "SELECT * FROM documents WHERE document_id = ?", (document_id,)
        ).fetchone()
        if row is None:
            return None
        return self._row_to_document(row)

    def list_documents(self) -> list[DocumentMeta]:
        """List all documents."""
        conn = self._get_conn()
        rows = conn.execute("SELECT * FROM documents ORDER BY created_at").fetchall()
        return [self._row_to_document(row) for row in rows]

    def get_document_by_sha256(self, sha256: str) -> Optional[DocumentMeta]:
        """Retrieve a document by its original SHA-256 hash."""
        conn = self._get_conn()
        row = conn.execute(
            "SELECT * FROM documents WHERE original_sha256 = ?", (sha256,)
        ).fetchone()
        if row is None:
            return None
        return self._row_to_document(row)

    def _row_to_document(self, row: sqlite3.Row) -> DocumentMeta:
        from .schema import DocType
        return DocumentMeta(
            document_id=row["document_id"],
            document_family_id=row["document_family_id"],
            edition_id=row["edition_id"],
            title=row["title"],
            publisher=row["publisher"],
            jurisdiction=row["jurisdiction"],
            doc_type=DocType(row["doc_type"]),
            language=row["language"],
            publication_date=date.fromisoformat(row["publication_date"]) if row["publication_date"] else None,
            revision_date=date.fromisoformat(row["revision_date"]) if row["revision_date"] else None,
            source_url=row["source_url"],
            retrieved_at=date.fromisoformat(row["retrieved_at"]) if row["retrieved_at"] else None,
            original_sha256=row["original_sha256"],
            total_pdf_pages=row["total_pdf_pages"],
            parser_name=row["parser_name"],
            parser_version=row["parser_version"],
            parser_config=json.loads(row["parser_config"]) if row["parser_config"] else {},
            parse_run_id=row["parse_run_id"],
            completeness=row["completeness"],
            qc_status=row["qc_status"],
            amendment_relations=json.loads(row["amendment_relations"]) if row["amendment_relations"] else [],
            version_relations=json.loads(row["version_relations"]) if row["version_relations"] else [],
        )

    # --- Item operations ---

    def add_items(self, items: list[ContentItem]) -> None:
        """Bulk insert or update items (transactional)."""
        if not items:
            return
        with self._transaction() as conn:
            for item in items:
                ch = content_hash(item.text)
                # DEBUG
                print(f"DEBUG add_items: item_id={item.item_id}, summary_en={item.summary_en}")
                print(f"DEBUG: item tuple len = {len((item.item_id, item.document_id, item.edition_id, item.type.value, item.parent_id, json.dumps(item.section_path), item.number, item.text, item.latex, item.table_html, item.pdf_page, item.printed_page, json.dumps(item.bbox) if item.bbox else None, json.dumps(item.asset_ids), json.dumps(item.cross_refs), json.dumps(item.footnote_ids), json.dumps(item.amendment_links), item.content_origin.value, json.dumps(item.derived_info) if item.derived_info else None, item.status.value, ch, item.summary_en, item.summary_tc, json.dumps(item.generated_questions) if item.generated_questions else None))}")
                conn.execute(
                    """
                    INSERT INTO items (
                        item_id, document_id, edition_id, type, parent_id, section_path,
                        number, text, latex, table_html, pdf_page, printed_page, bbox,
                        asset_ids, cross_refs, footnote_ids, amendment_links,
                        content_origin, derived_info, status, content_hash,
                        summary_en, summary_tc, generated_questions, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, datetime('now'))
                    ON CONFLICT(item_id) DO UPDATE SET
                        document_id = excluded.document_id,
                        edition_id = excluded.edition_id,
                        type = excluded.type,
                        parent_id = excluded.parent_id,
                        section_path = excluded.section_path,
                        number = excluded.number,
                        text = excluded.text,
                        latex = excluded.latex,
                        table_html = excluded.table_html,
                        pdf_page = excluded.pdf_page,
                        printed_page = excluded.printed_page,
                        bbox = excluded.bbox,
                        asset_ids = excluded.asset_ids,
                        cross_refs = excluded.cross_refs,
                        footnote_ids = excluded.footnote_ids,
                        amendment_links = excluded.amendment_links,
                        content_origin = excluded.content_origin,
                        derived_info = excluded.derived_info,
                        status = excluded.status,
                        content_hash = excluded.content_hash,
                        summary_en = excluded.summary_en,
                        summary_tc = excluded.summary_tc,
                        generated_questions = excluded.generated_questions,
                        updated_at = datetime('now')
                    """,
                    (
                        item.item_id,
                        item.document_id,
                        item.edition_id,
                        item.type.value,
                        item.parent_id,
                        json.dumps(item.section_path),
                        item.number,
                        item.text,
                        item.latex,
                        item.table_html,
                        item.pdf_page,
                        item.printed_page,
                        json.dumps(item.bbox) if item.bbox else None,
                        json.dumps(item.asset_ids),
                        json.dumps(item.cross_refs),
                        json.dumps(item.footnote_ids),
                        json.dumps(item.amendment_links),
                        item.content_origin.value,
                        json.dumps(item.derived_info) if item.derived_info else None,
                        item.status.value,
                        ch,
                        item.summary_en,
                        item.summary_tc,
                        json.dumps(item.generated_questions) if item.generated_questions else None,
                    ),
                )
                
                # Insert/update FTS5 with pre-tokenized text
                # FTS5 virtual tables don't support ON CONFLICT, so delete first then insert
                tokenized_text = cjk_tokenize(item.text)
                conn.execute(
                    "DELETE FROM items_fts WHERE item_id = ?",
                    (item.item_id,),
                )
                conn.execute(
                    """
                    INSERT INTO items_fts (item_id, document_id, edition_id, type, number, section_path, text)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        item.item_id,
                        item.document_id,
                        item.edition_id,
                        item.type.value,
                        item.number,
                        json.dumps(item.section_path),
                        tokenized_text,
                    ),
                )

    def get_item(self, item_id: str) -> Optional[ContentItem]:
        """Retrieve a single item by ID."""
        conn = self._get_conn()
        row = conn.execute("SELECT * FROM items WHERE item_id = ?", (item_id,)).fetchone()
        if row is None:
            return None
        return self._row_to_item(row)

    def items_by_number(
        self, document_id: str, edition_id: str, item_type: str, number: str
    ) -> list[ContentItem]:
        """Exact lookup by type + number within a document/edition."""
        conn = self._get_conn()
        rows = conn.execute(
            """
            SELECT * FROM items
            WHERE document_id = ? AND edition_id = ? AND type = ? AND number = ?
            ORDER BY pdf_page, item_id
            """,
            (document_id, edition_id, item_type, number),
        ).fetchall()
        return [self._row_to_item(row) for row in rows]

    def get_items_by_section_path(
        self, document_id: str, section_path: list[str]
    ) -> list[ContentItem]:
        """Get all items in a document whose section_path starts with the given path."""
        if not section_path:
            return []
        path_json = json.dumps(section_path)
        # Match exact path or path as prefix (e.g., ["1"] matches ["1"] and ["1", "2"])
        conn = self._get_conn()
        rows = conn.execute(
            """
            SELECT * FROM items
            WHERE document_id = ? AND (section_path = ? OR section_path LIKE ?)
            ORDER BY pdf_page, item_id
            """,
            (document_id, path_json, path_json[:-1] + ",%"),
        ).fetchall()
        return [self._row_to_item(row) for row in rows]

    def get_items_by_page(
        self, document_id: str, pdf_page: int
    ) -> list[ContentItem]:
        """Get all items on a specific PDF page in a document."""
        conn = self._get_conn()
        rows = conn.execute(
            """
            SELECT * FROM items
            WHERE document_id = ? AND pdf_page = ?
            ORDER BY item_id
            """,
            (document_id, pdf_page),
        ).fetchall()
        return [self._row_to_item(row) for row in rows]

    def get_asset(self, asset_id: str):
        """Get asset by ID."""
        from .schema import AssetMeta
        conn = self._get_conn()
        row = conn.execute(
            "SELECT * FROM assets WHERE asset_id = ?", (asset_id,)
        ).fetchone()
        if row is None:
            return None
        sha256_val = row["sha256"] if "sha256" in row.keys() else ""
        return AssetMeta(
            asset_id=row["asset_id"],
            document_id=row["document_id"],
            item_id=row["item_id"],
            kind=row["kind"],
            file_path=row["path"],
            mime_type=row["mime_type"],
            width=row["width"],
            height=row["height"],
            sha256=sha256_val,
            description=row["description"],
        )

    def upsert_asset(self, asset: "AssetMeta") -> None:
        """Insert or update an asset."""
        from .schema import AssetMeta
        with self._transaction() as conn:
            conn.execute(
                """
                INSERT INTO assets (
                    asset_id, document_id, item_id, kind, mime_type, path,
                    width, height, description, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, datetime('now'))
                ON CONFLICT(asset_id) DO UPDATE SET
                    document_id = excluded.document_id,
                    item_id = excluded.item_id,
                    kind = excluded.kind,
                    mime_type = excluded.mime_type,
                    path = excluded.path,
                    width = excluded.width,
                    height = excluded.height,
                    description = excluded.description
                """,
                (
                    asset.asset_id,
                    asset.document_id,
                    asset.item_id,
                    asset.kind,
                    asset.mime_type,
                    asset.file_path,
                    asset.width,
                    asset.height,
                    asset.description,
                ),
            )

    def _row_to_item(self, row: sqlite3.Row) -> ContentItem:
        from .schema import ItemType, ContentOrigin, ItemStatus
        return ContentItem(
            item_id=row["item_id"],
            document_id=row["document_id"],
            edition_id=row["edition_id"],
            type=ItemType(row["type"]),
            parent_id=row["parent_id"],
            section_path=json.loads(row["section_path"]) if row["section_path"] else [],
            number=row["number"],
            text=row["text"],
            latex=row["latex"],
            table_html=row["table_html"],
            pdf_page=row["pdf_page"],
            printed_page=row["printed_page"],
            bbox=json.loads(row["bbox"]) if row["bbox"] else None,
            asset_ids=json.loads(row["asset_ids"]) if row["asset_ids"] else [],
            cross_refs=json.loads(row["cross_refs"]) if row["cross_refs"] else [],
            footnote_ids=json.loads(row["footnote_ids"]) if row["footnote_ids"] else [],
            amendment_links=json.loads(row["amendment_links"]) if row["amendment_links"] else [],
            content_origin=ContentOrigin(row["content_origin"]),
            derived_info=json.loads(row["derived_info"]) if row["derived_info"] else None,
            status=ItemStatus(row["status"]),
            summary_en=row["summary_en"] if "summary_en" in row.keys() else None,
            summary_tc=row["summary_tc"] if "summary_tc" in row.keys() else None,
            generated_questions=json.loads(row["generated_questions"]) if row["generated_questions"] else None,
        )

    # --- Page status operations ---

    def add_page_status(self, page: PageStatus) -> None:
        """Insert or update a page status record."""
        with self._transaction() as conn:
            conn.execute(
                """
                INSERT INTO pages (document_id, pdf_page, status, parser, detail, updated_at)
                VALUES (?, ?, ?, ?, ?, datetime('now'))
                ON CONFLICT(document_id, pdf_page) DO UPDATE SET
                    status = excluded.status,
                    parser = excluded.parser,
                    detail = excluded.detail,
                    updated_at = datetime('now')
                """,
                (
                    page.document_id,
                    page.pdf_page,
                    page.status.value,
                    page.parser,
                    page.detail,
                ),
            )

    def get_page_statuses(self, document_id: str) -> list[PageStatus]:
        """Get all page statuses for a document."""
        from .schema import PageStatusEnum
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT * FROM pages WHERE document_id = ? ORDER BY pdf_page",
            (document_id,),
        ).fetchall()
        return [
            PageStatus(
                document_id=row["document_id"],
                pdf_page=row["pdf_page"],
                status=PageStatusEnum(row["status"]),
                parser=row["parser"],
                detail=row["detail"],
            )
            for row in rows
        ]

    def get_done_pages(self, document_id: str) -> set[int]:
        """Get set of page numbers that have been successfully parsed."""
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT pdf_page FROM pages WHERE document_id = ? AND status = 'done'",
            (document_id,),
        ).fetchall()
        return {row["pdf_page"] for row in rows}

    def get_failed_pages(self, document_id: str) -> set[int]:
        """Get set of page numbers that failed parsing."""
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT pdf_page FROM pages WHERE document_id = ? AND status = 'failed'",
            (document_id,),
        ).fetchall()
        return {row["pdf_page"] for row in rows}

    # --- Run operations ---

    def create_run(
        self,
        run_id: str,
        document_id: str,
        parser: str,
        parser_version: str,
        config: dict,
    ) -> None:
        """Create a new parse run record."""
        with self._transaction() as conn:
            conn.execute(
                """
                INSERT INTO runs (
                    run_id, document_id, parser, parser_version, config_json,
                    started_at, finished_at, pages_done, pages_failed,
                    items_created, status
                ) VALUES (?, ?, ?, ?, ?, datetime('now'), NULL, '[]', '[]', 0, 'running')
                """,
                (
                    run_id,
                    document_id,
                    parser,
                    parser_version,
                    json.dumps(config),
                ),
            )

    def update_run(
        self,
        run_id: str,
        pages_done: list[int] = None,
        pages_failed: list[int] = None,
        items_created: int = None,
        status: str = None,
        error_detail: str = None,
    ) -> None:
        """Update a parse run record."""
        with self._transaction() as conn:
            # Build dynamic update
            updates = []
            params = []

            if pages_done is not None:
                updates.append("pages_done = ?")
                params.append(json.dumps(pages_done))
            if pages_failed is not None:
                updates.append("pages_failed = ?")
                params.append(json.dumps(pages_failed))
            if items_created is not None:
                updates.append("items_created = ?")
                params.append(items_created)
            if status is not None:
                updates.append("status = ?")
                params.append(status)
                if status in ("completed", "failed"):
                    updates.append("finished_at = datetime('now')")
            if error_detail is not None:
                updates.append("error_detail = ?")
                params.append(error_detail)

            if not updates:
                return

            params.append(run_id)
            conn.execute(
                f"UPDATE runs SET {', '.join(updates)} WHERE run_id = ?",
                params,
            )

    def get_run(self, run_id: str) -> Optional[dict]:
        """Get a parse run by ID."""
        conn = self._get_conn()
        row = conn.execute(
            "SELECT * FROM runs WHERE run_id = ?", (run_id,)
        ).fetchone()
        if row is None:
            return None
        return dict(row)

    def get_runs_for_document(self, document_id: str) -> list[dict]:
        """Get all parse runs for a document."""
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT * FROM runs WHERE document_id = ? ORDER BY started_at DESC",
            (document_id,),
        ).fetchall()
        return [dict(row) for row in rows]

    # --- FTS search ---

    def fts_search(
        self, query: str, document_id: Optional[str] = None, edition_id: Optional[str] = None, limit: int = 20
    ) -> list[SearchResult]:
        """
        Full-text search using FTS5 BM25 ranking.

        Args:
            query: Search query (FTS5 syntax supported)
            document_id: Optional filter by document
            edition_id: Optional filter by edition
            limit: Maximum results to return

        Returns:
            List of SearchResult ordered by BM25 rank (best first).
        """
        conn = self._get_conn()

        # Pre-tokenize query for CJK (same as index time)
        tokenized_query = cjk_tokenize(query)
        # Wrap in double quotes for FTS5 phrase search to handle punctuation
        fts_query = f'"{tokenized_query}"' if not (tokenized_query.startswith('"') and tokenized_query.endswith('"')) else tokenized_query
        where_clauses = ["items_fts MATCH ?"]
        params = [fts_query]

        if document_id:
            where_clauses.append("i.document_id = ?")
            params.append(document_id)
        if edition_id:
            where_clauses.append("i.edition_id = ?")
            params.append(edition_id)

        where_sql = " AND ".join(where_clauses)
        params.append(limit)

        sql = f"""
            SELECT
                i.item_id, i.document_id, i.edition_id, i.type, i.number,
                i.section_path, i.text, i.pdf_page, i.printed_page,
                i.content_origin, i.status,
                d.title as doc_title,
                bm25(items_fts) as rank
            FROM items_fts
            JOIN items i ON i.item_id = items_fts.item_id
            JOIN documents d ON d.document_id = i.document_id
            WHERE {where_sql}
            ORDER BY rank
            LIMIT ?
        """

        rows = conn.execute(sql, params).fetchall()
        results = []
        for row in rows:
            # Create snippet (first 200 chars of text)
            snippet = row["text"][:200]
            if len(row["text"]) > 200:
                snippet += "…"

            results.append(
                SearchResult(
                    item_id=row["item_id"],
                    document_id=row["document_id"],
                    edition_id=row["edition_id"],
                    doc_title=row["doc_title"],
                    number=row["number"],
                    section_path=json.loads(row["section_path"]) if row["section_path"] else [],
                    snippet=snippet,
                    pdf_page=row["pdf_page"],
                    printed_page=row["printed_page"],
                    content_origin=row["content_origin"],
                    status=row["status"],
                    score=-row["rank"],  # BM25 lower is better; negate for higher=better
                    has_more_context=len(row["text"]) > 200,
                    related=[],
                )
            )
        return results

    # --- Export ---

    def export_jsonl(self, output_path: Path) -> int:
        """
        Export all items and documents to JSONL file.

        Each line is a JSON object: either an item or a document.
        Document lines have a "_kind": "document" field.
        Item lines have a "_kind": "item" field.

        Returns:
            Number of lines written.
        """
        conn = self._get_conn()
        count = 0

        with open(output_path, "w", encoding="utf-8") as f:
            # Export documents
            doc_rows = conn.execute("SELECT * FROM documents ORDER BY document_id").fetchall()
            for row in doc_rows:
                doc = self._row_to_document(row)
                line = doc.model_dump(mode="json")
                line["_kind"] = "document"
                f.write(json.dumps(line, ensure_ascii=False) + "\n")
                count += 1

            # Export items
            item_rows = conn.execute("SELECT * FROM items ORDER BY document_id, pdf_page, item_id").fetchall()
            for row in item_rows:
                item = self._row_to_item(row)
                line = item.model_dump(mode="json")
                line["_kind"] = "item"
                f.write(json.dumps(line, ensure_ascii=False) + "\n")
                count += 1

        return count


# Need to import date for _row_to_document
from datetime import date