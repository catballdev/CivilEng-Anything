"""
SQLite store for CivilEng-Anything canonical data.

Uses stdlib sqlite3 with FTS5 for full-text search.
"""

import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Optional

from .schema import DocumentMeta, ContentItem, PageStatus, SearchResult
from .ids import content_hash


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

-- FTS5 virtual table for full-text search
-- tokenizer unicode61 preserves engineering symbols (no stemming)
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

-- Triggers to keep FTS in sync
CREATE TRIGGER IF NOT EXISTS items_fts_insert AFTER INSERT ON items BEGIN
    INSERT INTO items_fts (item_id, document_id, edition_id, type, number, section_path, text)
    VALUES (new.item_id, new.document_id, new.edition_id, new.type, new.number, new.section_path, new.text);
END;

CREATE TRIGGER IF NOT EXISTS items_fts_update AFTER UPDATE ON items BEGIN
    UPDATE items_fts SET
        document_id = new.document_id,
        edition_id = new.edition_id,
        type = new.type,
        number = new.number,
        section_path = new.section_path,
        text = new.text
    WHERE item_id = new.item_id;
END;

CREATE TRIGGER IF NOT EXISTS items_fts_delete AFTER DELETE ON items BEGIN
    DELETE FROM items_fts WHERE item_id = old.item_id;
END;

-- Indexes for common lookups
CREATE INDEX IF NOT EXISTS idx_items_document_edition ON items(document_id, edition_id);
CREATE INDEX IF NOT EXISTS idx_items_type_number ON items(document_id, edition_id, type, number);
CREATE INDEX IF NOT EXISTS idx_pages_document ON pages(document_id);
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
        conn.commit()

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
                conn.execute(
                    """
                    INSERT INTO items (
                        item_id, document_id, edition_id, type, parent_id, section_path,
                        number, text, latex, table_html, pdf_page, printed_page, bbox,
                        asset_ids, cross_refs, footnote_ids, amendment_links,
                        content_origin, derived_info, status, content_hash, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, datetime('now'))
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

        # Build WHERE clause for filters (qualify columns to avoid ambiguity)
        # Wrap query in double quotes for FTS5 phrase search to handle punctuation
        fts_query = f'"{query}"' if not (query.startswith('"') and query.endswith('"')) else query
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