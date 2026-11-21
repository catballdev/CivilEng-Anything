"""
Vector index stored in SQLite for CivilEng-Anything.

Vectors are stored per-embedder namespace to prevent mixing different models.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import struct
from pathlib import Path
from typing import Any, Optional

import numpy as np

from .store import Store
from .embeddings import Embedder

logger = logging.getLogger(__name__)

VECTOR_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS vectors (
    item_id TEXT NOT NULL,
    embedder_name TEXT NOT NULL,
    dim INTEGER NOT NULL,
    vector BLOB NOT NULL,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    PRIMARY KEY (item_id, embedder_name)
);

CREATE INDEX IF NOT EXISTS idx_vectors_embedder ON vectors(embedder_name);
"""

INDEX_META_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS index_meta (
    embedder_name TEXT PRIMARY KEY,
    built_at TEXT NOT NULL,
    item_count INTEGER NOT NULL,
    dim INTEGER NOT NULL
);
"""


class VectorIndex:
    """Vector index backed by SQLite with numpy cosine similarity search."""

    def __init__(self, db_path: Path):
        self.db_path = db_path
        self._init_tables()

    def _init_tables(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.executescript(VECTOR_TABLE_SQL)
            conn.executescript(INDEX_META_TABLE_SQL)

    def _get_conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def has_vectors(self, embedder_name: str) -> bool:
        """Check if any vectors exist for this embedder."""
        with self._get_conn() as conn:
            row = conn.execute(
                "SELECT COUNT(*) as cnt FROM vectors WHERE embedder_name = ?",
                (embedder_name,)
            ).fetchone()
            return row["cnt"] > 0

    def get_index_meta(self, embedder_name: str) -> Optional[dict]:
        """Get index metadata for an embedder."""
        with self._get_conn() as conn:
            row = conn.execute(
                "SELECT * FROM index_meta WHERE embedder_name = ?",
                (embedder_name,)
            ).fetchone()
            return dict(row) if row else None

    def build_vectors(
        self,
        store: Store,
        embedder: Embedder,
        batch_size: int = 32,
        document_id: Optional[str] = None,
    ) -> dict[str, Any]:
        """
        Build vectors for items missing embeddings under the current embedder.

        Args:
            store: Store instance
            embedder: Embedder to use
            batch_size: Batch size for embedding
            document_id: Optional filter to only build for one document

        Returns:
            Dict with stats: built_count, skipped_count, total_items
        """
        embedder_name = embedder.name
        dim = embedder.dim

        # Get items that don't have vectors for this embedder
        with self._get_conn() as conn:
            if document_id:
                rows = conn.execute("""
                    SELECT i.item_id, i.text, i.number, i.section_path
                    FROM items i
                    LEFT JOIN vectors v ON i.item_id = v.item_id AND v.embedder_name = ?
                    WHERE i.document_id = ? AND v.item_id IS NULL
                    ORDER BY i.item_id
                """, (embedder_name, document_id)).fetchall()
            else:
                rows = conn.execute("""
                    SELECT i.item_id, i.text, i.number, i.section_path
                    FROM items i
                    LEFT JOIN vectors v ON i.item_id = v.item_id AND v.embedder_name = ?
                    WHERE v.item_id IS NULL
                    ORDER BY i.item_id
                """, (embedder_name,)).fetchall()

        items_to_embed = [dict(r) for r in rows]
        total_items = len(items_to_embed)

        if total_items == 0:
            logger.info("No items to embed for %s", embedder_name)
            return {"built_count": 0, "skipped_count": 0, "total_items": 0}

        logger.info("Building vectors for %d items with %s", total_items, embedder_name)

        built_count = 0
        for i in range(0, total_items, batch_size):
            batch = items_to_embed[i:i + batch_size]

            # Prepare texts: number + section_path + text
            texts = []
            for item in batch:
                parts = []
                if item["number"]:
                    parts.append(item["number"])
                if item["section_path"]:
                    try:
                        import json
                        section_path = json.loads(item["section_path"])
                        parts.extend(section_path)
                    except Exception:
                        pass
                parts.append(item["text"])
                texts.append(" ".join(parts))

            # Embed batch
            vectors = embedder.embed(texts)

            # Store vectors
            with self._get_conn() as conn:
                for item, vec in zip(batch, vectors):
                    vec_blob = self._pack_vector(vec)
                    conn.execute("""
                        INSERT INTO vectors (item_id, embedder_name, dim, vector)
                        VALUES (?, ?, ?, ?)
                    """, (item["item_id"], embedder_name, dim, vec_blob))
                conn.commit()

            built_count += len(batch)
            logger.debug("Embedded batch %d/%d", i // batch_size + 1, (total_items + batch_size - 1) // batch_size)

        # Update metadata
        self._update_meta(embedder_name, built_count, dim)

        logger.info("Built %d vectors for %s", built_count, embedder_name)
        return {
            "built_count": built_count,
            "skipped_count": 0,
            "total_items": total_items,
        }

    def _update_meta(self, embedder_name: str, item_count: int, dim: int):
        """Update index metadata."""
        with self._get_conn() as conn:
            conn.execute("""
                INSERT INTO index_meta (embedder_name, built_at, item_count, dim)
                VALUES (?, datetime('now'), ?, ?)
                ON CONFLICT(embedder_name) DO UPDATE SET
                    built_at = datetime('now'),
                    item_count = item_count,
                    dim = dim
            """, (embedder_name, item_count, dim))
            conn.commit()

    def _pack_vector(self, vec: list[float]) -> bytes:
        """Pack float32 vector to bytes."""
        return struct.pack(f"{len(vec)}f", *vec)

    def _unpack_vector(self, blob: bytes, dim: int) -> np.ndarray:
        """Unpack bytes to float32 numpy array."""
        return np.frombuffer(blob, dtype=np.float32, count=dim)

    def vector_search(
        self,
        store: Store,
        embedder: Embedder,
        query: str,
        top_k: int = 8,
        document_id: Optional[str] = None,
        edition_id: Optional[str] = None,
        jurisdiction: Optional[str] = None,
        doc_type: Optional[str] = None,
    ) -> list[dict]:
        """
        Cosine similarity search over stored vectors.

        Returns list of dicts with item_id and score, ordered by score desc.
        """
        embedder_name = embedder.name

        if not self.has_vectors(embedder_name):
            return []

        # Embed query
        query_vec = embedder.embed([query])[0]
        query_vec_np = np.array(query_vec, dtype=np.float32)

        # Fetch all vectors for this embedder with optional filters
        with self._get_conn() as conn:
            # Build filter query
            where_clauses = ["v.embedder_name = ?"]
            params = [embedder_name]

            if document_id:
                where_clauses.append("i.document_id = ?")
                params.append(document_id)
            if edition_id:
                where_clauses.append("i.edition_id = ?")
                params.append(edition_id)

            where_sql = " AND ".join(where_clauses)

            # For jurisdiction/doc_type, need to join documents table
            join_docs = jurisdiction or doc_type
            if join_docs:
                from_clause = "FROM vectors v JOIN items i ON v.item_id = i.item_id JOIN documents d ON i.document_id = d.document_id"
                if jurisdiction:
                    where_clauses.append("d.jurisdiction = ?")
                    params.append(jurisdiction)
                if doc_type:
                    where_clauses.append("d.doc_type = ?")
                    params.append(doc_type)
            else:
                from_clause = "FROM vectors v JOIN items i ON v.item_id = i.item_id"

            rows = conn.execute(f"""
                SELECT v.item_id, v.vector, v.dim
                {from_clause}
                WHERE {where_sql}
            """, params).fetchall()

        if not rows:
            return []

        # Compute cosine similarity
        results = []
        for row in rows:
            item_id = row["item_id"]
            vec_blob = row["vector"]
            dim = row["dim"]

            if dim != len(query_vec_np):
                continue  # Skip dimension mismatch

            vec_np = self._unpack_vector(vec_blob, dim)

            # Cosine similarity (vectors are normalized)
            score = float(np.dot(query_vec_np, vec_np))
            results.append({"item_id": item_id, "score": score})

        # Sort by score descending
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]

    def delete_vectors(self, embedder_name: str):
        """Delete all vectors for an embedder namespace."""
        with self._get_conn() as conn:
            conn.execute("DELETE FROM vectors WHERE embedder_name = ?", (embedder_name,))
            conn.execute("DELETE FROM index_meta WHERE embedder_name = ?", (embedder_name,))
            conn.commit()