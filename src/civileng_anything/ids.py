"""
ID generation and hashing utilities for CivilEng-Anything.
"""

import hashlib
import re


def make_item_id(document_id: str, pdf_page: int, seq: int) -> str:
    """
    Produce a stable, readable, version-safe item ID.

    Format: {document_id}-p{pdf_page:04d}-{seq:04d}

    Args:
        document_id: The document identifier (should be filesystem-safe)
        pdf_page: 1-based PDF page number
        seq: Sequence number within the page (1-based)

    Returns:
        A stable, readable item ID string.
    """
    # Sanitize document_id to be safe in the ID (replace non-alphanumeric with _)
    safe_doc_id = re.sub(r'[^a-zA-Z0-9_-]', '_', document_id)
    return f"{safe_doc_id}-p{pdf_page:04d}-{seq:04d}"


def content_hash(text: str) -> str:
    """
    Compute SHA-256 hash of text content.

    Args:
        text: The text content to hash

    Returns:
        Hexadecimal SHA-256 hash string (64 characters)
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()