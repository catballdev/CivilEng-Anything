"""
CivilEng-Anything — traceable knowledge base for civil/structural engineering documents.
"""

__version__ = "0.1.0"

# Canonical schema version — must match the schema defined in ARCHITECTURE.md
CANONICAL_SCHEMA = "civileng-canonical/1"

from .config import Settings, get_settings
from .schema import (
    DocumentMeta,
    ContentItem,
    PageStatus,
    SearchResult,
    DocType,
    ItemType,
    PageStatusEnum,
    ContentOrigin,
    ItemStatus,
)
from .ids import make_item_id, content_hash
from .store import Store
from .pdfinspect import inspect_pdf, PdfInfo
from .parsers import run_mineru, MineruResult, run_docling, DoclingResult, ParserNotAvailable
from .normalize import normalize_mineru, extract_cross_refs
from .ingest import Ingestor, IngestReport

__all__ = [
    "__version__",
    "CANONICAL_SCHEMA",
    "Settings",
    "get_settings",
    "DocumentMeta",
    "ContentItem",
    "PageStatus",
    "SearchResult",
    "DocType",
    "ItemType",
    "PageStatusEnum",
    "ContentOrigin",
    "ItemStatus",
    "make_item_id",
    "content_hash",
    "Store",
    "inspect_pdf",
    "PdfInfo",
    "run_mineru",
    "MineruResult",
    "run_docling",
    "DoclingResult",
    "ParserNotAvailable",
    "normalize_mineru",
    "extract_cross_refs",
    "Ingestor",
    "IngestReport",
]