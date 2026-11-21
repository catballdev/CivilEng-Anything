"""
Canonical schema for CivilEng-Anything (civileng-canonical/1).

Pydantic models matching the ARCHITECTURE.md specification exactly.
Unknown values are None — never invented defaults that look like data.
"""

from datetime import date
from enum import Enum
from typing import Any, Optional
from pydantic import BaseModel, Field, ConfigDict


class DocType(str, Enum):
    """Document type enumeration."""
    GOVERNMENT_CODE = "government_code"
    AMENDMENT = "amendment"
    GUIDELINE = "guideline"
    TEXTBOOK = "textbook"
    OTHER = "other"


class ItemType(str, Enum):
    """Content item type enumeration."""
    SECTION = "section"
    CLAUSE = "clause"
    PARAGRAPH = "paragraph"
    EQUATION = "equation"
    TABLE = "table"
    FIGURE = "figure"
    FOOTNOTE = "footnote"
    CROSS_REFERENCE = "cross_reference"
    AMENDMENT = "amendment"


class PageStatusEnum(str, Enum):
    """Per-page parsing status."""
    DONE = "done"
    FAILED = "failed"
    RETRY_PENDING = "retry_pending"
    NEEDS_REVIEW = "needs_review"


class ContentOrigin(str, Enum):
    """Origin of content text."""
    ORIGINAL = "original"
    OCR = "ocr"
    DERIVED = "derived"


class ItemStatus(str, Enum):
    """Verification status of a content item."""
    PARSED = "parsed"
    CHECKED = "checked"
    PENDING_VERIFICATION = "pending_verification"
    FAILED = "failed"


class SearchMode(str, Enum):
    """Search mode for retrieval."""
    EXACT = "exact"
    KEYWORD = "keyword"
    SEMANTIC = "semantic"
    HYBRID = "hybrid"


class DocumentMeta(BaseModel):
    """Document metadata per ARCHITECTURE.md canonical format."""
    model_config = ConfigDict(extra="forbid")

    document_id: str
    document_family_id: str
    edition_id: str
    title: str
    publisher: Optional[str] = None
    jurisdiction: Optional[str] = None
    doc_type: DocType
    language: Optional[str] = None
    publication_date: Optional[date] = None
    revision_date: Optional[date] = None
    source_url: Optional[str] = None
    retrieved_at: Optional[date] = None
    original_sha256: str
    total_pdf_pages: int
    parser_name: str
    parser_version: str
    parser_config: dict = Field(default_factory=dict)
    parse_run_id: str
    completeness: float = 0.0
    qc_status: str = "pending"
    amendment_relations: list[dict] = Field(default_factory=list)
    version_relations: list[dict] = Field(default_factory=list)


class ContentItem(BaseModel):
    """Content item per ARCHITECTURE.md canonical format."""
    model_config = ConfigDict(extra="forbid")

    item_id: str
    document_id: str
    edition_id: str
    type: ItemType
    parent_id: Optional[str] = None
    section_path: list[str] = Field(default_factory=list)
    number: Optional[str] = None
    text: str
    latex: Optional[str] = None
    table_html: Optional[str] = None
    pdf_page: int  # 1-based
    printed_page: Optional[str] = None
    bbox: Optional[list[float]] = None
    asset_ids: list[str] = Field(default_factory=list)
    cross_refs: list[str] = Field(default_factory=list)
    footnote_ids: list[str] = Field(default_factory=list)
    amendment_links: list[dict] = Field(default_factory=list)
    content_origin: ContentOrigin = ContentOrigin.ORIGINAL
    derived_info: Optional[dict] = None
    status: ItemStatus = ItemStatus.PARSED


class PageStatus(BaseModel):
    """Per-page parsing status."""
    model_config = ConfigDict(extra="forbid")

    document_id: str
    pdf_page: int  # 1-based
    status: PageStatusEnum
    parser: str
    detail: Optional[str] = None


class SearchResult(BaseModel):
    """Unified search result per ARCHITECTURE.md retrieval spec."""
    model_config = ConfigDict(extra="forbid")

    item_id: str
    document_id: str
    edition_id: str
    doc_title: str
    number: Optional[str] = None
    section_path: list[str] = Field(default_factory=list)
    snippet: str
    pdf_page: int
    printed_page: Optional[str] = None
    content_origin: ContentOrigin
    status: ItemStatus
    score: float
    has_more_context: bool = False
    related: list[str] = Field(default_factory=list)
    match_type: Optional[str] = None

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return self.model_dump()


class AssetMeta(BaseModel):
    """Asset metadata per ARCHITECTURE.md canonical format."""
    model_config = ConfigDict(extra="forbid")

    asset_id: str
    document_id: str
    item_id: Optional[str] = None
    kind: str  # figure, page_render, etc.
    mime_type: str
    file_path: str
    width: Optional[int] = None
    height: Optional[int] = None
    sha256: str
    description: Optional[str] = None