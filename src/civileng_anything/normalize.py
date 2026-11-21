"""
Normalization of MinerU content_list to CivilEng-Anything canonical ContentItem.
"""

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .schema import ContentItem, ItemType, ContentOrigin, ItemStatus
from .ids import make_item_id, content_hash


# Regex for clause numbers like "9.2.1", "6.2.3.1", etc.
CLAUSE_NUMBER_RE = re.compile(r"^\d+(?:\.\d+){1,4}\b")

# Regex for cross-references
CROSS_REF_PATTERNS = [
    (re.compile(r"\bclause\s+(\d+(?:\.\d+){1,4})\b", re.IGNORECASE), "clause"),
    (re.compile(r"\bcl\.\s*(\d+(?:\.\d+){1,4})\b", re.IGNORECASE), "clause"),
    (re.compile(r"\bsection\s+(\d+(?:\.\d+){1,4})\b", re.IGNORECASE), "section"),
    (re.compile(r"\btable\s+(\d+(?:\.\d+){1,4})\b", re.IGNORECASE), "table"),
    (re.compile(r"\bfigure\s+(\d+(?:\.\d+){1,4})\b", re.IGNORECASE), "figure"),
    (re.compile(r"\bfig\.\s*(\d+(?:\.\d+){1,4})\b", re.IGNORECASE), "figure"),
    (re.compile(r"\bequation\s+(\d+(?:\.\d+){1,4})\b", re.IGNORECASE), "equation"),
    (re.compile(r"\beq\.\s*(\d+(?:\.\d+){1,4})\b", re.IGNORECASE), "equation"),
]


@dataclass
class NormalizeContext:
    """Context for normalization across pages."""
    document_id: str
    edition_id: str
    page_offset: int
    section_path_stack: list[str]
    seq_counter: int = 0


def normalize_mineru(
    content_list: list[dict[str, Any]],
    document_id: str,
    edition_id: str,
    page_offset: int = 0,
) -> list[ContentItem]:
    """
    Convert MinerU content_list entries to ContentItem list.

    Args:
        content_list: List of MinerU content_list JSON entries.
        document_id: Document identifier.
        edition_id: Edition identifier.
        page_offset: Offset to add to MinerU's 0-based page_idx (default 0).
                    Use this when processing page ranges that don't start at page 1.

    Returns:
        List of ContentItem in canonical format.
    """
    ctx = NormalizeContext(
        document_id=document_id,
        edition_id=edition_id,
        page_offset=page_offset,
        section_path_stack=[],
    )

    items = []

    for entry in content_list:
        item = _normalize_entry(entry, ctx)
        if item:
            items.append(item)

    return items


def _normalize_entry(entry: dict[str, Any], ctx: NormalizeContext) -> ContentItem | None:
    """Normalize a single MinerU content_list entry."""
    entry_type = entry.get("type", "").lower()

    # Page number: MinerU page_idx is 0-based
    page_idx = entry.get("page_idx", 0)
    pdf_page = page_idx + 1 + ctx.page_offset

    # Bounding box
    bbox = entry.get("bbox")
    if bbox and isinstance(bbox, list) and len(bbox) == 4:
        bbox = [float(x) for x in bbox]
    else:
        bbox = None

    # Increment global sequence counter
    ctx.seq_counter += 1
    seq = ctx.seq_counter
    item_id = make_item_id(ctx.document_id, pdf_page, seq)

    # Base fields (without cross_refs or asset_ids - each type handles its own)
    base_kwargs = {
        "item_id": item_id,
        "document_id": ctx.document_id,
        "edition_id": ctx.edition_id,
        "pdf_page": pdf_page,
        "printed_page": None,
        "bbox": bbox,
        "footnote_ids": [],
        "amendment_links": [],
        "content_origin": ContentOrigin.ORIGINAL,
        "derived_info": None,
        "status": ItemStatus.PARSED,
    }

    if entry_type in ("title", "heading"):
        return _normalize_heading(entry, ctx, base_kwargs)

    elif entry_type == "text":
        return _normalize_text(entry, ctx, base_kwargs)

    elif entry_type in ("equation", "formula"):
        return _normalize_equation(entry, ctx, base_kwargs)

    elif entry_type == "table":
        return _normalize_table(entry, ctx, base_kwargs)

    elif entry_type in ("image", "figure"):
        return _normalize_figure(entry, ctx, base_kwargs)

    # Unknown type - skip
    return None


def _normalize_heading(entry: dict, ctx: NormalizeContext, base: dict) -> ContentItem:
    """Normalize a heading/title to a section item."""
    text = entry.get("text", "").strip()
    if not text:
        text = "Untitled Section"

    # Determine heading level from MinerU's "level" field (1, 2, 3...)
    level = entry.get("level", 1)
    if not isinstance(level, int):
        level = 1

    # Maintain section path stack
    # Pop levels deeper than current
    while len(ctx.section_path_stack) >= level:
        ctx.section_path_stack.pop()

    # Add current heading
    ctx.section_path_stack.append(text)

    # Section number - try to extract from text
    number = None
    # First try start of text
    match = re.match(r"^(\d+(?:\.\d+)*)\s+", text)
    if not match:
        # Try to find a number pattern like "9" or "9.2" anywhere in text
        match = re.search(r"\b(\d+(?:\.\d+)*)\b", text)
    if match:
        number = match.group(1)

    item = ContentItem(
        type=ItemType.SECTION,
        parent_id=None,
        section_path=ctx.section_path_stack.copy(),
        number=number,
        text=text,
        latex=None,
        table_html=None,
        cross_refs=[],
        **base,
    )
    return item


def _normalize_text(entry: dict, ctx: NormalizeContext, base: dict) -> ContentItem:
    """Normalize a text block to clause or paragraph."""
    text = entry.get("text", "").strip()
    if not text:
        return None

    # Extract cross-references
    cross_refs = extract_cross_refs(text)

    # Check if it starts with a clause number
    match = CLAUSE_NUMBER_RE.match(text)
    if match:
        number = match.group(0)
        item_type = ItemType.CLAUSE
    else:
        number = None
        item_type = ItemType.PARAGRAPH

    # Parent is the last section in the path
    parent_id = None
    if ctx.section_path_stack:
        pass  # Simplification; full implementation would track section item_ids

    item = ContentItem(
        type=item_type,
        parent_id=parent_id,
        section_path=ctx.section_path_stack.copy(),
        number=number,
        text=text,
        latex=None,
        table_html=None,
        cross_refs=cross_refs,
        **base,
    )
    return item


def _normalize_equation(entry: dict, ctx: NormalizeContext, base: dict) -> ContentItem:
    """Normalize an equation entry."""
    latex = entry.get("latex") or entry.get("text", "")
    text = entry.get("text", latex)

    # Try to extract equation number from text (e.g., trailing "(Eq. 6.33)" or "(6.33)")
    number = None
    match = re.search(r"\((?:Eq\.?\s*)?(\d+(?:\.\d+){1,4})\)\s*$", text, re.IGNORECASE)
    if match:
        number = match.group(1)

    # Cross-refs from text
    cross_refs = extract_cross_refs(text)

    item = ContentItem(
        type=ItemType.EQUATION,
        parent_id=None,
        section_path=ctx.section_path_stack.copy(),
        number=number,
        text=text,
        latex=latex if latex != text else None,
        table_html=None,
        cross_refs=cross_refs,
        **base,
    )
    return item


def _normalize_table(entry: dict, ctx: NormalizeContext, base: dict) -> ContentItem:
    """Normalize a table entry."""
    # MinerU tables have "html" field with the table HTML
    table_html = entry.get("html") or entry.get("table_html") or ""
    # Caption/text
    text = entry.get("caption") or entry.get("text") or ""
    if not text and table_html:
        text = table_html[:500]  # Truncate for text field

    # Table number from caption
    number = None
    match = re.search(r"[Tt]able\s+(\d+(?:\.\d+)*)", text)
    if match:
        number = match.group(1)

    # Cross-refs from text
    cross_refs = extract_cross_refs(text)

    item = ContentItem(
        type=ItemType.TABLE,
        parent_id=None,
        section_path=ctx.section_path_stack.copy(),
        number=number,
        text=text,
        latex=None,
        table_html=table_html,
        cross_refs=cross_refs,
        **base,
    )
    return item


def _normalize_figure(entry: dict, ctx: NormalizeContext, base: dict) -> ContentItem:
    """Normalize an image/figure entry."""
    # MinerU images have "img_path" or "path" field
    img_path = entry.get("img_path") or entry.get("path") or ""
    caption = entry.get("caption") or entry.get("text") or ""

    # Figure number from caption
    number = None
    match = re.search(r"[Ff]igure\s+(\d+(?:\.\d+)*)", caption)
    if not match:
        match = re.search(r"[Ff]ig\.\s*(\d+(?:\.\d+)*)", caption)
    if match:
        number = match.group(1)

    # Cross-refs from caption
    cross_refs = extract_cross_refs(caption)

    # Record source image path in asset_ids as placeholder
    asset_ids = [img_path] if img_path else []

    item = ContentItem(
        type=ItemType.FIGURE,
        parent_id=None,
        section_path=ctx.section_path_stack.copy(),
        number=number,
        text=caption,
        latex=None,
        table_html=None,
        asset_ids=asset_ids,
        cross_refs=cross_refs,
        **base,
    )
    return item


def extract_cross_refs(text: str) -> list[str]:
    """
    Extract cross-references from text.

    Finds patterns like "clause 9.2.1", "Table 3.2", "Figure 9.5", "equation 6.33"
    (case-insensitive) and returns normalized bare number strings.

    Args:
        text: Text to search for cross-references.

    Returns:
        List of normalized reference strings (e.g., "9.2.1", "Table 3.2").
    """
    refs = []
    for pattern, ref_type in CROSS_REF_PATTERNS:
        for match in pattern.finditer(text):
            number = match.group(1)
            # Normalize to "Type Number" format
            refs.append(f"{ref_type.capitalize()} {number}")

    # Deduplicate while preserving order
    seen = set()
    unique_refs = []
    for r in refs:
        if r not in seen:
            seen.add(r)
            unique_refs.append(r)

    return unique_refs