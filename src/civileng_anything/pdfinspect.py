"""
PDF inspection utilities using pypdf.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from pypdf import PdfReader


@dataclass
class PdfInfo:
    """Information extracted from a PDF file."""
    page_count: int
    is_encrypted: bool
    encryption_detail: Optional[str]
    has_text: list[bool]  # 1-based: has_text[0] = page 1
    classification: str  # "text", "scanned", or "mixed"


def _encryption_algorithm_name(enc) -> str:
    """Map pypdf encryption V/R values to algorithm name."""
    v = getattr(enc, 'V', None)
    r = getattr(enc, 'R', None)

    if v == 1 and r == 2:
        return "RC4-40"
    elif v == 2 and r == 3:
        return "RC4-128"
    elif v == 4 and r == 4:
        return "AES-128"
    elif v == 5 and r == 5:
        return "AES-256"
    elif v is not None and r is not None:
        return f"V{v}R{r}"
    return "Unknown"


def inspect_pdf(path: Path) -> PdfInfo:
    """
    Inspect a PDF file and extract metadata.

    Args:
        path: Path to the PDF file.

    Returns:
        PdfInfo with page count, encryption info, per-page text presence,
        and document classification.
    """
    reader = PdfReader(str(path))

    # Encryption info
    is_encrypted = reader.is_encrypted
    encryption_detail = None
    if is_encrypted and reader._encryption is not None:
        enc = reader._encryption
        # pypdf stores encryption info in _encryption object
        algo = _encryption_algorithm_name(enc)
        # Permissions: print, modify, copy, annotate
        perms = getattr(enc, 'P', 0)
        perm_strs = []
        if perms & 4:  # print
            perm_strs.append("print:yes")
        else:
            perm_strs.append("print:no")
        if perms & 8:  # modify
            perm_strs.append("change:yes")
        else:
            perm_strs.append("change:no")
        if perms & 16:  # copy
            perm_strs.append("copy:yes")
        else:
            perm_strs.append("copy:no")
        encryption_detail = f"{algo}, {', '.join(perm_strs)}"

    page_count = len(reader.pages)
    has_text = []

    for page in reader.pages:
        try:
            text = page.extract_text()
            has_text.append(bool(text and text.strip()))
        except Exception:
            # If extraction fails, treat as no text
            has_text.append(False)

    # Classification
    if page_count == 0:
        classification = "scanned"
    else:
        text_pages = sum(has_text)
        ratio = text_pages / page_count
        if ratio >= 0.9:
            classification = "text"
        elif ratio <= 0.1:
            classification = "scanned"
        else:
            classification = "mixed"

    return PdfInfo(
        page_count=page_count,
        is_encrypted=is_encrypted,
        encryption_detail=encryption_detail,
        has_text=has_text,
        classification=classification,
    )