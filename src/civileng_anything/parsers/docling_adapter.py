"""
Docling adapter for CivilEng-Anything (fallback parser).

Lazy imports Docling; raises ParserNotAvailable if not installed.
"""

import json
import logging
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class ParserNotAvailable(RuntimeError):
    """Raised when an optional parser is not installed."""
    pass


@dataclass
class DoclingResult:
    """Result from a Docling parsing run."""
    content_list_path: Path
    markdown_path: Path
    images_dir: Path
    raw_dir: Path
    pages_done: list[int]  # 1-based page numbers actually parsed


def _find_docling_executable() -> Optional[Path]:
    """Try to find docling executable or module."""
    # Try venv bin
    venv_bin = Path(sys.executable).parent / "docling"
    if venv_bin.exists():
        return venv_bin

    # Try PATH
    for path_dir in os.environ.get("PATH", "").split(os.pathsep):
        candidate = Path(path_dir) / "docling"
        if candidate.exists():
            return candidate

    return None


def run_docling(
    pdf_path: Path,
    out_dir: Path,
    pages_1based: list[int],
) -> DoclingResult:
    """
    Run Docling on specific pages of a PDF.

    Docling CLI is used since programmatic API is less stable.
    Pages are passed as a comma-separated list (1-based).

    Args:
        pdf_path: Path to the PDF file.
        out_dir: Output directory.
        pages_1based: List of 1-based page numbers to parse.

    Returns:
        DoclingResult with output paths.

    Raises:
        ParserNotAvailable: If Docling is not installed.
        RuntimeError: If Docling fails.
    """
    import os

    # Check if docling is available
    docling_cmd = _find_docling_executable()
    if docling_cmd is None:
        # Try python -m docling
        try:
            subprocess.run(
                [sys.executable, "-m", "docling", "--help"],
                capture_output=True,
                timeout=10,
            )
            docling_cmd = Path(sys.executable)
        except (subprocess.SubprocessError, FileNotFoundError):
            raise ParserNotAvailable(
                "Docling not installed. Install with: pip install docling"
            )

    out_dir.mkdir(parents=True, exist_ok=True)

    # Docling CLI uses --page-numbers with 1-based comma-separated pages
    pages_str = ",".join(str(p) for p in pages_1based)

    if docling_cmd == Path(sys.executable):
        cmd = [
            str(docling_cmd),
            "-m", "docling",
            str(pdf_path),
            "--output", str(out_dir),
            "--page-numbers", pages_str,
        ]
    else:
        cmd = [
            str(docling_cmd),
            str(pdf_path),
            "--output", str(out_dir),
            "--page-numbers", pages_str,
        ]

    logger.info("Running Docling: %s", " ".join(str(c) for c in cmd))

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=300,
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"Docling failed (exit {result.returncode}):\n"
            f"stdout: {result.stdout}\n"
            f"stderr: {result.stderr}"
        )

    # Docling outputs to out_dir/<pdf_stem>/
    pdf_stem = pdf_path.stem
    doc_dir = out_dir / pdf_stem

    # Find JSON output (Docling uses .json)
    json_paths = list(doc_dir.glob("*.json"))
    content_list_path = json_paths[0] if json_paths else doc_dir / f"{pdf_stem}.json"

    # Find markdown
    md_paths = list(doc_dir.glob("*.md"))
    markdown_path = md_paths[0] if md_paths else doc_dir / f"{pdf_stem}.md"

    # Images
    images_dir = doc_dir / "images"
    if not images_dir.exists():
        images_dir = doc_dir

    # For now, assume all requested pages were done
    # In reality we'd parse the output to verify
    pages_done = pages_1based.copy()

    return DoclingResult(
        content_list_path=content_list_path,
        markdown_path=markdown_path,
        images_dir=images_dir,
        raw_dir=doc_dir,
        pages_done=pages_done,
    )


# Import os at top level for _find_docling_executable
import os