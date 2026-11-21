"""
MinerU CLI adapter for CivilEng-Anything.

Invokes the MinerU executable with explicit page ranges and archives outputs.
"""

import json
import logging
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class MineruResult:
    """Result from a MinerU parsing run."""
    content_list_path: Path
    markdown_path: Path
    images_dir: Path
    raw_dir: Path
    pages_done: list[int]  # 1-based page numbers actually parsed


def _find_mineru_executable() -> Path:
    """
    Locate the MinerU executable in the current venv or PATH.

    Returns:
        Path to the mineru executable.
    """
    # Try venv bin directory first
    venv_bin = Path(sys.executable).parent / "mineru"
    if venv_bin.exists() and os.access(venv_bin, os.X_OK):
        return venv_bin

    # Fallback to PATH
    for path_dir in os.environ.get("PATH", "").split(os.pathsep):
        candidate = Path(path_dir) / "mineru"
        if candidate.exists() and os.access(candidate, os.X_OK):
            return candidate

    # Last resort: try python -m mineru
    return Path(sys.executable)


def run_mineru(
    pdf_path: Path,
    out_dir: Path,
    start_page_1based: int,
    end_page_1based: int,
    lang: str = "en",
    method: str = "auto",
) -> MineruResult:
    """
    Run MinerU on a PDF page range.

    MinerU CLI uses 0-based page indices for -s (start) and -e (end).
    We convert from 1-based to 0-based carefully.

    Args:
        pdf_path: Path to the PDF file.
        out_dir: Output directory for MinerU results.
        start_page_1based: First page to parse (1-based, inclusive).
        end_page_1based: Last page to parse (1-based, inclusive).
        lang: Language code (default: "en").
        method: Parsing method (default: "auto").

    Returns:
        MineruResult with paths to outputs and list of pages actually parsed.

    Raises:
        RuntimeError: If MinerU fails or content_list.json is not found.
    """
    # Convert to 0-based for MinerU CLI
    # -s is start page index (0-based), -e is end page index (0-based, inclusive)
    start_0based = start_page_1based - 1
    end_0based = end_page_1based - 1

    # Build command
    mineru_cmd = _find_mineru_executable()
    if mineru_cmd == Path(sys.executable):
        # Use python -m mineru
        cmd = [
            str(mineru_cmd),
            "-m", "mineru",
            "-p", str(pdf_path),
            "-o", str(out_dir),
            "-s", str(start_0based),
            "-e", str(end_0based),
            "-l", lang,
            "-m", method,
            "-b", "pipeline",  # backend
        ]
    else:
        cmd = [
            str(mineru_cmd),
            "-p", str(pdf_path),
            "-o", str(out_dir),
            "-s", str(start_0based),
            "-e", str(end_0based),
            "-l", lang,
            "-m", method,
            "-b", "pipeline",
        ]

    # Log the exact command (no secrets)
    logger.info("Running MinerU: %s", " ".join(str(c) for c in cmd))

    out_dir.mkdir(parents=True, exist_ok=True)

    # Prepare environment with sanitized proxy settings
    # MinerU talks to its own local API service; inherited NO_PROXY causes httpx crashes
    env = os.environ.copy()
    env["NO_PROXY"] = "127.0.0.1,localhost"
    env["no_proxy"] = "127.0.0.1,localhost"

    # Run MinerU
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=300,  # 5 minutes max
        env=env,
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"MinerU failed (exit {result.returncode}):\n"
            f"stdout: {result.stdout}\n"
            f"stderr: {result.stderr}"
        )

    # Find outputs by globbing the MinerU output tree
    # MinerU creates: out_dir/<pdf_stem>/<method>/content_list.json, etc.
    pdf_stem = pdf_path.stem
    method_dir = out_dir / pdf_stem / method

    # Find content_list.json
    content_list_paths = list(method_dir.glob("**/content_list.json"))
    if not content_list_paths:
        # List what was actually found for debugging
        found = list(method_dir.rglob("*"))
        raise RuntimeError(
            f"MinerU content_list.json not found in {method_dir}. "
            f"Found: {[str(f) for f in found]}"
        )
    content_list_path = content_list_paths[0]

    # Find markdown file
    md_paths = list(method_dir.glob("**/*.md"))
    markdown_path = md_paths[0] if md_paths else method_dir / f"{pdf_stem}.md"

    # Images directory
    images_dir = method_dir / "images"
    if not images_dir.exists():
        images_dir = method_dir

    # Raw directory is the method_dir itself
    raw_dir = method_dir

    # Determine pages actually present in output by reading content_list
    pages_done = _extract_pages_from_content_list(content_list_path)

    return MineruResult(
        content_list_path=content_list_path,
        markdown_path=markdown_path,
        images_dir=images_dir,
        raw_dir=raw_dir,
        pages_done=pages_done,
    )


def _extract_pages_from_content_list(content_list_path: Path) -> list[int]:
    """
    Extract page numbers (1-based) from MinerU content_list.json.

    MinerU content_list entries have a "page_idx" field (0-based).
    """
    try:
        with open(content_list_path, "r", encoding="utf-8") as f:
            content_list = json.load(f)
    except Exception as e:
        raise RuntimeError(f"Failed to read content_list.json: {e}")

    pages = set()
    for entry in content_list:
        page_idx = entry.get("page_idx")
        if isinstance(page_idx, int):
            # Convert to 1-based
            pages.add(page_idx + 1)

    return sorted(pages)