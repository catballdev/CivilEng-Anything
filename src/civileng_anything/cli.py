"""
CLI for CivilEng-Anything.

Provides subcommands: doctor, ingest, status.
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import date

from .config import Settings, get_settings
from .store import Store
from .schema import DocumentMeta, DocType
from .ingest import Ingestor
from .pdfinspect import inspect_pdf


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser."""
    parser = argparse.ArgumentParser(
        prog="civileng",
        description="CivilEng-Anything: traceable knowledge base for civil/structural engineering documents",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Data directory (overrides CIVILENG_DATA_DIR env var)",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # doctor
    doctor_parser = subparsers.add_parser(
        "doctor",
        help="Print environment diagnostics",
    )

    # ingest
    ingest_parser = subparsers.add_parser(
        "ingest",
        help="Ingest a PDF document",
    )
    ingest_parser.add_argument("path", type=Path, help="Path to PDF file")
    ingest_parser.add_argument("--document-id", required=True, help="Document ID")
    ingest_parser.add_argument("--edition-id", required=True, help="Edition ID")
    ingest_parser.add_argument("--title", required=True, help="Document title")
    ingest_parser.add_argument("--document-family-id", default=None, help="Document family ID")
    ingest_parser.add_argument("--source-url", default=None, help="Source URL")
    ingest_parser.add_argument("--publisher", default=None, help="Publisher")
    ingest_parser.add_argument("--jurisdiction", default=None, help="Jurisdiction")
    ingest_parser.add_argument("--doc-type", default="government_code",
                               choices=[e.value for e in DocType],
                               help="Document type")
    ingest_parser.add_argument("--language", default="en", help="Language code")
    ingest_parser.add_argument("--parser-name", default="mineru", help="Parser name")
    ingest_parser.add_argument("--parser-version", default="3.4.5", help="Parser version")

    # status
    status_parser = subparsers.add_parser(
        "status",
        help="Show per-document page status counts",
    )
    status_parser.add_argument("--document-id", default=None, help="Filter by document ID")

    return parser


def cmd_doctor(args: argparse.Namespace, settings: Settings) -> int:
    """Run doctor diagnostics."""
    print("=== CivilEng-Anything Doctor ===")
    print(f"Python: {sys.version.split()[0]}")
    print(f"Data dir: {settings.data_dir}")
    print(f"DB path: {settings.db_path}")
    print(f"Originals dir: {settings.originals_dir}")
    print(f"Parsed dir: {settings.parsed_dir}")
    print(f"Assets dir: {settings.assets_dir}")
    print(f"Indexes dir: {settings.indexes_dir}")
    print(f"Embed model: {settings.embed_model}")
    print(f"Top-K: {settings.top_k}")
    print(f"Token budget: {settings.token_budget}")
    print(f"API host:port: {settings.host}:{settings.api_port}")

    # Check optional dependencies
    print("\n--- Optional Dependencies ---")
    for name, import_path in [
        ("mineru", "mineru"),
        ("docling", "docling"),
        ("torch", "torch"),
        ("sentence_transformers", "sentence_transformers"),
        ("fastapi", "fastapi"),
        ("uvicorn", "uvicorn"),
        ("mcp", "mcp"),
    ]:
        try:
            __import__(import_path)
            print(f"  {name}: available")
        except ImportError:
            print(f"  {name}: NOT available")

    # Check database
    print("\n--- Database ---")
    try:
        store = Store(settings.db_path)
        docs = store.list_documents()
        print(f"  Documents: {len(docs)}")
        for doc in docs:
            print(f"    - {doc.document_id} ({doc.title})")
        store.close()
    except Exception as e:
        print(f"  Error: {e}")

    return 0


def cmd_ingest(args: argparse.Namespace, settings: Settings) -> int:
    """Ingest a PDF document."""
    pdf_path = args.path
    if not pdf_path.exists():
        print(f"Error: File not found: {pdf_path}", file=sys.stderr)
        return 1

    # Build DocumentMeta
    doc_family_id = args.document_family_id or args.document_id
    meta = DocumentMeta(
        document_id=args.document_id,
        document_family_id=doc_family_id,
        edition_id=args.edition_id,
        title=args.title,
        publisher=args.publisher,
        jurisdiction=args.jurisdiction,
        doc_type=DocType(args.doc_type),
        language=args.language,
        publication_date=None,
        revision_date=None,
        source_url=args.source_url,
        retrieved_at=date.today(),
        original_sha256="",  # Will be computed
        total_pdf_pages=0,   # Will be set after inspection
        parser_name=args.parser_name,
        parser_version=args.parser_version,
        parser_config={},
        parse_run_id="",     # Will be set after ingest
        completeness=0.0,
        qc_status="pending",
        amendment_relations=[],
        version_relations=[],
    )

    # Run ingest
    store = Store(settings.db_path)
    ingestor = Ingestor(settings, store)

    try:
        report = ingestor.ingest_file(pdf_path, meta, source_url=args.source_url)
        print(json.dumps(report.to_dict(), indent=2))
        return 0
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    finally:
        store.close()


def cmd_status(args: argparse.Namespace, settings: Settings) -> int:
    """Show page status counts."""
    store = Store(settings.db_path)

    try:
        if args.document_id:
            docs = [store.get_document(args.document_id)]
            if docs[0] is None:
                print(f"Document not found: {args.document_id}")
                return 1
        else:
            docs = store.list_documents()

        for doc in docs:
            page_statuses = store.get_page_statuses(doc.document_id)
            counts = {}
            for ps in page_statuses:
                counts[ps.status.value] = counts.get(ps.status.value, 0) + 1

            total = doc.total_pdf_pages
            done = counts.get("done", 0)
            failed = counts.get("failed", 0)
            retry = counts.get("retry_pending", 0)
            review = counts.get("needs_review", 0)

            print(f"{doc.document_id} ({doc.edition_id}): {total} pages")
            print(f"  done: {done}, failed: {failed}, retry_pending: {retry}, needs_review: {review}")
            if done + failed + retry + review < total:
                print(f"  pending: {total - (done + failed + retry + review)}")

        return 0
    finally:
        store.close()


def main() -> int:
    """Main entry point."""
    parser = build_parser()
    args = parser.parse_args()

    # Load settings
    if args.data_dir:
        import os
        os.environ["CIVILENG_DATA_DIR"] = str(args.data_dir)

    settings = get_settings()

    # Dispatch
    if args.command == "doctor":
        return cmd_doctor(args, settings)
    elif args.command == "ingest":
        return cmd_ingest(args, settings)
    elif args.command == "status":
        return cmd_status(args, settings)
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())