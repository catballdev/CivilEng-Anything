"""
CLI for CivilEng-Anything.

Provides subcommands: doctor, ingest, status, search, clause, table, figure,
equation, page, manifest, tree, build-vectors, serve-api, serve-mcp.
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
from .import_parse import ImportParser, register_document_if_needed
from .pdfinspect import inspect_pdf
from .retrieval import (
    list_documents,
    get_document_manifest,
    search,
    get_section,
    get_clause,
    get_equation,
    get_table,
    get_figure,
    get_page,
    get_related,
    SearchMode,
)
from .embeddings import get_embedder, HashingEmbedder


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

    # import-parse
    import_parse_parser = subparsers.add_parser(
        "import-parse",
        help="Import an existing MinerU parse output directory",
    )
    import_parse_parser.add_argument("--document-id", required=True, help="Document ID (must be registered)")
    import_parse_parser.add_argument("--edition-id", required=True, help="Edition ID")
    import_parse_parser.add_argument("--mineru-dir", required=True, type=Path, help="Path to MinerU output directory (e.g., .../auto/)")
    import_parse_parser.add_argument("--parser-name", default="mineru", help="Parser name")
    import_parse_parser.add_argument("--parser-version", default="3.4.5", help="Parser version")
    import_parse_parser.add_argument("--parse-location", default="Kaggle (GPU T4 x2)", help="Where the parse was run")
    import_parse_parser.add_argument("--parse-date", default=None, help="Parse date (ISO format, default: today)")
    import_parse_parser.add_argument("--register-if-missing", action="store_true", help="Register document if not found (requires --title, --pdf-path)")
    import_parse_parser.add_argument("--title", default=None, help="Document title (required if --register-if-missing)")
    import_parse_parser.add_argument("--pdf-path", default=None, type=Path, help="Path to original PDF (required if --register-if-missing)")
    import_parse_parser.add_argument("--document-family-id", default=None, help="Document family ID (for registration)")
    import_parse_parser.add_argument("--publisher", default=None, help="Publisher (for registration)")
    import_parse_parser.add_argument("--jurisdiction", default=None, help="Jurisdiction (for registration)")
    import_parse_parser.add_argument("--doc-type", default="government_code", choices=[e.value for e in DocType], help="Document type (for registration)")
    import_parse_parser.add_argument("--language", default="en", help="Language code (for registration)")
    import_parse_parser.add_argument("--json", action="store_true", help="Output as JSON")

    # status
    status_parser = subparsers.add_parser(
        "status",
        help="Show per-document page status counts",
    )
    status_parser.add_argument("--document-id", default=None, help="Filter by document ID")

    # search
    search_parser = subparsers.add_parser(
        "search",
        help="Search documents (exact/keyword/semantic/hybrid/hybrid-lexical)",
    )
    search_parser.add_argument("query", help="Search query")
    search_parser.add_argument("--document-id", default=None, help="Filter by document ID")
    search_parser.add_argument("--edition-id", default=None, help="Filter by edition ID")
    search_parser.add_argument("--jurisdiction", default=None, help="Filter by jurisdiction")
    search_parser.add_argument("--doc-type", default=None, help="Filter by document type")
    search_parser.add_argument("--mode", default="hybrid-lexical",
                               choices=["exact", "keyword", "semantic", "hybrid", "hybrid-lexical"],
                               help="Search mode (default: hybrid-lexical, vectorless-first)")
    search_parser.add_argument("--top-k", type=int, default=8, help="Maximum results")
    search_parser.add_argument("--token-budget", type=int, default=6000, help="Token budget for results")
    search_parser.add_argument("--expand", action="store_true", help="Expand results with context")
    search_parser.add_argument("--json", action="store_true", help="Output as JSON")

    # clause
    clause_parser = subparsers.add_parser(
        "clause",
        help="Get exact clause with children, footnotes, amendment links",
    )
    clause_parser.add_argument("--document-id", required=True, help="Document ID")
    clause_parser.add_argument("number", help="Clause number")
    clause_parser.add_argument("--json", action="store_true", help="Output as JSON")

    # table
    table_parser = subparsers.add_parser(
        "table",
        help="Get exact table with HTML, header context, footnotes",
    )
    table_parser.add_argument("--document-id", required=True, help="Document ID")
    table_parser.add_argument("number", help="Table number")
    table_parser.add_argument("--json", action="store_true", help="Output as JSON")

    # figure
    figure_parser = subparsers.add_parser(
        "figure",
        help="Get exact figure with asset info",
    )
    figure_parser.add_argument("--document-id", required=True, help="Document ID")
    figure_parser.add_argument("number", help="Figure number")
    figure_parser.add_argument("--json", action="store_true", help="Output as JSON")

    # equation
    equation_parser = subparsers.add_parser(
        "equation",
        help="Get exact equation",
    )
    equation_parser.add_argument("--document-id", required=True, help="Document ID")
    equation_parser.add_argument("number", help="Equation number")
    equation_parser.add_argument("--json", action="store_true", help="Output as JSON")

    # page
    page_parser = subparsers.add_parser(
        "page",
        help="Get all items on a PDF page",
    )
    page_parser.add_argument("--document-id", required=True, help="Document ID")
    page_parser.add_argument("pdf_page", type=int, help="PDF page number")
    page_parser.add_argument("--json", action="store_true", help="Output as JSON")

    # manifest
    manifest_parser = subparsers.add_parser(
        "manifest",
        help="Get document manifest with page counts and amendment relations",
    )
    manifest_parser.add_argument("--document-id", required=True, help="Document ID")
    manifest_parser.add_argument("--json", action="store_true", help="Output as JSON")

    # tree
    tree_parser = subparsers.add_parser(
        "tree",
        help="Show document clause tree structure",
    )
    tree_parser.add_argument("--document-id", required=True, help="Document ID")
    tree_parser.add_argument("--edition-id", default=None, help="Edition ID (default: latest)")
    tree_parser.add_argument("--json", action="store_true", help="Output as JSON")

    # build-vectors
    build_vectors_parser = subparsers.add_parser(
        "build-vectors",
        help="Build vector index for current embedder",
    )
    build_vectors_parser.add_argument("--document-id", default=None, help="Optional document ID to limit build")
    build_vectors_parser.add_argument("--json", action="store_true", help="Output as JSON")

    # serve-api
    serve_api_parser = subparsers.add_parser(
        "serve-api",
        help="Start FastAPI HTTP server",
    )
    serve_api_parser.add_argument("--host", default=None, help="Host (default from settings)")
    serve_api_parser.add_argument("--port", type=int, default=None, help="Port (default from settings)")

    # serve-mcp
    serve_mcp_parser = subparsers.add_parser(
        "serve-mcp",
        help="Start MCP server (stdio or HTTP)",
    )
    serve_mcp_parser.add_argument("--transport", choices=["stdio", "http"], default="stdio", help="Transport mode")
    serve_mcp_parser.add_argument("--host", default="0.0.0.0", help="Host for HTTP transport")
    serve_mcp_parser.add_argument("--port", type=int, default=8001, help="Port for HTTP transport")

    return parser


def _get_store_and_embedder(settings: Settings):
    """Get store and embedder (with HashingEmbedder fallback for tests)."""
    store = Store(settings.db_path)
    try:
        embedder = get_embedder(settings)
    except RuntimeError:
        embedder = HashingEmbedder()
    return store, embedder


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


def cmd_import_parse(args: argparse.Namespace, settings: Settings) -> int:
    """Import an existing MinerU parse output."""
    store = Store(settings.db_path)
    
    try:
        # Ensure document is registered (and edition_id is synced)
        if args.register_if_missing:
            if not args.title or not args.pdf_path:
                print("Error: --title and --pdf-path are required with --register-if-missing", file=sys.stderr)
                return 1
            if not args.pdf_path.exists():
                print(f"Error: PDF file not found: {args.pdf_path}", file=sys.stderr)
                return 1
            
            doc = register_document_if_needed(
                settings=settings,
                store=store,
                document_id=args.document_id,
                edition_id=args.edition_id,
                title=args.title,
                pdf_path=args.pdf_path,
                document_family_id=args.document_family_id,
                publisher=args.publisher,
                jurisdiction=args.jurisdiction,
                doc_type=DocType(args.doc_type),
                language=args.language,
                parser_name=args.parser_name,
                parser_version=args.parser_version,
            )
            print(f"Registered/updated document: {doc.document_id} ({doc.edition_id})")
        else:
            # Document must exist; sync edition_id if needed
            doc = store.get_document(args.document_id)
            if doc is None:
                print(f"Error: Document not registered: {args.document_id}", file=sys.stderr)
                print("Use --register-if-missing with --title and --pdf-path to register first.", file=sys.stderr)
                return 1
            # Sync edition_id if it differs
            if doc.edition_id != args.edition_id:
                doc.edition_id = args.edition_id
                store.upsert_document(doc)
                print(f"Updated document edition_id: {doc.document_id} ({doc.edition_id})")
        
        # Import the parse
        importer = ImportParser(settings, store)
        
        report = importer.import_parse(
            document_id=args.document_id,
            edition_id=args.edition_id,
            mineru_dir=args.mineru_dir,
            parser_name=args.parser_name,
            parser_version=args.parser_version,
            parse_location=args.parse_location,
            parse_date=args.parse_date,
        )
        
        if args.json:
            print(json.dumps(report.to_dict(), indent=2))
        else:
            print(f"Import complete for {args.document_id} ({args.edition_id})")
            print(f"  Run ID: {report.run_id}")
            print(f"  Pages done: {len(report.pages_done)}")
            print(f"  Pages failed: {len(report.pages_failed)}")
            print(f"  Items created: {report.items_created}")
            print(f"  Items updated: {report.items_updated}")
            print(f"  Parser: {report.parser} {report.parser_version}")
            print(f"  Location: {report.parse_location}")
            print(f"  Date: {report.parse_date}")
        
        return 0
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    finally:
        store.close()


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


def cmd_search(args: argparse.Namespace, settings: Settings) -> int:
    """Search documents."""
    store, embedder = _get_store_and_embedder(settings)

    try:
        mode = SearchMode(args.mode)
    except ValueError:
        print(f"Error: Invalid mode: {args.mode}", file=sys.stderr)
        return 1

    response = search(
        store=store,
        query=args.query,
        document_id=args.document_id,
        edition_id=args.edition_id,
        jurisdiction=args.jurisdiction,
        doc_type=args.doc_type,
        mode=args.mode,
        top_k=args.top_k,
        token_budget=args.token_budget,
        expand=args.expand,
        embedder=embedder,
        settings=settings,
    )

    if args.json:
        print(json.dumps({
            "results": [r.to_dict() for r in response.results],
            "mode": response.mode.value,
            "baseline_editions": response.baseline_editions,
            "amendments_included": response.amendments_included,
            "has_more_context": response.has_more_context,
            "total_matches": response.total_matches,
            "degraded": response.degraded,
            "degradation_reason": response.degradation_reason,
        }, indent=2))
    else:
        print(f"Mode: {response.mode.value} | Matches: {response.total_matches} | Degraded: {response.degraded}")
        if response.degradation_reason:
            print(f"  Degradation: {response.degradation_reason}")
        print()
        for i, result in enumerate(response.results):
            print(f"  [{i+1}] {result.item_id} - Score: {result.score:.4f}")
            if result.number:
                print(f"       Number: {result.number}")
            print(f"       Document: {result.document_id} ({result.edition_id})")
            if result.match_type:
                print(f"       Match: {result.match_type}")
            if result.has_more_context:
                print(f"       [has more context]")
            # Print snippet
            snippet = result.snippet[:200] + ("..." if len(result.snippet) > 200 else "")
            print(f"       {snippet}")
            print()

    store.close()
    return 0


def _print_result(result, as_json: bool):
    """Print a single SearchResult."""
    if as_json:
        print(json.dumps(result.to_dict(), indent=2))
    else:
        # SearchResult has item_id, document_id, edition_id, doc_title, number, snippet, pdf_page, etc.
        print(f"Item: {result.item_id}")
        if result.number:
            print(f"Number: {result.number}")
        print(f"Document: {result.document_id} ({result.edition_id})")
        print(f"Title: {result.doc_title}")
        print(f"PDF Page: {result.pdf_page}")
        if result.printed_page:
            print(f"Printed Page: {result.printed_page}")
        print(f"Text: {result.snippet}")
        if result.has_more_context:
            print("[has more context]")


def cmd_clause(args: argparse.Namespace, settings: Settings) -> int:
    """Get clause."""
    store, _ = _get_store_and_embedder(settings)
    result = get_clause(store, args.document_id, args.number)
    store.close()
    if result is None:
        print(f"Clause not found: {args.number}", file=sys.stderr)
        return 1
    _print_result(result, args.json)
    return 0


def cmd_table(args: argparse.Namespace, settings: Settings) -> int:
    """Get table."""
    store, _ = _get_store_and_embedder(settings)
    result = get_table(store, args.document_id, args.number)
    store.close()
    if result is None:
        print(f"Table not found: {args.number}", file=sys.stderr)
        return 1
    _print_result(result, args.json)
    return 0


def cmd_figure(args: argparse.Namespace, settings: Settings) -> int:
    """Get figure."""
    store, _ = _get_store_and_embedder(settings)
    result = get_figure(store, args.document_id, args.number)
    store.close()
    if result is None:
        print(f"Figure not found: {args.number}", file=sys.stderr)
        return 1
    _print_result(result, args.json)
    return 0


def cmd_equation(args: argparse.Namespace, settings: Settings) -> int:
    """Get equation."""
    store, _ = _get_store_and_embedder(settings)
    result = get_equation(store, args.document_id, args.number)
    store.close()
    if result is None:
        print(f"Equation not found: {args.number}", file=sys.stderr)
        return 1
    _print_result(result, args.json)
    return 0


def cmd_page(args: argparse.Namespace, settings: Settings) -> int:
    """Get page items."""
    store, _ = _get_store_and_embedder(settings)
    results = get_page(store, args.document_id, args.pdf_page)
    store.close()
    if args.json:
        print(json.dumps([r.to_dict() for r in results], indent=2))
    else:
        print(f"Page {args.pdf_page} of {args.document_id}: {len(results)} items")
        for i, result in enumerate(results):
            item = result.item
            print(f"  [{i+1}] {item.item_id} ({item.type.value}) - {item.number or ''}")
            text = item.text or ""
            print(f"       {text[:200]}{'...' if len(text) > 200 else ''}")
    return 0


def cmd_manifest(args: argparse.Namespace, settings: Settings) -> int:
    """Get document manifest."""
    store, _ = _get_store_and_embedder(settings)
    manifest = get_document_manifest(store, args.document_id)
    store.close()
    if manifest is None:
        print(f"Document not found: {args.document_id}", file=sys.stderr)
        return 1
    if args.json:
        print(json.dumps(manifest, indent=2))
    else:
        doc = manifest["document"]
        counts = manifest["page_counts"]
        print(f"Document: {doc.document_id} ({doc.edition_id})")
        print(f"  Title: {doc.title}")
        print(f"  Pages: total={counts['total']} done={counts['done']} failed={counts['failed']} retry={counts['retry_pending']} review={counts['needs_review']} pending={counts['pending']}")
        if manifest["amendments"]:
            print("  Amendments:")
            for am in manifest["amendments"]:
                print(f"    - {am['document_id']} ({am['edition_id']}): {am['title']}")
    return 0


def cmd_tree(args: argparse.Namespace, settings: Settings) -> int:
    """Show document clause tree structure."""
    from .tree import build_document_tree, get_tree_as_text
    store, _ = _get_store_and_embedder(settings)
    tree_data = build_document_tree(store, args.document_id, args.edition_id)
    store.close()
    if tree_data is None:
        print(f"Document not found: {args.document_id}", file=sys.stderr)
        return 1
    if args.json:
        print(json.dumps(tree_data, indent=2, ensure_ascii=False))
    else:
        print(get_tree_as_text(tree_data))
    return 0


def cmd_build_vectors(args: argparse.Namespace, settings: Settings) -> int:
    """Build vector index."""
    store, embedder = _get_store_and_embedder(settings)

    from .index import VectorIndex
    index = VectorIndex(settings.db_path)

    result = index.build_vectors(store, embedder, document_id=args.document_id)
    store.close()

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print(f"Built: {result['built_count']} vectors")
        print(f"Total items: {result['total_items']}")
    return 0


def cmd_serve_api(args: argparse.Namespace, settings: Settings) -> int:
    """Start FastAPI HTTP server."""
    try:
        import uvicorn
    except ImportError:
        print("Error: uvicorn not installed. Install with: pip install uvicorn", file=sys.stderr)
        return 1

    host = args.host or settings.host
    port = args.port or settings.api_port

    print(f"Starting API server on {host}:{port}...")
    # Import here to avoid loading at CLI startup
    from .api import app
    uvicorn.run(app, host=host, port=port)
    return 0


def cmd_serve_mcp(args: argparse.Namespace, settings: Settings) -> int:
    """Start MCP server."""
    try:
        from .mcp_server import run_stdio_server, run_http_server
    except ImportError as e:
        print(f"Error: MCP not available: {e}", file=sys.stderr)
        return 1

    if args.transport == "stdio":
        import asyncio
        asyncio.run(run_stdio_server())
    else:
        run_http_server(args.host, args.port)
    return 0


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
    elif args.command == "import-parse":
        return cmd_import_parse(args, settings)
    elif args.command == "status":
        return cmd_status(args, settings)
    elif args.command == "search":
        return cmd_search(args, settings)
    elif args.command == "clause":
        return cmd_clause(args, settings)
    elif args.command == "table":
        return cmd_table(args, settings)
    elif args.command == "figure":
        return cmd_figure(args, settings)
    elif args.command == "equation":
        return cmd_equation(args, settings)
    elif args.command == "page":
        return cmd_page(args, settings)
    elif args.command == "manifest":
        return cmd_manifest(args, settings)
    elif args.command == "tree":
        return cmd_tree(args, settings)
    elif args.command == "build-vectors":
        return cmd_build_vectors(args, settings)
    elif args.command == "serve-api":
        return cmd_serve_api(args, settings)
    elif args.command == "serve-mcp":
        return cmd_serve_mcp(args, settings)
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())