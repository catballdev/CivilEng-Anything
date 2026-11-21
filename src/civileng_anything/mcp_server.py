"""
MCP Server for CivilEng-Anything.

Provides stdio and Streamable HTTP transports.
Lazy imports so non-server commands work without mcp.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any, Optional

from .config import Settings, get_settings
from .schema import SearchMode, ItemType, SearchResult
from .store import Store
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
    SearchResponse,
)
from .embeddings import get_embedder, HashingEmbedder

logger = logging.getLogger(__name__)

# Global state
_store: Optional[Store] = None
_settings: Optional[Settings] = None
_embedder = None


def initialize_globals(settings: Optional[Settings] = None):
    """Initialize global state."""
    global _store, _settings, _embedder
    if settings is None:
        settings = get_settings()
    _settings = settings
    _store = Store(settings.db_path)
    try:
        _embedder = get_embedder(settings)
    except RuntimeError:
        logger.warning("No real embedder available; using HashingEmbedder for MCP")
        _embedder = HashingEmbedder()


def get_store() -> Store:
    if _store is None:
        raise RuntimeError("Store not initialized")
    return _store


def get_embedder_dep():
    if _embedder is None:
        raise RuntimeError("Embedder not initialized")
    return _embedder


# ===== MCP Tool Definitions =====

TOOLS = [
    {
        "name": "list_documents",
        "description": "List all documents in the knowledge base",
        "inputSchema": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
    {
        "name": "get_document_manifest",
        "description": "Get document metadata with page-status counts and amendment relations",
        "inputSchema": {
            "type": "object",
            "properties": {
                "document_id": {"type": "string", "description": "Document ID"},
            },
            "required": ["document_id"],
        },
    },
    {
        "name": "search",
        "description": "Unified search across exact, keyword, semantic, and hybrid modes",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"},
                "document_id": {"type": "string", "description": "Filter by document ID"},
                "edition_id": {"type": "string", "description": "Filter by edition ID"},
                "jurisdiction": {"type": "string", "description": "Filter by jurisdiction"},
                "doc_type": {"type": "string", "description": "Filter by document type"},
                "mode": {
                    "type": "string",
                    "enum": ["exact", "keyword", "semantic", "hybrid"],
                    "default": "hybrid",
                    "description": "Search mode",
                },
                "top_k": {"type": "integer", "default": 8, "description": "Maximum results"},
                "token_budget": {"type": "integer", "default": 6000, "description": "Token budget for results"},
                "expand": {"type": "boolean", "default": False, "description": "Expand results with context"},
            },
            "required": ["query"],
        },
    },
    {
        "name": "get_section",
        "description": "Get a section item plus its descendant items in order",
        "inputSchema": {
            "type": "object",
            "properties": {
                "document_id": {"type": "string", "description": "Document ID"},
                "section_number": {"type": "string", "description": "Section number"},
            },
            "required": ["document_id", "section_number"],
        },
    },
    {
        "name": "get_clause",
        "description": "Get exact clause + child paragraphs + footnotes + amendment links",
        "inputSchema": {
            "type": "object",
            "properties": {
                "document_id": {"type": "string", "description": "Document ID"},
                "number": {"type": "string", "description": "Clause number"},
            },
            "required": ["document_id", "number"],
        },
    },
    {
        "name": "get_equation",
        "description": "Get exact equation",
        "inputSchema": {
            "type": "object",
            "properties": {
                "document_id": {"type": "string", "description": "Document ID"},
                "number": {"type": "string", "description": "Equation number"},
            },
            "required": ["document_id", "number"],
        },
    },
    {
        "name": "get_table",
        "description": "Get exact table with HTML, header context, footnotes",
        "inputSchema": {
            "type": "object",
            "properties": {
                "document_id": {"type": "string", "description": "Document ID"},
                "number": {"type": "string", "description": "Table number"},
            },
            "required": ["document_id", "number"],
        },
    },
    {
        "name": "get_figure",
        "description": "Get exact figure with asset IDs and file info (returns image as resource)",
        "inputSchema": {
            "type": "object",
            "properties": {
                "document_id": {"type": "string", "description": "Document ID"},
                "number": {"type": "string", "description": "Figure number"},
            },
            "required": ["document_id", "number"],
        },
    },
    {
        "name": "get_page",
        "description": "Get all items on a PDF page in stored order",
        "inputSchema": {
            "type": "object",
            "properties": {
                "document_id": {"type": "string", "description": "Document ID"},
                "pdf_page": {"type": "integer", "description": "PDF page number"},
            },
            "required": ["document_id", "pdf_page"],
        },
    },
    {
        "name": "get_related",
        "description": "Get cross-refs resolved to items, parent section, footnotes, amendment links",
        "inputSchema": {
            "type": "object",
            "properties": {
                "item_id": {"type": "string", "description": "Item ID"},
            },
            "required": ["item_id"],
        },
    },
    {
        "name": "build_vectors",
        "description": "Build vector index for the current embedder",
        "inputSchema": {
            "type": "object",
            "properties": {
                "document_id": {"type": "string", "description": "Optional document ID to limit build"},
            },
            "required": [],
        },
    },
]


def _result_to_dict(result: SearchResult) -> dict:
    """Convert SearchResult to JSON-serializable dict."""
    return result.to_dict()


def _handle_list_documents(args: dict) -> dict:
    store = get_store()
    docs = list_documents(store)
    return {"documents": [doc.model_dump() for doc in docs]}


def _handle_get_document_manifest(args: dict) -> dict:
    store = get_store()
    document_id = args["document_id"]
    manifest = get_document_manifest(store, document_id)
    if manifest is None:
        raise ValueError(f"Document not found: {document_id}")
    # Serialize DocumentMeta to dict
    if "document" in manifest and hasattr(manifest["document"], "model_dump"):
        manifest = dict(manifest)
        manifest["document"] = manifest["document"].model_dump()
    return manifest


def _handle_search(args: dict) -> dict:
    store = get_store()
    embedder = get_embedder_dep()

    try:
        mode = SearchMode(args.get("mode", "hybrid"))
    except ValueError:
        raise ValueError(f"Invalid mode: {args.get('mode')}")

    response = search(
        store=store,
        query=args["query"],
        document_id=args.get("document_id"),
        edition_id=args.get("edition_id"),
        jurisdiction=args.get("jurisdiction"),
        doc_type=args.get("doc_type"),
        mode=args.get("mode", "hybrid"),
        top_k=args.get("top_k", 8),
        token_budget=args.get("token_budget", 6000),
        expand=args.get("expand", False),
        embedder=embedder,
        settings=_settings,
    )

    return {
        "results": [_result_to_dict(r) for r in response.results],
        "mode": response.mode.value,
        "baseline_editions": response.baseline_editions,
        "amendments_included": response.amendments_included,
        "has_more_context": response.has_more_context,
        "total_matches": response.total_matches,
        "degraded": response.degraded,
        "degradation_reason": response.degradation_reason,
    }


def _handle_get_section(args: dict) -> dict:
    store = get_store()
    result = get_section(store, args["document_id"], args["section_number"])
    if result is None:
        raise ValueError("Section not found")
    return _result_to_dict(result)


def _handle_get_clause(args: dict) -> dict:
    store = get_store()
    result = get_clause(store, args["document_id"], args["number"])
    if result is None:
        raise ValueError("Clause not found")
    return _result_to_dict(result)


def _handle_get_equation(args: dict) -> dict:
    store = get_store()
    result = get_equation(store, args["document_id"], args["number"])
    if result is None:
        raise ValueError("Equation not found")
    return _result_to_dict(result)


def _handle_get_table(args: dict) -> dict:
    store = get_store()
    result = get_table(store, args["document_id"], args["number"])
    if result is None:
        raise ValueError("Table not found")
    return _result_to_dict(result)


def _handle_get_figure(args: dict) -> dict:
    store = get_store()
    result = get_figure(store, args["document_id"], args["number"])
    if result is None:
        raise ValueError("Figure not found")

    # For MCP, include asset as resource
    result_dict = _result_to_dict(result)

    # Add asset resources if any - need to fetch item from store to get asset_ids
    item = store.get_item(result.item_id)
    if item and item.asset_ids:
        resources = []
        for asset_id in item.asset_ids:
            asset = store.get_asset(asset_id)
            if asset:
                # Read image data and encode as base64 for MCP resource
                import base64
                asset_path = Path(asset.file_path)
                if asset_path.exists():
                    with open(asset_path, "rb") as f:
                        data = base64.b64encode(f.read()).decode()
                    resources.append({
                        "uri": f"asset://{asset_id}",
                        "mimeType": asset.mime_type,
                        "blob": data,
                    })
        if resources:
            result_dict["resources"] = resources

    return result_dict


def _handle_get_page(args: dict) -> dict:
    store = get_store()
    results = get_page(store, args["document_id"], args["pdf_page"])
    return {"results": [_result_to_dict(r) for r in results]}


def _handle_get_related(args: dict) -> dict:
    store = get_store()
    result = get_related(store, args["item_id"])
    if result is None:
        raise ValueError("Item not found")
    return _result_to_dict(result)


def _handle_build_vectors(args: dict) -> dict:
    store = get_store()
    embedder = get_embedder_dep()

    from .index import VectorIndex

    index = VectorIndex(_settings.db_path)
    result = index.build_vectors(store, embedder, document_id=args.get("document_id"))
    return result


HANDLERS = {
    "list_documents": _handle_list_documents,
    "get_document_manifest": _handle_get_document_manifest,
    "search": _handle_search,
    "get_section": _handle_get_section,
    "get_clause": _handle_get_clause,
    "get_equation": _handle_get_equation,
    "get_table": _handle_get_table,
    "get_figure": _handle_get_figure,
    "get_page": _handle_get_page,
    "get_related": _handle_get_related,
    "build_vectors": _handle_build_vectors,
}


# ===== MCP Server (stdio) =====

async def run_stdio_server():
    """Run MCP server over stdio."""
    try:
        from mcp.server import Server
        from mcp.server.stdio import stdio_server
    except ImportError:
        print("MCP not installed. Install with: pip install mcp", file=sys.stderr)
        sys.exit(1)

    server = Server("civileng-anything")

    @server.list_tools()
    async def list_tools():
        return TOOLS

    @server.call_tool()
    async def call_tool(name: str, arguments: dict):
        handler = HANDLERS.get(name)
        if handler is None:
            raise ValueError(f"Unknown tool: {name}")
        try:
            result = handler(arguments)
            return [{"type": "text", "text": json.dumps(result, indent=2)}]
        except Exception as e:
            return [{"type": "text", "text": json.dumps({"error": str(e)}, indent=2)}]

    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


# ===== MCP Server (Streamable HTTP) =====

def create_http_app():
    """Create FastAPI app with MCP Streamable HTTP endpoint."""
    try:
        from fastapi import FastAPI
        from mcp.server.streamable_http import create_streamable_http_app
    except ImportError:
        raise RuntimeError("FastAPI and/or MCP not installed")

    http_app = FastAPI(title="CivilEng-Anything MCP", lifespan=None)

    # Initialize globals before creating MCP app
    initialize_globals()

    mcp_app = create_streamable_http_app(
        server_name="civileng-anything",
        list_tools=TOOLS,
        call_tool=lambda name, args: HANDLERS[name](args),
    )

    # Mount MCP app
    http_app.mount("/mcp", mcp_app)

    @http_app.get("/health")
    async def health():
        return {"status": "ok"}

    return http_app


def run_http_server(host: str = "0.0.0.0", port: int = 8001):
    """Run MCP server over Streamable HTTP."""
    try:
        import uvicorn
    except ImportError:
        print("Uvicorn not installed. Install with: pip install uvicorn", file=sys.stderr)
        sys.exit(1)

    app = create_http_app()
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="CivilEng-Anything MCP Server")
    parser.add_argument("--transport", choices=["stdio", "http"], default="stdio")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8001)
    args = parser.parse_args()

    if args.transport == "stdio":
        import asyncio
        asyncio.run(run_stdio_server())
    else:
        run_http_server(args.host, args.port)