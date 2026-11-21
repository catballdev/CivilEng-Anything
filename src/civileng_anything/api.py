"""
FastAPI application for CivilEng-Anything.

Provides HTTP API over the shared retrieval core.
Lazy imports so non-server commands work without fastapi/uvicorn.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, HTTPException, Query, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .config import Settings, get_settings
from .schema import SearchMode, ItemType, DocType, SearchResult
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
from .tree import build_document_tree
from .embeddings import get_embedder, HashingEmbedder

logger = logging.getLogger(__name__)

# Global state (initialized at startup)
_store: Optional[Store] = None
_settings: Optional[Settings] = None
_embedder = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    global _store, _settings, _embedder
    _settings = get_settings()
    _store = Store(_settings.db_path)
    try:
        _embedder = get_embedder(_settings)
    except RuntimeError:
        logger.warning("No real embedder available; using HashingEmbedder for API")
        _embedder = HashingEmbedder()
    yield
    if _store:
        _store.close()


app = FastAPI(
    title="CivilEng-Anything API",
    description="Traceable knowledge base for civil/structural engineering documents",
    version="0.3.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_store() -> Store:
    if _store is None:
        raise HTTPException(status_code=503, detail="Store not initialized")
    return _store


def get_embedder_dep():
    if _embedder is None:
        raise HTTPException(status_code=503, detail="Embedder not initialized")
    return _embedder


# ===== Request/Response Models =====

class SearchRequest(BaseModel):
    query: str
    document_id: Optional[str] = None
    edition_id: Optional[str] = None
    jurisdiction: Optional[str] = None
    doc_type: Optional[str] = None
    mode: str = "hybrid-lexical"
    top_k: int = 8
    token_budget: int = 6000
    expand: bool = False


class SearchResponseModel(BaseModel):
    results: list[SearchResult]
    mode: str
    baseline_editions: list[str]
    amendments_included: bool
    has_more_context: bool
    total_matches: int
    degraded: bool = False
    degradation_reason: Optional[str] = None


class ManifestResponse(BaseModel):
    document: dict
    page_counts: dict
    amendments: list[dict]


# ===== Endpoints =====

@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "ok"}


@app.get("/documents")
async def list_documents_endpoint():
    """List all documents."""
    store = get_store()
    docs = list_documents(store)
    return [doc.model_dump() for doc in docs]


@app.get("/documents/{document_id}/manifest")
async def get_manifest(document_id: str):
    """Get document manifest with page counts and amendment relations."""
    store = get_store()
    manifest = get_document_manifest(store, document_id)
    if manifest is None:
        raise HTTPException(status_code=404, detail="Document not found")
    return manifest


@app.get("/documents/{document_id}/tree")
async def get_document_tree(document_id: str, edition_id: Optional[str] = None):
    """Get document clause tree structure."""
    store = get_store()
    tree_data = build_document_tree(store, document_id, edition_id)
    if tree_data is None:
        raise HTTPException(status_code=404, detail="Document not found")
    return tree_data


@app.post("/search", response_model=SearchResponseModel)
async def search_endpoint(request: SearchRequest):
    """Unified search endpoint."""
    store = get_store()
    embedder = get_embedder_dep()

    try:
        mode = SearchMode(request.mode)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid mode: {request.mode}")

    response = search(
        store=store,
        query=request.query,
        document_id=request.document_id,
        edition_id=request.edition_id,
        jurisdiction=request.jurisdiction,
        doc_type=request.doc_type,
        mode=request.mode,
        top_k=request.top_k,
        token_budget=request.token_budget,
        expand=request.expand,
        embedder=embedder,
        settings=_settings,
    )

    return SearchResponseModel(
        results=response.results,
        mode=response.mode.value,
        baseline_editions=response.baseline_editions,
        amendments_included=response.amendments_included,
        has_more_context=response.has_more_context,
        total_matches=response.total_matches,
        degraded=response.degraded,
        degradation_reason=response.degradation_reason,
    )


@app.get("/documents/{document_id}/sections/{section_number}")
async def get_section_endpoint(document_id: str, section_number: str):
    """Get a section and its descendants."""
    store = get_store()
    result = get_section(store, document_id, section_number)
    if result is None:
        raise HTTPException(status_code=404, detail="Section not found")
    return result


@app.get("/documents/{document_id}/clauses/{number}")
async def get_clause_endpoint(document_id: str, number: str):
    """Get a clause with children, footnotes, and amendment links."""
    store = get_store()
    result = get_clause(store, document_id, number)
    if result is None:
        raise HTTPException(status_code=404, detail="Clause not found")
    return result


@app.get("/documents/{document_id}/equations/{number}")
async def get_equation_endpoint(document_id: str, number: str):
    """Get an equation."""
    store = get_store()
    result = get_equation(store, document_id, number)
    if result is None:
        raise HTTPException(status_code=404, detail="Equation not found")
    return result


@app.get("/documents/{document_id}/tables/{number}")
async def get_table_endpoint(document_id: str, number: str):
    """Get a table with HTML, header context, footnotes."""
    store = get_store()
    result = get_table(store, document_id, number)
    if result is None:
        raise HTTPException(status_code=404, detail="Table not found")
    return result


@app.get("/documents/{document_id}/figures/{number}")
async def get_figure_endpoint(document_id: str, number: str):
    """Get a figure with asset info."""
    store = get_store()
    result = get_figure(store, document_id, number)
    if result is None:
        raise HTTPException(status_code=404, detail="Figure not found")
    return result


@app.get("/documents/{document_id}/pages/{pdf_page}")
async def get_page_endpoint(document_id: str, pdf_page: int):
    """Get all items on a PDF page."""
    store = get_store()
    results = get_page(store, document_id, pdf_page)
    return results


@app.get("/items/{item_id}/related")
async def get_related_endpoint(item_id: str):
    """Get related items: cross-refs, parent, footnotes, amendment links."""
    store = get_store()
    result = get_related(store, item_id)
    if result is None:
        raise HTTPException(status_code=404, detail="Item not found")
    return result


@app.get("/assets/{asset_id}")
async def get_asset(asset_id: str):
    """Serve an asset (image) by ID."""
    store = get_store()
    asset = store.get_asset(asset_id)
    if asset is None:
        raise HTTPException(status_code=404, detail="Asset not found")

    asset_path = Path(asset.file_path)
    if not asset_path.exists():
        raise HTTPException(status_code=404, detail="Asset file not found")

    with open(asset_path, "rb") as f:
        content = f.read()

    return Response(content=content, media_type=asset.mime_type)


@app.post("/index/build")
async def build_index(
    document_id: Optional[str] = Query(None, description="Optional document ID to limit build"),
):
    """Build vector index for the current embedder."""
    store = get_store()
    embedder = get_embedder_dep()

    from .index import VectorIndex

    index = VectorIndex(_settings.db_path)
    result = index.build_vectors(store, embedder, document_id=document_id)
    return result


if __name__ == "__main__":
    import uvicorn

    settings = get_settings()
    uvicorn.run(app, host=settings.host, port=settings.api_port)