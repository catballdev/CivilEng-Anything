"""
Retrieval core for CivilEng-Anything.

Single implementation shared by CLI, FastAPI, and MCP interfaces.
"""

from __future__ import annotations

import json
import logging
import math
import re
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

from .config import Settings
from .schema import (
    SearchResult,
    SearchMode,
    ItemType,
    ContentOrigin,
    ItemStatus,
    DocumentMeta,
    ContentItem,
)
from .store import Store
from .index import VectorIndex
from .embeddings import Embedder, get_embedder, HashingEmbedder

logger = logging.getLogger(__name__)

# Token estimation: ~4 chars per token
CHARS_PER_TOKEN = 4


@dataclass
class SearchResponse:
    """Response from search with metadata."""
    results: list[SearchResult]
    mode: SearchMode
    baseline_editions: list[str]
    amendments_included: bool
    has_more_context: bool
    total_matches: int
    degraded: bool = False
    degradation_reason: Optional[str] = None


def list_documents(store: Store) -> list[DocumentMeta]:
    """List all documents."""
    return store.list_documents()


def get_document_manifest(store: Store, document_id: str) -> Optional[dict]:
    """Get document metadata with page-status counts and amendment relations."""
    doc = store.get_document(document_id)
    if not doc:
        return None

    page_statuses = store.get_page_statuses(document_id)
    counts = {}
    for ps in page_statuses:
        counts[ps.status.value] = counts.get(ps.status.value, 0) + 1

    total = doc.total_pdf_pages
    done = counts.get("done", 0)
    failed = counts.get("failed", 0)
    retry = counts.get("retry_pending", 0)
    review = counts.get("needs_review", 0)
    pending = total - (done + failed + retry + review)

    # Get amendment relations (handle both dict and string formats)
    amendment_docs = []
    for am_rel in doc.amendment_relations:
        if isinstance(am_rel, dict):
            am_id = am_rel.get("document_id")
        else:
            am_id = am_rel
        if am_id:
            am_doc = store.get_document(am_id)
            if am_doc:
                amendment_docs.append({
                    "document_id": am_doc.document_id,
                    "edition_id": am_doc.edition_id,
                    "title": am_doc.title,
                })

    return {
        "document": doc,
        "page_counts": {
            "total": total,
            "done": done,
            "failed": failed,
            "retry_pending": retry,
            "needs_review": review,
            "pending": max(0, pending),
        },
        "amendments": amendment_docs,
    }


def _estimate_tokens(text: str) -> int:
    """Estimate token count from text length."""
    return max(1, math.ceil(len(text) / CHARS_PER_TOKEN))


def _trim_to_token_budget(
    results: list[SearchResult],
    token_budget: int,
) -> tuple[list[SearchResult], bool]:
    """
    Trim results to token budget.

    Keeps footnotes attached to kept items. Sets has_more_context on last kept item.
    """
    if token_budget <= 0:
        return results, False

    kept = []
    used_tokens = 0
    has_more = False

    for i, result in enumerate(results):
        # Estimate tokens for this result (use snippet as proxy)
        item_text = result.snippet or ""
        result_tokens = _estimate_tokens(item_text)

        if used_tokens + result_tokens <= token_budget or not kept:
            kept.append(result)
            used_tokens += result_tokens
        else:
            has_more = True
            break

    # Mark last kept result
    if kept and has_more:
        kept[-1].has_more_context = True

    return kept, has_more


def _reciprocal_rank_fusion(
    keyword_results: list[SearchResult],
    semantic_results: list[SearchResult],
    top_k: int,
) -> list[SearchResult]:
    """
    Merge keyword and semantic results using Reciprocal Rank Fusion.

    RRF score = sum(1 / (k + rank)) for each list where item appears.
    k is typically 60.
    """
    k = 60
    scores = {}
    items = {}

    # Process keyword results
    for rank, result in enumerate(keyword_results):
        item_id = result.item_id
        scores[item_id] = scores.get(item_id, 0) + 1.0 / (k + rank + 1)
        items[item_id] = result

    # Process semantic results
    for rank, result in enumerate(semantic_results):
        item_id = result.item_id
        scores[item_id] = scores.get(item_id, 0) + 1.0 / (k + rank + 1)
        if item_id not in items:
            items[item_id] = result

    # Sort by RRF score
    fused = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    # Build results preserving original SearchResult objects
    merged = []
    for item_id, score in fused[:top_k]:
        result = items[item_id]
        # Update score to RRF score
        result.score = score
        merged.append(result)

    return merged


def _exact_search(
    store: Store,
    query: str,
    document_id: Optional[str],
    edition_id: Optional[str],
    jurisdiction: Optional[str],
    doc_type: Optional[str],
    top_k: int,
) -> list[SearchResult]:
    """
    Exact search by clause/figure/table/equation number.

    Supports partial matches like "9.2" matching "9.2.1", "9.2.2", etc.
    """
    # Try to extract a number pattern from query
    number_match = re.search(r"\b(\d+(?:\.\d+)*)\b", query)
    if not number_match:
        return []

    number = number_match.group(1)

    # Search for items with matching number across all types
    conn = store._get_conn()
    
    # Build base query
    sql = "SELECT * FROM items WHERE 1=1"
    params = []
    
    if document_id:
        sql += " AND document_id = ?"
        params.append(document_id)
    if edition_id:
        sql += " AND edition_id = ?"
        params.append(edition_id)
    
    # Exact match on number
    sql += " AND number = ?"
    params.append(number)
    sql += " ORDER BY pdf_page, item_id"
    
    rows = conn.execute(sql, params).fetchall()
    items = [store._row_to_item(row) for row in rows]

    # Also search for prefix matches (e.g., "9.2" matches "9.2.1")
    if "." in number:
        sql2 = "SELECT * FROM items WHERE 1=1"
        params2 = []
        if document_id:
            sql2 += " AND document_id = ?"
            params2.append(document_id)
        if edition_id:
            sql2 += " AND edition_id = ?"
            params2.append(edition_id)
        sql2 += " AND number LIKE ?"
        params2.append(number + ".%")
        sql2 += " ORDER BY pdf_page, item_id"
        
        rows2 = conn.execute(sql2, params2).fetchall()
        prefix_items = [store._row_to_item(row) for row in rows2]
        
        # Merge, avoiding duplicates
        seen = {item.item_id for item in items}
        for item in prefix_items:
            if item.item_id not in seen:
                items.append(item)
                seen.add(item.item_id)

    # Apply jurisdiction/doc_type filters
    if jurisdiction or doc_type:
        filtered = []
        for item in items:
            doc = store.get_document(item.document_id)
            if doc:
                if jurisdiction and doc.jurisdiction != jurisdiction:
                    continue
                if doc_type and doc.doc_type.value != doc_type:
                    continue
                filtered.append(item)
        items = filtered

    # Convert to SearchResult
    results = []
    for item in items[:top_k]:
        doc = store.get_document(item.document_id)
        if doc:
            # Extract amendment item IDs for related field
            related = []
            for link in item.amendment_links:
                if isinstance(link, dict) and "item_id" in link:
                    related.append(link["item_id"])
                elif isinstance(link, str):
                    related.append(link)
            
            results.append(SearchResult(
                item_id=item.item_id,
                document_id=item.document_id,
                edition_id=item.edition_id,
                doc_title=doc.title,
                number=item.number,
                section_path=item.section_path,
                snippet=item.text[:200],
                pdf_page=item.pdf_page,
                printed_page=item.printed_page,
                content_origin=item.content_origin,
                status=item.status,
                score=1.0,  # Exact matches get max score
                has_more_context=False,
                related=related,
            ))

    return results


def _keyword_search(
    store: Store,
    query: str,
    document_id: Optional[str],
    edition_id: Optional[str],
    jurisdiction: Optional[str],
    doc_type: Optional[str],
    top_k: int,
) -> list[SearchResult]:
    """Full-text search using SQLite FTS5."""
    # Use the store's FTS search (filtering for jurisdiction/doc_type done post-search)
    fts_results = store.fts_search(
        query=query,
        document_id=document_id,
        edition_id=edition_id,
        limit=top_k * 2,  # Get more to allow post-filtering
    )

    results = []
    for result in fts_results:
        # Apply jurisdiction/doc_type filters
        if jurisdiction or doc_type:
            doc = store.get_document(result.document_id)
            if doc:
                if jurisdiction and doc.jurisdiction != jurisdiction:
                    continue
                if doc_type and doc.doc_type.value != doc_type:
                    continue
        # Create new SearchResult with match_type
        result_dict = result.model_dump()
        result_dict["match_type"] = "keyword"
        results.append(SearchResult(**result_dict))
        if len(results) >= top_k:
            break

    return results


def _semantic_search(
    store: Store,
    index: VectorIndex,
    embedder: Embedder,
    query: str,
    document_id: Optional[str],
    edition_id: Optional[str],
    jurisdiction: Optional[str],
    doc_type: Optional[str],
    top_k: int,
) -> tuple[list[SearchResult], bool]:
    """
    Semantic search using vector index.

    Returns (results, available) where available=False means index not built.
    """
    if not index.has_vectors(embedder.name):
        return [], False

    vector_results = index.vector_search(
        store=store,
        embedder=embedder,
        query=query,
        top_k=top_k,
        document_id=document_id,
        edition_id=edition_id,
        jurisdiction=jurisdiction,
        doc_type=doc_type,
    )

    results = []
    for vr in vector_results:
        item = store.get_item(vr["item_id"])
        if item:
            doc = store.get_document(item.document_id)
            if doc:
                # Extract amendment item IDs for related field
                related = []
                for link in item.amendment_links:
                    if isinstance(link, dict) and "item_id" in link:
                        related.append(link["item_id"])
                    elif isinstance(link, str):
                        related.append(link)
                
                results.append(SearchResult(
                    item_id=item.item_id,
                    document_id=item.document_id,
                    edition_id=item.edition_id,
                    doc_title=doc.title,
                    number=item.number,
                    section_path=item.section_path,
                    snippet=item.text[:200],
                    pdf_page=item.pdf_page,
                    printed_page=item.printed_page,
                    content_origin=item.content_origin,
                    status=item.status,
                    score=vr["score"],
                    has_more_context=False,
                    related=related,
                    match_type="semantic",
                ))

    return results, True


def search(
    store: Store,
    query: str,
    *,
    document_id: Optional[str] = None,
    edition_id: Optional[str] = None,
    jurisdiction: Optional[str] = None,
    doc_type: Optional[str] = None,
    mode: str = "hybrid",
    top_k: int = 8,
    token_budget: int = 6000,
    expand: bool = False,
    embedder: Optional[Embedder] = None,
    settings: Optional[Settings] = None,
) -> SearchResponse:
    """
    Unified search across exact, keyword, semantic, and hybrid modes.

    Args:
        store: Store instance
        query: Search query
        document_id: Filter by document
        edition_id: Filter by edition
        jurisdiction: Filter by jurisdiction
        doc_type: Filter by document type
        mode: Search mode (exact, keyword, semantic, hybrid)
        top_k: Maximum results
        token_budget: Token budget for results
        expand: Whether to expand results with context
        embedder: Embedder to use (if None, will try to get from settings)
        settings: Settings for embedder creation

    Returns:
        SearchResponse with results and metadata
    """
    # Determine embedder
    if embedder is None and settings is not None:
        try:
            embedder = get_embedder(settings)
        except RuntimeError:
            embedder = HashingEmbedder()  # Fallback for tests

    if embedder is None:
        embedder = HashingEmbedder()

    index = VectorIndex(store.db_path)

    mode_enum = SearchMode(mode)
    degraded = False
    degradation_reason = None

    if mode_enum == SearchMode.EXACT:
        results = _exact_search(store, query, document_id, edition_id, jurisdiction, doc_type, top_k)
        # Get baseline editions
        baseline_editions = _get_baseline_editions(store, results, document_id, edition_id)
        amendments_included = _check_amendments_included(store, results)

    elif mode_enum == SearchMode.KEYWORD:
        results = _keyword_search(store, query, document_id, edition_id, jurisdiction, doc_type, top_k)
        baseline_editions = _get_baseline_editions(store, results, document_id, edition_id)
        amendments_included = _check_amendments_included(store, results)

    elif mode_enum == SearchMode.SEMANTIC:
        results, available = _semantic_search(
            store, index, embedder, query, document_id, edition_id, jurisdiction, doc_type, top_k
        )
        if not available:
            degraded = True
            degradation_reason = "semantic index not built"
            results = []
        baseline_editions = _get_baseline_editions(store, results, document_id, edition_id)
        amendments_included = _check_amendments_included(store, results)

    elif mode_enum == SearchMode.HYBRID:
        keyword_results = _keyword_search(store, query, document_id, edition_id, jurisdiction, doc_type, top_k)
        semantic_results, available = _semantic_search(
            store, index, embedder, query, document_id, edition_id, jurisdiction, doc_type, top_k
        )

        if not available:
            degraded = True
            degradation_reason = "semantic index not built; falling back to keyword"
            results = keyword_results
        else:
            results = _reciprocal_rank_fusion(keyword_results, semantic_results, top_k)

        baseline_editions = _get_baseline_editions(store, results, document_id, edition_id)
        amendments_included = _check_amendments_included(store, results)

    else:
        raise ValueError(f"Unknown search mode: {mode}")

    # Trim to token budget
    trimmed_results, has_more = _trim_to_token_budget(results, token_budget)

    return SearchResponse(
        results=trimmed_results,
        mode=mode_enum,
        baseline_editions=baseline_editions,
        amendments_included=amendments_included,
        has_more_context=has_more,
        total_matches=len(results),
        degraded=degraded,
        degradation_reason=degradation_reason,
    )


def _get_baseline_editions(
    store: Store,
    results: list[SearchResult],
    document_id: Optional[str],
    edition_id: Optional[str],
) -> list[str]:
    """Get list of baseline editions in results."""
    editions = set()
    if edition_id:
        editions.add(edition_id)
    elif document_id:
        doc = store.get_document(document_id)
        if doc:
            editions.add(doc.edition_id)
    else:
        for result in results:
            # Use edition_id from the result
            if result.edition_id:
                editions.add(result.edition_id)
    return sorted(editions)


def _check_amendments_included(store: Store, results: list[SearchResult]) -> bool:
    """Check if any amendment documents are in results or linked."""
    for result in results:
        # Check if result is from an amendment document
        doc = store.get_document(result.document_id)
        if doc and doc.doc_type.value == "amendment":
            return True
        # Check if this item has amendment links
        if result.related:
            return True
    return False


def get_section(
    store: Store,
    document_id: str,
    section_number: str,
    edition_id: Optional[str] = None,
) -> Optional[SearchResult]:
    """Get a section item plus its descendant items in order."""
    doc = store.get_document(document_id)
    if not doc:
        return None
    
    if edition_id is None:
        edition_id = doc.edition_id
    
    # Find the section item
    items = store.items_by_number(document_id, edition_id, ItemType.SECTION, section_number)
    if not items:
        # Try partial match
        items = store.items_by_number(document_id, edition_id, ItemType.SECTION, section_number)

    if not items:
        return None

    section_item = items[0]

    # Get descendant items (same document, section_path starts with this section's path)
    section_path = section_item.section_path
    descendants = store.get_items_by_section_path(document_id, section_path)

    # Extract amendment item IDs for related field
    related = []
    for link in section_item.amendment_links:
        if isinstance(link, dict) and "item_id" in link:
            related.append(link["item_id"])
        elif isinstance(link, str):
            related.append(link)

    return SearchResult(
        item_id=section_item.item_id,
        document_id=section_item.document_id,
        edition_id=section_item.edition_id,
        doc_title=doc.title,
        number=section_item.number,
        section_path=section_item.section_path,
        snippet=section_item.text[:200],
        pdf_page=section_item.pdf_page,
        printed_page=section_item.printed_page,
        content_origin=section_item.content_origin,
        status=section_item.status,
        score=1.0,
        match_type="exact",
        has_more_context=len(descendants) > 1,
        related=related,
    )


def get_clause(
    store: Store,
    document_id: str,
    number: str,
    edition_id: Optional[str] = None,
) -> Optional[SearchResult]:
    """Get exact clause + child paragraphs + footnotes + amendment links."""
    doc = store.get_document(document_id)
    if not doc:
        return None
    
    if edition_id is None:
        edition_id = doc.edition_id
    
    items = store.items_by_number(document_id, edition_id, ItemType.CLAUSE, number)
    if not items:
        return None

    clause_item = items[0]

    # Get child paragraphs (same section path, type=paragraph)
    children = store.get_items_by_section_path(document_id, clause_item.section_path)
    child_paragraphs = [c for c in children if c.type == ItemType.PARAGRAPH]

    # Get footnotes linked to this clause
    footnotes = []
    for fn_id in clause_item.footnote_ids:
        fn = store.get_item(fn_id)
        if fn:
            footnotes.append(fn)

    # Get amendment items targeting this clause
    amendment_items = []
    for am_link in clause_item.amendment_links:
        am_doc_id = am_link.get("document_id")
        am_item_id = am_link.get("item_id")
        if am_doc_id and am_item_id:
            am_item = store.get_item(am_item_id)
            if am_item:
                am_doc = store.get_document(am_doc_id)
                # Map status string to ItemStatus enum
                status_str = am_link.get("status", "pending_verification")
                if status_str == "verified":
                    status_str = "checked"
                am_item.status = ItemStatus(status_str)
                amendment_items.append({
                    "item": am_item,
                    "document": am_doc,
                    "link_status": am_link.get("status", "pending_verification"),
                })

    # Extract amendment item IDs for related field
    related = []
    for link in clause_item.amendment_links:
        if isinstance(link, dict) and "item_id" in link:
            related.append(link["item_id"])
        elif isinstance(link, str):
            related.append(link)

    return SearchResult(
        item_id=clause_item.item_id,
        document_id=clause_item.document_id,
        edition_id=clause_item.edition_id,
        doc_title=doc.title,
        number=clause_item.number,
        section_path=clause_item.section_path,
        snippet=clause_item.text[:200],
        pdf_page=clause_item.pdf_page,
        printed_page=clause_item.printed_page,
        content_origin=clause_item.content_origin,
        status=clause_item.status,
        score=1.0,
        match_type="exact",
        has_more_context=bool(child_paragraphs or footnotes or amendment_items),
        related=related,
    )


def get_equation(
    store: Store,
    document_id: str,
    number: str,
    edition_id: Optional[str] = None,
) -> Optional[SearchResult]:
    """Get exact equation."""
    doc = store.get_document(document_id)
    if not doc:
        return None
    
    if edition_id is None:
        edition_id = doc.edition_id
    
    items = store.items_by_number(document_id, edition_id, ItemType.EQUATION, number)
    if not items:
        return None
    item = items[0]
    
    # Extract amendment item IDs for related field
    related = []
    for link in item.amendment_links:
        if isinstance(link, dict) and "item_id" in link:
            related.append(link["item_id"])
        elif isinstance(link, str):
            related.append(link)
    
    return SearchResult(
        item_id=item.item_id,
        document_id=item.document_id,
        edition_id=item.edition_id,
        doc_title=doc.title,
        number=item.number,
        section_path=item.section_path,
        snippet=item.text[:200],
        pdf_page=item.pdf_page,
        printed_page=item.printed_page,
        content_origin=item.content_origin,
        status=item.status,
        score=1.0,
        match_type="exact",
        related=related,
    )


def get_table(
    store: Store,
    document_id: str,
    number: str,
    edition_id: Optional[str] = None,
) -> Optional[SearchResult]:
    """Get exact table with HTML, header context, footnotes."""
    doc = store.get_document(document_id)
    if not doc:
        return None
    
    if edition_id is None:
        edition_id = doc.edition_id
    
    items = store.items_by_number(document_id, edition_id, ItemType.TABLE, number)
    if not items:
        return None
    item = items[0]
    doc = store.get_document(document_id)

    # Get footnotes for table
    footnotes = []
    for fn_id in item.footnote_ids:
        fn = store.get_item(fn_id)
        if fn:
            footnotes.append(fn)

    # Extract amendment item IDs for related field
    related = []
    for link in item.amendment_links:
        if isinstance(link, dict) and "item_id" in link:
            related.append(link["item_id"])
        elif isinstance(link, str):
            related.append(link)

    return SearchResult(
        item_id=item.item_id,
        document_id=item.document_id,
        edition_id=item.edition_id,
        doc_title=doc.title,
        number=item.number,
        section_path=item.section_path,
        snippet=item.text[:200],
        pdf_page=item.pdf_page,
        printed_page=item.printed_page,
        content_origin=item.content_origin,
        status=item.status,
        score=1.0,
        match_type="exact",
        has_more_context=bool(footnotes),
        related=related,
    )


def get_figure(
    store: Store,
    document_id: str,
    number: str,
    edition_id: Optional[str] = None,
) -> Optional[SearchResult]:
    """Get exact figure with asset IDs and file info."""
    doc = store.get_document(document_id)
    if not doc:
        return None
    
    if edition_id is None:
        edition_id = doc.edition_id
    
    items = store.items_by_number(document_id, edition_id, ItemType.FIGURE, number)
    if not items:
        return None
    item = items[0]
    doc = store.get_document(document_id)

    # Extract amendment item IDs for related field
    related = []
    for link in item.amendment_links:
        if isinstance(link, dict) and "item_id" in link:
            related.append(link["item_id"])
        elif isinstance(link, str):
            related.append(link)

    return SearchResult(
        item_id=item.item_id,
        document_id=item.document_id,
        edition_id=item.edition_id,
        doc_title=doc.title,
        number=item.number,
        section_path=item.section_path,
        snippet=item.text[:200],
        pdf_page=item.pdf_page,
        printed_page=item.printed_page,
        content_origin=item.content_origin,
        status=item.status,
        score=1.0,
        match_type="exact",
        has_more_context=bool(item.asset_ids),
        related=related,
    )


def get_page(
    store: Store,
    document_id: str,
    pdf_page: int,
) -> list[SearchResult]:
    """Get all items on a PDF page in stored order."""
    items = store.get_items_by_page(document_id, pdf_page)
    results = []
    for item in items:
        doc = store.get_document(document_id)
        if doc:
            # Extract amendment item IDs for related field
            related = []
            for link in item.amendment_links:
                if isinstance(link, dict) and "item_id" in link:
                    related.append(link["item_id"])
                elif isinstance(link, str):
                    related.append(link)
            
            results.append(SearchResult(
                item_id=item.item_id,
                document_id=item.document_id,
                edition_id=item.edition_id,
                doc_title=doc.title,
                number=item.number,
                section_path=item.section_path,
                snippet=item.text[:200],
                pdf_page=item.pdf_page,
                printed_page=item.printed_page,
                content_origin=item.content_origin,
                status=item.status,
                score=1.0,
                has_more_context=False,
                related=related,
            ))
    return results


def get_related(
    store: Store,
    item_id: str,
) -> Optional[SearchResult]:
    """Get cross-refs resolved to items, parent section, footnotes, amendment links."""
    item = store.get_item(item_id)
    if not item:
        return None

    doc = store.get_document(item.document_id)

    # Resolve cross-refs
    cross_ref_items = []
    for ref in item.cross_refs:
        # Parse "Type Number" format
        match = re.match(r"(\w+)\s+(\S+)", ref)
        if match:
            ref_type = match.group(1).lower()
            ref_number = match.group(2)
            type_map = {
                "clause": ItemType.CLAUSE,
                "section": ItemType.SECTION,
                "table": ItemType.TABLE,
                "figure": ItemType.FIGURE,
                "equation": ItemType.EQUATION,
            }
            itype = type_map.get(ref_type)
            if itype:
                refs = store.items_by_number(item.document_id, item.edition_id, itype, ref_number)
                cross_ref_items.extend(refs)

    # Parent section
    parent_item = None
    if item.parent_id:
        parent_item = store.get_item(item.parent_id)

    # Footnotes
    footnote_items = []
    for fn_id in item.footnote_ids:
        fn = store.get_item(fn_id)
        if fn:
            footnote_items.append(fn)

    # Amendment links
    amendment_items = []
    for am_link in item.amendment_links:
        am_doc_id = am_link.get("document_id")
        am_item_id = am_link.get("item_id")
        if am_doc_id and am_item_id:
            am_item = store.get_item(am_item_id)
            if am_item:
                am_doc = store.get_document(am_doc_id)
                amendment_items.append({
                    "item": am_item,
                    "document": am_doc,
                    "link_status": am_link.get("status", "pending_verification"),
                })

    # Extract amendment item IDs for related field
    related = []
    for link in item.amendment_links:
        if isinstance(link, dict) and "item_id" in link:
            related.append(link["item_id"])
        elif isinstance(link, str):
            related.append(link)
    
    return SearchResult(
        item_id=item.item_id,
        document_id=item.document_id,
        edition_id=item.edition_id,
        doc_title=doc.title if doc else "",
        number=item.number,
        section_path=item.section_path,
        snippet=item.text[:200],
        pdf_page=item.pdf_page,
        printed_page=item.printed_page,
        content_origin=item.content_origin,
        status=item.status,
        score=1.0,
        has_more_context=bool(cross_ref_items or parent_item or footnote_items or amendment_items),
        related=related,
    )