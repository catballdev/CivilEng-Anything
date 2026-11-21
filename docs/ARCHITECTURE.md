# CivilEng-Anything — architecture (v1)

## Purpose

A traceable knowledge base for civil/structural engineering documents (codes of
practice, design manuals, textbooks, amendments). Documents are parsed **once**
into a canonical, versioned format with full provenance; every AI client
(MCP / HTTP / CLI) then retrieves small, verifiable excerpts — original text,
clause/table/figure/equation numbers, PDF page and printed page — instead of
re-processing whole PDFs.

## Layers

```
PDF ──► ingest pipeline ──► canonical store (system of record)
                              ├─ SQLite DB (documents, items, pages, links)
                              ├─ JSONL export of every item (portable)
                              └─ assets/ (extracted figures, page renders)
                                   │
              derived indexes (rebuildable, never the source of truth)
                              ├─ exact indexes (clause/figure/table/equation no.)
                              ├─ SQLite FTS5 (full-text / BM25, symbol-safe)
                              ├─ embedding vectors (model-fingerprinted)
                              └─ LightRAG graph/semantic index (optional, one-way feed)
                                   │
              retrieval core (one implementation, one result schema)
                                   │
                    CLI ── FastAPI HTTP ── MCP (stdio + Streamable HTTP)
```

## Canonical format (schema `civileng-canonical/1`)

Document metadata: `document_id`, `document_family_id`, `edition_id`, official
title, publisher, jurisdiction, `doc_type` (government_code | amendment |
guideline | textbook | other), language, publication/revision date (or null),
source URL, retrieved_at, original SHA-256, total PDF pages, parser name +
version + config + parse run id, completeness, QC status, known amendment and
version relations.

Content item types: `section`, `clause`, `paragraph`, `equation`, `table`,
`figure`, `footnote`, `cross_reference`, `amendment`.

Every item carries: stable `item_id` (document-scoped, version-safe), parent id,
section path, clause/figure/table/equation number (or null), original text,
**PDF page (1-based in all external APIs)** and printed page (separate field,
null until verified), bounding box when available, asset ids, cross-references,
footnote links, amendment links, and a verification status
(`parsed | checked | pending_verification | failed`).

Original text, OCR output and AI-derived content are stored in separate fields
(`content_origin: original | ocr | derived`); derived content records the model
and generation config. Unknown values are null — never invented.

## Ingest pipeline (resumable, tracked)

`register → store original (SHA-256, write-once) → inspect PDF (pages, text /
scanned / mixed) → parse (MinerU, explicit full page range; Docling fallback
per failed page, with the choice and reason recorded) → archive raw parser
output → normalize to canonical → per-page QC → chunk & link (cross-refs,
footnotes, amendment targets) → build indexes → publish dataset version`.

- Per-page status table: `done | failed | retry_pending | needs_review`.
  Partial success is never labelled "complete".
- Re-importing identical content (same SHA-256) reuses stored results — no
  duplicate documents or items.
- Amendment → clause mapping is only recorded when it is stated in the
  amendment document itself or verified against the parsed baseline; otherwise
  the link status is `pending_verification`.

## Retrieval

Three complementary modes over the same core:

1. **Exact** — by document/edition + clause, figure, table or equation number.
2. **Full-text / BM25** — SQLite FTS5; engineering symbols and identifiers
   (e.g. `fcu`, `b_w`, `9.2.1`) are kept searchable (no destructive stemming).
3. **Semantic** — multilingual embeddings (default BGE-M3, replaceable);
   Traditional Chinese / Cantonese / English queries against English documents.

Results always show the baseline edition and whether registered amendments are
included; filters: document, edition, jurisdiction, doc_type. When a hit clause
has a known amendment, both the baseline text and the amendment source are
returned. Default behaviour is "search small, expand on demand"
(`top_k=8`, ~6,000-token text budget, both adjustable); truncated results
report a continuation cursor and a `has_more_context` flag instead of silently
dropping exceptions or footnotes. Table hits are returned with their header
rows, units and footnotes. Figure hits return a retrievable image asset
(MCP image content / HTTP resource), never a bare local path.

Pure retrieval returns evidence directly; no chat model is required. Graph
reasoning / answer generation (LightRAG) is an optional layer on top.

## Interfaces (shared core, shared result schema)

Tools: `list_documents`, `get_document_manifest`, `search`, `get_section`,
`get_clause`, `get_equation`, `get_table`, `get_figure`, `get_page`,
`get_related`. CLI mirrors them 1:1; the mapping table lives in the README.

## Storage, privacy, safety

- Everything persists under `data/` (originals, parsed output, canonical DB,
  assets, indexes) and survives restarts. `data/` is **not** committed to Git;
  only code, config templates, schemas, docs and the document *catalog*
  (metadata + hashes) are versioned.
- Local services bind to localhost by default.
- PDF content is data, never instructions: text extracted from documents must
  never be treated as commands for an agent.
- Cache keys include: original hash, parser + model version, parse config,
  embedding model + config, derivation model + prompt version. Vectors from
  different embedding models are never mixed; a model change builds a new
  index namespace.
