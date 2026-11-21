# Version compatibility — verified 2026-11-21

All facts below were checked against live PyPI metadata and the projects' official
repositories on 2026-11-21, before installation. Nothing here is assumed from a
README on a moving main branch.

## The constraint that decides the stack

`raganything==1.4.4` (latest on PyPI, checked 2026-11-21) declares, in its own
package metadata:

- `mineru[core] >=3.4.1, <4` (for Python < 3.14)
- `lightrag-hku <1.5`

MinerU's latest release is **4.2.3**, whose CLI and output schema differ from 3.x.
Mixing MinerU 4.x commands/output with RAG-Anything 1.4.4 is therefore **not**
supported. The known conflict the project brief warned about is real and current.

## Pinned combination (v1)

| Component | Pinned version | Source of truth |
|---|---|---|
| Python | 3.12.3 (system) | environment check |
| MinerU | `mineru[core]==3.4.5` (latest 3.x on PyPI) | PyPI release list |
| RAG-Anything | `raganything==1.4.4` | PyPI metadata (constraints above) |
| LightRAG | `lightrag-hku<1.5` → **resolved: 1.4.16** | PyPI (1.5.x exists but is excluded by RAG-Anything) |
| MCP Python SDK | `mcp` (official `modelcontextprotocol/python-sdk`) | PyPI |
| FastAPI / uvicorn | latest stable at lock time | PyPI |
| Docling | backup parser only; installed separately, not in the core lock | PyPI 2.151.1 at check time |
| Embedding (default) | BGE-M3 via sentence-transformers, CPU | replaceable via config |

Exact resolved versions are frozen in `requirements.lock` after installation.

## How the pieces are used (one import pipeline, no duplication)

- **MinerU 3.4.5 is the primary parser.** Its raw output is archived unchanged,
  then normalized into the CivilEng-Anything canonical format.
- **RAG-Anything 1.4.4** is used as the MinerU-integration reference and for its
  multimodal processing helpers where they fit the pinned versions. It is *not*
  run as a second, parallel import pipeline.
- **LightRAG (<1.5)** is a *derived* index fed one-way from canonical chunks
  (semantic/graph layer). It can be rebuilt from the canonical store at any
  time and is never the system of record. Exact clause lookup, full-text/BM25
  and provenance live in the CivilEng-Anything core and never depend on an LLM.
- **Docling** is a selective fallback for failed/low-quality pages only.

## MinerU page-range warning

MinerU processes the pages it is told to process; several wrappers and examples
default to a limited range. The import pipeline always passes an explicit full
page range (`-s 0 -e <last>`) and afterwards reconciles the per-page status
table against the PDF's real page count. A partial parse is never reported as
"book complete".

## Install gotchas found on this machine (2026-11-21, all reproduced)

- **pip + /tmp**: this VM's `/tmp` is a 512 MB tmpfs. pip stages big wheels
  there, so the 539 MB torch wheel died with `No space left on device`.
  Always run installs with `TMPDIR` pointing at a home-disk directory.
- **torch must be the CPU build**: install `torch` and `torchvision` from
  `https://download.pytorch.org/whl/cpu`. PyPI's plain `torchvision` wheel
  installed against `torch 2.15.0+cpu` fails at import with
  `operator torchvision::nms does not exist`, which also kills MinerU's
  layout model. Working pair: `torch 2.15.0+cpu` + `torchvision 0.29.2+cpu`.
- **MinerU + proxy env**: MinerU talks to its own local API service via httpx;
  the VM's inherited `NO_PROXY` (bracketed IPv6 entries) makes httpx raise
  `Invalid port: ':1]'`. The ingest adapter therefore launches MinerU with a
  sanitized `NO_PROXY=127.0.0.1,localhost`.

## Environment constraints recorded

- VM: 4 vCPU, ~7.7 GB RAM, no GPU, ~60 GB free disk. MinerU runs its CPU
  pipeline backend; parsing is batched by page ranges so memory stays bounded
  and an interrupted run can resume.
