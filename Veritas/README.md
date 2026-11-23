# VeriDoc: Zero-Guessing & Contradiction-Checking Document AI

A web app that answers questions across several PDFs **only with verbatim proof**, flags conflicting clauses, works out
which amendment applies today, and says *"I don't have proof"* instead of guessing. The whole pipeline runs on
open-source models served by Ollama. No paid API is called anywhere.

Built on the StudyBuddy RAG project (PyMuPDF -> chunks -> embeddings -> Ollama), but with page-aware chunks, hybrid
retrieval, a verification layer, contradiction and amendment reasoning, and a FastAPI + browser UI.

## Quick start

```bash
# 1. On the machine that runs Ollama (e.g. the MacBook Air)
ollama pull qwen3.5:9b        # 16 GB RAM.  Use qwen3.5:4b on an 8 GB machine
ollama pull bge-m3            # multilingual embeddings (Hindi + English)
OLLAMA_HOST=0.0.0.0 ollama serve        # only needed if the web app runs on a different computer

# 2. On the machine that runs the web app
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
export OLLAMA_HOST=http://<ollama-machine-ip>:11434    # skip if same machine; Windows: set OLLAMA_HOST=...
export LLM_MODEL=qwen3.5:9b                            # or qwen3.5:4b
./run.sh                                               # or: uvicorn backend.main:app --port 8000
# open http://localhost:8000  (the badge top-right shows whether Ollama + both models are reachable)
```

Demo PDFs are in `demo_pdfs/` (regenerate with `python scripts/make_demo_pdfs.py`). Offline logic tests:
`python tests/test_logic.py`.

## Architecture

```
PDFs -> PyMuPDF (text blocks, tables, OCR fallback) -> page-aware chunks -> bge-m3 embeddings + BM25
                                                                              |
question -> embed -> hybrid search (dense + BM25, RRF) + neighbouring chunks --+
        -> Qwen3.5 drafts CLAIMS, each with {chunk, verbatim quote}   (JSON-schema constrained, temperature 0)
        -> VERIFY (code):  quote must exist in the cited chunk (token-level fuzzy match, snapped to real text)
                           every figure in the claim must appear in its quote
        -> VERIFY (model): strict judge: supported / partial / unsupported / contradicted, fails closed
        -> blocked claims are never shown; faithfulness = verified / drafted sentences
        -> CONTRADICTIONS: Qwen proposes conflicting pairs, code re-checks the quotes, then classifies:
              same chain + later effective date  -> "superseded by amendment" (shows who prevails)
              amendment effective after as-of    -> "not yet in force"
              unlinked documents / same document -> "contradiction" (neither is picked)
        -> status: verified | partial | conflict | abstained
```

## How each requirement is met

| Requirement | Where |
|---|---|
| Multi-document retrieval incl. tables and multi-page sections | `ingest.py` (tables via `find_tables`, kept as `a \| b` rows), `store.py` hybrid search, neighbour-chunk expansion in `qa.build_evidence` |
| Exact citations (doc, page, verbatim quote) | `textutil.find_quote` snaps every quote to the real text; UI chips open the page with the passage highlighted |
| Contradiction detection, side by side | `reasoning.find_conflicts`, "Conflicts & amendments" tab and inline in answers |
| Amendment chains and "what applies today" | `reasoning.link_amendments`, `classify`, `chain_view`; as-of date input; future-dated amendments flagged |
| Zero-guessing abstention | pre-LLM similarity gate, quote check, number guard, judge; fails closed if the verifier is down |
| Faithfulness score | `qa.ask` -> `faithfulness` (verified claims / drafted claims, as a percentage) |
| Bonus: highlight on PDF page | `highlight.py` |
| Bonus: Hindi | bge-m3 retrieval + Qwen3.5; the prompt answers in the question's language, quotes stay verbatim |
| Bonus: calculations | model proposes an expression over **quoted** inputs; Python evaluates it (`safe_eval`); inputs not found in a quote are rejected |

## Models

| Role | Model | Notes |
|---|---|---|
| Answer, claim extraction, judge, conflict finder, amendment linker | `qwen3.5:9b` (or `:4b`) via Ollama | `think:false` + JSON schema for fast, well-formed output |
| Embeddings | `bge-m3` via Ollama | multilingual (Hindi), dense retrieval |
| Lexical retrieval | BM25 (own implementation) | catches clause numbers and exact amounts |
| OCR (optional) | Tesseract | only for pages with no text layer |

## Demo script (about 4 minutes)

1. Upload the five PDFs in `demo_pdfs/`. The sidebar shows dates, "amends ..." links, and Amendment 3 as *not yet in force*.
2. *"What is the late payment charge that applies today?"* -> 2.5% from Amendment 2, with the chain 2% -> 3% -> 2.5%.
3. *"What is the prepayment charge?"* -> 2% today; Amendment 3 (0%) shown as not yet in force.
4. *"What is the processing fee for a loan of Rs. 4,00,000?"* -> 1.5% from the fee table, and a **contradiction** with the brochure (0.5%).
5. *"What is the foreclosure penalty for a gold loan?"* -> abstains, lists the missing evidence.
6. *"If an instalment of Rs. 20,000 is overdue for one month, what is the late charge?"* -> calculation with quoted inputs.
7. Click any citation chip to show the highlighted PDF page. Open the "Conflicts & amendments" tab and run a full scan.
   The Loan Agreement also contradicts itself (grace period 7 days on page 2 vs 5 days in Schedule A).

## Tuning knobs (environment variables, see `backend/config.py`)

`MIN_SIM` (abstain threshold; raise it if unrelated questions get through), `QUOTE_MIN_RATIO`, `TOP_K`, `NUM_CTX`,
`SCAN_SIM`, `SCAN_MAX_GROUPS`, `LLM_MODEL`, `EMBED_MODEL`, `OLLAMA_HOST`.

## Known limits (say these before the judges ask)

- Retrieval can miss a passage; the system then abstains or answers partially. It never fills the gap.
- Contradiction finding depends on the model noticing the pair; the global scan covers similar chunks, not every pair.
- Amendment links come from the model and dates parsed by regex. Wrong links or dates can be fixed with **Edit** on a document.
- Scanned pages are only as good as Tesseract. Handwriting is not supported.
- A 9B model on a laptop is slow: expect 1-3 minutes per question with contradiction checking on, less with it off.
