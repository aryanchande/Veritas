"""FastAPI app: upload PDFs, ask verified questions, scan for conflicts, view highlighted pages."""
from __future__ import annotations

import shutil
import threading
from datetime import date, datetime, timezone
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import config, ingest, llm, qa, reasoning
from .store import Store

config.PDF_DIR.mkdir(parents=True, exist_ok=True)
store = Store(config.INDEX_DIR)
app = FastAPI(title="VeriDoc - Zero-Guessing Document AI")
_busy = threading.Lock()                                   # one heavy LLM job at a time
JOB = {"running": False, "done": 0, "total": 0, "error": None}


class AskBody(BaseModel):
    question: str
    as_of: str | None = None
    doc_ids: list[str] | None = None
    check_conflicts: bool = True


class DocPatch(BaseModel):
    doc_date: str | None = None
    effective_date: str | None = None
    amends: list[str] | None = None


def _public(d: dict) -> dict:
    return {**d, "status": reasoning.status_of(d, date.today().isoformat())}


@app.get("/api/health")
def health():
    h = llm.health()
    h.update({"documents": len(store.docs), "chunks": len(store.chunks)})
    return h


@app.get("/api/documents")
def documents():
    return sorted((_public(d) for d in store.docs.values()), key=lambda d: reasoning.eff_date(d))


@app.post("/api/documents")
def upload(files: list[UploadFile] = File(...)):
    results = []
    with _busy:
        for f in files:
            if not f.filename.lower().endswith(".pdf"):
                results.append({"name": f.filename, "error": "Only PDF files are supported"})
                continue
            tmp = config.PDF_DIR / f"_tmp_{f.filename}"
            with tmp.open("wb") as out:
                shutil.copyfileobj(f.file, out)
            try:
                doc, new = ingest.ingest_pdf(tmp, store, llm.embed, display_name=Path(f.filename).name)
                if new:
                    tmp.replace(config.PDF_DIR / f"{doc['id']}.pdf")
                results.append({"name": f.filename, "id": doc["id"], "new": new, "pages": doc["pages"], "ocr_pages": doc["ocr_pages"]})
            except Exception as e:                           # surface parse/Ollama errors to the UI
                results.append({"name": f.filename, "error": str(e)})
            finally:
                tmp.unlink(missing_ok=True)
        if any(r.get("new") for r in results):
            try:
                reasoning.link_amendments(store)
            except Exception as e:
                results.append({"name": "(amendment linking)", "error": str(e)})
            store.scan = {"items": [], "finished": None}
            store.save()
    return results


@app.patch("/api/documents/{doc_id}")
def patch_doc(doc_id: str, body: DocPatch):
    d = store.docs.get(doc_id)
    if not d:
        raise HTTPException(404, "unknown document")
    if body.doc_date is not None:
        d["doc_date"] = body.doc_date or None
    if body.effective_date is not None:
        d["effective_date"] = body.effective_date or None
    if body.amends is not None:
        d["amends"] = [a for a in body.amends if a in store.docs and a != doc_id]
        d["amends_locked"] = True
    store.save()
    return _public(d)


@app.delete("/api/documents/{doc_id}")
def delete_doc(doc_id: str):
    if doc_id not in store.docs:
        raise HTTPException(404, "unknown document")
    store.remove_document(doc_id)
    (config.PDF_DIR / f"{doc_id}.pdf").unlink(missing_ok=True)
    return {"ok": True}


@app.get("/api/documents/{doc_id}/pdf")
def get_pdf(doc_id: str):
    p = config.PDF_DIR / f"{doc_id}.pdf"
    if not p.exists():
        raise HTTPException(404, "file missing")
    return FileResponse(p, media_type="application/pdf")


@app.get("/api/documents/{doc_id}/page/{page}.png")
def page_png(doc_id: str, page: int, quote: str | None = None):
    from .highlight import render_page
    p = config.PDF_DIR / f"{doc_id}.pdf"
    if not p.exists():
        raise HTTPException(404, "file missing")
    try:
        return Response(render_page(p, page, quote), media_type="image/png")
    except IndexError:
        raise HTTPException(404, "page out of range")


@app.post("/api/ask")
def ask(body: AskBody):
    if not body.question.strip():
        raise HTTPException(400, "empty question")
    try:
        with _busy:
            return qa.ask(store, body.question.strip(), body.as_of, body.doc_ids, body.check_conflicts)
    except llm.LLMError as e:
        raise HTTPException(503, str(e))


@app.get("/api/chain")
def chain():
    today = date.today().isoformat()
    return reasoning.chain_view(store, list(store.docs), today)


def _scan_job():
    try:
        with _busy:
            def prog(done, total):
                JOB.update(done=done, total=total)
            items = reasoning.scan_all(store, progress=prog)
            store.scan = {"items": items, "finished": datetime.now(timezone.utc).isoformat()}
            store.save()
    except Exception as e:
        JOB["error"] = str(e)
    finally:
        JOB["running"] = False


@app.post("/api/scan")
def start_scan():
    if JOB["running"]:
        return JOB
    JOB.update(running=True, done=0, total=0, error=None)
    threading.Thread(target=_scan_job, daemon=True).start()
    return JOB


@app.get("/api/scan")
def scan_status():
    return {"job": JOB, "result": store.scan}


@app.get("/")
def index():
    return FileResponse(config.FRONTEND_DIR / "index.html")


app.mount("/static", StaticFiles(directory=config.FRONTEND_DIR), name="static")
