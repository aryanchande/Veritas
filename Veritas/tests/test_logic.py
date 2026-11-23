"""Offline tests of the verification / amendment / conflict logic with a stubbed LLM.
Run:  python tests/test_logic.py      (needs numpy; pdftotext optional - falls back to inline text)"""
import hashlib
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from backend import config, ingest, llm, qa, reasoning  # noqa: E402
from backend.store import Store  # noqa: E402
from backend.textutil import find_quote, unsupported_numbers  # noqa: E402

DEMO = Path(__file__).resolve().parent.parent / "demo_pdfs"
AS_OF = "2026-11-22"


# ---------- stub embedding: hashed bag of words ----------
def fake_embed(texts, batch=16):
    out = np.zeros((len(texts), 512), dtype=np.float32)
    for i, t in enumerate(texts):
        for w in re.findall(r"\w+", t.lower()):
            out[i, int(hashlib.md5(w.encode()).hexdigest(), 16) % 512] += 1
    return out / np.clip(np.linalg.norm(out, axis=1, keepdims=True), 1e-9, None)


llm.embed = fake_embed


def pdf_pages(path):
    txt = subprocess.run(["pdftotext", "-layout", str(path), "-"], capture_output=True, text=True).stdout
    return [{"page": i + 1, "text": re.sub(r"[ \t]+\n", "\n", p), "tables": [], "method": "text"}
            for i, p in enumerate(txt.split("\f")) if p.strip()]


def load_demo(store):
    for f in sorted(DEMO.glob("*.pdf")):
        pages = pdf_pages(f)
        did = hashlib.sha1(f.name.encode()).hexdigest()[:10]
        chunks = ingest.make_chunks(did, f.name, pages)
        d, e = ingest.guess_dates("\n".join(p["text"] for p in pages[:2]))
        store.add_document({"id": did, "name": f.name, "sha": did, "pages": len(pages), "n_chunks": len(chunks), "ocr_pages": 0,
                            "doc_date": d, "effective_date": e, "amends": [], "amends_locked": False}, chunks,
                           fake_embed([c["section"] + "\n" + c["text"] for c in chunks]))
    return {d["name"]: d["id"] for d in store.docs.values()}


def check(name, cond):
    print(("PASS" if cond else "FAIL"), name)
    if not cond:
        check.failed += 1


check.failed = 0

# ---------- 1. quote matching ----------
text = "The late payment charge shall be 2.5% per month on the overdue amount. All other terms remain unchanged."
s, snap, span = find_quote("late payment charge shall be 2.5% per month on the overdue amount", text)
check("exact quote found verbatim", s == 1.0 and snap in text)
s, snap, _ = find_quote("The late payment charge shall be 2.5% per month on the overdue amounts", text)
check("slightly-off quote is snapped to real text", bool(snap) and snap in text)
s, snap, _ = find_quote("The late payment charge shall be 9% per month plus a penalty of Rs. 500", text)
check("fabricated quote rejected", snap == "")
check("number guard flags figure missing from quote", unsupported_numbers("The fee is 3% per month", ["fee is 2.5% per month"]) == ["3"])
check("number guard ignores small bare integers", unsupported_numbers("There are two conditions", ["conditions apply"]) == [])

# ---------- 2. dates ----------
check("dates: doc 15 Jun 2024 / effective 1 Jul 2024",
      ingest.guess_dates("Dated: 15 June 2024. Effective from: 1 July 2024. This Amendment") == ("2024-06-15", "2024-07-01"))

store = Store(Path(tempfile.mkdtemp()))
ids = load_demo(store)
A, A1, A2, A3, BRO = (ids[n] for n in ["Loan_Agreement_2023.pdf", "Amendment_1_2024.pdf", "Amendment_2_2025.pdf",
                                       "Amendment_3_2027.pdf", "Product_Brochure_2025.pdf"])
check("5 demo docs loaded", len(store.docs) == 5)
check("page numbers preserved (late payment on page 2)",
      any(c["page"] == 2 and "late payment charge of 2%" in c["text"] for c in store.doc_chunks(A)))
check("effective date of Amendment 3 parsed as 2027-04-01", store.docs[A3]["effective_date"] == "2027-04-01")

# ---------- 3. amendment linking (LLM down -> heuristic fallback) ----------
def boom(*a, **k):
    raise llm.LLMError("offline")


llm.chat_json = boom
reasoning.link_amendments(store)
check("heuristic: amendments link to the agreement", all(A in store.docs[x]["amends"] for x in (A1, A2, A3)))
check("heuristic: brochure does not amend anything", store.docs[BRO]["amends"] == [])

# ---------- 4. classification + chain ----------
c = reasoning.classify(store.docs, A, A1, AS_OF)
check("amendment 1 supersedes agreement", c["status"] == "superseded" and c["winner"] == "b")
c = reasoning.classify(store.docs, A, A3, AS_OF)
check("future amendment is 'pending', current doc wins", c["status"] == "amendment_pending" and c["winner"] == "a")
c = reasoning.classify(store.docs, A, BRO, AS_OF)
check("unlinked docs -> contradiction", c["status"] == "contradiction" and c["winner"] is None)
view = reasoning.chain_view(store, [A], AS_OF)
names = [i["name"] for i in view[0]["items"]]
check("chain ordered by effective date", names == ["Loan_Agreement_2023.pdf", "Amendment_1_2024.pdf", "Amendment_2_2025.pdf", "Amendment_3_2027.pdf"])
check("chain marks Amendment 3 not yet in force", view[0]["items"][-1]["status"] == "not yet in force")

# ---------- 5. full QA with a scripted LLM ----------
config.MIN_SIM = 0.15
late_a2 = next(c for c in store.doc_chunks(A2) if "2.5%" in c["text"])
late_a = next(c for c in store.doc_chunks(A) if "2% per month" in c["text"])
calls = []


def scripted(system, user, schema=None, retries=1):
    calls.append(system[:20])
    if system.startswith("You are a verifiable"):
        keyfor = lambda doc: re.search(r"\[(C\d+)\] " + re.escape(doc), user).group(1)
        return {"claims": [
            {"text": "The late payment charge is 2.5% per month on the overdue amount.",
             "evidence": [{"chunk": keyfor("Amendment_2_2025.pdf"), "quote": "The late payment charge shall be 2.5% per month on the overdue amount"}]},
            {"text": "The original agreement set the charge at 2% per month.",
             "evidence": [{"chunk": keyfor("Loan_Agreement_2023.pdf"), "quote": "a late payment charge of 2% per month shall be levied on the overdue amount"}]},
            {"text": "A flat penalty of Rs. 500 also applies.",
             "evidence": [{"chunk": keyfor("Amendment_2_2025.pdf"), "quote": "a flat penalty of Rs. 500 shall also be payable on every default"}]},
            {"text": "The charge is 4% per month.",
             "evidence": [{"chunk": keyfor("Amendment_2_2025.pdf"), "quote": "The late payment charge shall be 2.5% per month on the overdue amount"}]}],
            "missing": [], "calculation": None}
    if system.startswith("You are a strict"):
        import json
        items = json.loads(user)
        return {"results": [{"id": i["id"], "verdict": "supported", "reason": "ok"} for i in items]}
    if system.startswith("You are a meticulous"):
        pa = re.search(r"\[(P\d+)\] \(Amendment_2_2025", user)
        pb = re.search(r"\[(P\d+)\] \(Loan_Agreement_2023", user)
        if pa and pb:
            return {"conflicts": [{"subject": "late payment charge", "reason": "2% vs 2.5%",
                                   "a": {"chunk": pb.group(1), "quote": "a late payment charge of 2% per month shall be levied on the overdue amount"},
                                   "b": {"chunk": pa.group(1), "quote": "The late payment charge shall be 2.5% per month on the overdue amount"}}]}
        return {"conflicts": []}
    raise AssertionError("unexpected prompt")


llm.chat_json = scripted
r = qa.ask(store, "What is the late payment charge that applies today?", AS_OF)
print("   status:", r["status"], "| faithfulness:", r["faithfulness"], "| blocked:", [b["reason"][:60] for b in r["blocked"]])
check("2 claims verified", len(r["answer"]) == 2)
check("fabricated-quote claim blocked", any("not appear verbatim" in b["reason"] for b in r["blocked"]))
check("claim with figure not in quote blocked by number guard", any("not in the quoted evidence" in b["reason"] for b in r["blocked"]))
check("faithfulness = 2/4 = 50%", r["faithfulness"] == {"supported": 2, "total": 4, "score": 50.0})
check("citations carry doc + page + verbatim quote",
      all(c["page"] and c["quote"] and c["doc"] for a in r["answer"] for c in a["citations"]))
check("conflict classified as superseded by amendment (not a contradiction)",
      len(r["conflicts"]) == 1 and r["conflicts"][0]["status"] == "superseded" and r["conflicts"][0]["winner"] == "b")
check("status 'partial' (blocked claims, no unresolved conflict)", r["status"] == "partial")
check("amendment chain returned", len(r["chain"]) == 1 and len(r["chain"][0]["items"]) == 4)

# unresolved contradiction -> status conflict
def conflicting(system, user, schema=None, retries=1):
    if system.startswith("You are a meticulous"):
        pa = re.search(r"\[(P\d+)\] \(Loan_Agreement_2023[^\[]*?tenure of the loan shall not exceed 60 months", user, re.S)
        pb = re.search(r"\[(P\d+)\] \(Product_Brochure_2025", user)
        if pa and pb:
            return {"conflicts": [{"subject": "tenure", "reason": "60 vs 84 months",
                                   "a": {"chunk": pa.group(1), "quote": "The tenure of the loan shall not exceed 60 months"},
                                   "b": {"chunk": pb.group(1), "quote": "flexible tenure of up to 84 months"}}]}
        return {"conflicts": []}
    return scripted(system, user, schema, retries)


llm.chat_json = conflicting
r = qa.ask(store, "What is the maximum tenure of the loan in months?", AS_OF)
print("   status:", r["status"], "| conflicts:", [(c["status"], c["subject"]) for c in r["conflicts"]])
check("unlinked docs conflict -> status 'conflict'", r["status"] == "conflict" and r["conflicts"][0]["status"] == "contradiction")

# abstention: unrelated question never reaches the LLM
llm.chat_json = boom
config.MIN_SIM = 0.4
r = qa.ask(store, "weather forecast tomorrow in Paris?", AS_OF)
check("abstains before calling the LLM when nothing relevant", r["status"] == "abstained" and r["missing"])
config.MIN_SIM = 0.15

# abstention: everything blocked
def all_fake(system, user, schema=None, retries=1):
    if system.startswith("You are a verifiable"):
        return {"claims": [{"text": "The foreclosure penalty is 5%.", "evidence": [{"chunk": "C1", "quote": "foreclosure penalty is five percent of principal"}]}], "missing": ["foreclosure penalty"], "calculation": None}
    return {"conflicts": []}


llm.chat_json = all_fake
r = qa.ask(store, "What is the loan prepayment charge penalty?", AS_OF)
check("all claims blocked -> abstained with missing evidence", r["status"] == "abstained" and r["missing"] == ["foreclosure penalty"] and r["faithfulness"]["score"] == 0.0)

# judge unavailable -> fail closed
def judge_down(system, user, schema=None, retries=1):
    if system.startswith("You are a strict"):
        raise llm.LLMError("down")
    return scripted(system, user, schema, retries)


llm.chat_json = judge_down
r = qa.ask(store, "What is the late payment charge that applies today?", AS_OF, check_conflicts=False)
check("verifier down -> nothing is shown as verified", r["status"] == "abstained" and not r["answer"] and r["warnings"])

# ---------- 6. calculation ----------
check("safe_eval arithmetic", abs(qa.safe_eval("P*r/100", {"P": 20000, "r": 2.5}) - 500) < 1e-9)
for bad in ["__import__('os').system('x')", "P.__class__", "2**100000"]:
    try:
        qa.safe_eval(bad, {"P": 1})
        check("safe_eval rejects " + bad, False)
    except Exception:
        check("safe_eval rejects " + bad, True)
ev = {"C1": {"key": "C1", "chunk": late_a2}}
calc = qa.run_calculation({"description": "late fee", "expression": "P*rate/100", "variables": [
    {"name": "P", "value": 20000, "chunk": "C1", "quote": "late payment charge shall be 2.5% per month on the overdue amount"},
    {"name": "rate", "value": 2.5, "chunk": "C1", "quote": "late payment charge shall be 2.5% per month on the overdue amount"}]}, ev)
check("calculation input not in quote is rejected (P=20000 is the user's number)", calc["ok"] is False)
calc = qa.run_calculation({"description": "x", "expression": "rate*2", "variables": [
    {"name": "rate", "value": 2.5, "chunk": "C1", "quote": "late payment charge shall be 2.5% per month on the overdue amount"}]}, ev)
check("calculation with quoted input works", calc["ok"] and calc["result"] == 5.0)

print("\nFAILED:" if check.failed else "\nALL TESTS PASSED", check.failed or "")
sys.exit(1 if check.failed else 0)
