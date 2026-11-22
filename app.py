"""
app.py - Mini screener backend: screener data, saved reports and the chat route.

Put this file in the project ROOT (next to ingest.py and chat_engine.py).
Run from the project root:
    uv run uvicorn app:app --reload
Then open http://127.0.0.1:8000
"""
import math
import re
import sqlite3
import threading
import time
import os
import uuid
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

import chat_engine

ROOT = Path(__file__).resolve().parent
os.chdir(ROOT)          # the crew writes report.md in the current folder
DB = ROOT / "knowledge" / "financials.db"
REPORT_DIR = ROOT / "knowledge" / "reports"
STATIC_DIR = ROOT / "static"

FIRST_FY, LAST_FY = 2022, 2026

COMPANIES = {
    "TCS":  {"name": "Tata Consultancy Services", "yf": "TCS.NS",         "sector": "IT Services"},
    "INFY": {"name": "Infosys",                   "yf": "INFY.NS",        "sector": "IT Services"},
    "HUL":  {"name": "Hindustan Unilever",        "yf": "HINDUNILVR.NS",  "sector": "FMCG"},
    "ITC":  {"name": "ITC",                       "yf": "ITC.NS",         "sector": "FMCG"},
}

# Columns we try to read from annual_wide. Missing ones are skipped (no crash).
ANNUAL_COLS = [
    "sales", "sales_growth_pct", "ebitda", "ebitda_margin_pct", "ebit_margin_pct",
    "net_profit", "net_margin_pct", "eps", "roe_pct_year_end_equity", "roce_pct",
    "borrowings", "debt_to_equity", "cash_bank",
]

DISCLAIMER = (
    "This report was written by AI agents (the crew). The crew can make mistakes. "
    "Please cross-check every number and claim with the original annual reports "
    "and the company filings before you use it. This is not investment advice."
)

app = FastAPI(title="Mini Screener")


# ----------------------------------------------------------------- database helpers
def _connect():
    if not DB.exists():
        raise HTTPException(503, "financials.db not found on the server")
    con = sqlite3.connect(DB.resolve().as_uri() + "?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    return con


def _columns(con, name):
    return [r[1] for r in con.execute(f"PRAGMA table_info({name})")]


def _clean(v):
    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
        return None
    return v


def _row(r):
    return {k: _clean(r[k]) for k in r.keys()}


def _code(code):
    c = code.strip().upper()
    if c not in COMPANIES:
        raise HTTPException(404, f"Unknown company. Use one of: {', '.join(COMPANIES)}")
    return c


def _annual(con, code):
    have = set(_columns(con, "annual_wide"))
    use = ["fiscal_year"] + [c for c in ANNUAL_COLS if c in have]
    rows = con.execute(
        f"SELECT {', '.join(use)} FROM annual_wide "
        "WHERE company=? AND fiscal_year BETWEEN ? AND ? ORDER BY fiscal_year",
        (code, FIRST_FY, LAST_FY),
    ).fetchall()
    return [_row(r) for r in rows]


def _quarterly(con, code):
    have = set(_columns(con, "quarterly_wide"))
    use = [c for c in ("period", "period_end", "sales", "operating_profit", "net_profit") if c in have]
    if not use:
        return []
    rows = con.execute(
        f"SELECT {', '.join(use)} FROM quarterly_wide WHERE company=? "
        "ORDER BY period_end DESC LIMIT 8",
        (code,),
    ).fetchall()
    return [_row(r) for r in reversed(rows)]


def _valuation(con, code):
    try:
        r = con.execute("SELECT * FROM valuation_now WHERE company=?", (code,)).fetchone()
    except sqlite3.Error:
        return {}
    return _row(r) if r else {}


# ----------------------------------------------------------------- price (yfinance, cached)
PRICE_TTL = 15 * 60       # keep a good answer for 15 minutes
FAIL_TTL = 60             # keep a failure for 1 minute (do not hammer Yahoo)
_price_cache = {}
_price_lock = threading.Lock()


def _num(v):
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return None if (math.isnan(f) or math.isinf(f)) else round(f, 2)


def get_price(code):
    now = time.time()
    with _price_lock:
        hit = _price_cache.get(code)
        if hit and now < hit["until"]:
            return hit["data"]
    data = {"price": None, "high52": None, "low52": None, "source": "unavailable"}
    ttl = FAIL_TTL
    try:
        import yfinance as yf
        fi = yf.Ticker(COMPANIES[code]["yf"]).fast_info
        p, hi, lo = _num(fi["lastPrice"]), _num(fi["yearHigh"]), _num(fi["yearLow"])
        if p is not None:
            data = {"price": p, "high52": hi, "low52": lo, "source": "Yahoo Finance (yfinance)"}
            ttl = PRICE_TTL
    except Exception:
        pass
    with _price_lock:
        _price_cache[code] = {"data": data, "until": now + ttl}
    return data


def _with_price(item, price, valuation):
    p = dict(price)
    if p["price"] is None and valuation.get("current_price") is not None:
        p["price"] = _num(valuation["current_price"])
        p["source"] = "Excel data file (date not stated)"
    if p["price"] is not None and p["high52"] and p["low52"] and p["high52"] > p["low52"]:
        p["from_high_pct"] = round((p["price"] / p["high52"] - 1) * 100, 1)
        pos = (p["price"] - p["low52"]) / (p["high52"] - p["low52"]) * 100
        p["range_pos_pct"] = round(min(max(pos, 0), 100), 1)
    item["price"] = p


# ----------------------------------------------------------------- API: screener data
@app.get("/api/health")
def health():
    return {"ok": True, "db": DB.exists(), "reports": len(_report_files())}


@app.get("/api/screener")
def screener():
    con = _connect()
    try:
        items = []
        for code, meta in COMPANIES.items():
            ann = _annual(con, code)
            items.append({
                "code": code, "name": meta["name"], "sector": meta["sector"],
                "latest": ann[-1] if ann else {},
                "valuation": _valuation(con, code),
            })
    finally:
        con.close()
    with ThreadPoolExecutor(max_workers=4) as ex:
        prices = list(ex.map(get_price, [i["code"] for i in items]))
    for item, price in zip(items, prices):
        _with_price(item, price, item["valuation"])
    return {"companies": items}


@app.get("/api/company/{code}")
def company(code: str):
    code = _code(code)
    con = _connect()
    try:
        annual = _annual(con, code)
        quarterly = _quarterly(con, code)
        valuation = _valuation(con, code)
    finally:
        con.close()
    item = {"code": code, "name": COMPANIES[code]["name"], "sector": COMPANIES[code]["sector"],
            "annual": annual, "quarterly": quarterly, "valuation": valuation}
    _with_price(item, get_price(code), valuation)
    return item


# ----------------------------------------------------------------- API: saved reports
def _report_files():
    return sorted(REPORT_DIR.glob("*.md")) if REPORT_DIR.exists() else []


def _report_path(rid):
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,60}", rid):
        raise HTTPException(404, "Report not found")
    path = REPORT_DIR / f"{rid}.md"
    if not path.is_file():
        raise HTTPException(404, "Report not found")
    return path


def _sections(md):
    """Split a markdown report on '#' and '##' headings."""
    parts, title, buf = [], "Report", []
    for line in md.splitlines():
        m = re.match(r"^#{1,2}\s+(.*\S)\s*$", line)
        if m:
            if "".join(buf).strip():
                parts.append({"title": title, "body": "\n".join(buf).strip()})
            title, buf = m.group(1), []
        else:
            buf.append(line)
    if "".join(buf).strip():
        parts.append({"title": title, "body": "\n".join(buf).strip()})
    return parts


@app.get("/api/reports")
def reports():
    return {"reports": [{"id": p.stem, "title": p.stem.replace("_", " ")} for p in _report_files()]}


@app.get("/api/reports/{rid}")
def report(rid: str):
    path = _report_path(rid)
    md = path.read_text(encoding="utf-8")
    return {"id": rid, "title": rid.replace("_", " "), "sections": _sections(md),
            "words": len(md.split()), "disclaimer": DISCLAIMER}


@app.get("/api/reports/{rid}/download")
def report_download(rid: str):
    path = _report_path(rid)
    return FileResponse(path, media_type="text/markdown", filename=f"{rid}.md")


# ----------------------------------------------------------------- API: chat
chat_engine.PRICE_FN = get_price          # lets the bot answer price questions with the same cached yfinance data


class ChatIn(BaseModel):
    message: str = Field(..., min_length=1, max_length=300)


def _visitor(request: Request):
    """Visitor id for the per-visitor cap. Behind a host proxy the real address is in x-forwarded-for.
    (A determined person can fake this header: fine for a demo, not for real security.)"""
    fwd = request.headers.get("x-forwarded-for", "")
    if fwd:
        return fwd.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


@app.post("/api/chat")
def chat(body: ChatIn, request: Request):
    return chat_engine.answer(body.message, _visitor(request))


@app.get("/api/chat/status")
def chat_status():
    return chat_engine.status()


# load the embedding model in the background so the first AI question is not slow
threading.Thread(target=chat_engine.warm_up, daemon=True).start()

# ----------------------------------------------------------------- API: make a detailed report on demand
REPORT_DAILY_CAP = int(os.getenv("REPORT_DAILY_CAP", "3"))      # reports per day for the whole site
REPORT_VISITOR_CAP = int(os.getenv("REPORT_VISITOR_CAP", "1"))  # reports per day for one visitor
_jobs = {}
_rep = {"day": None, "total": 0, "ip": {}, "current": None}
_rep_lock = threading.Lock()


class ReportIn(BaseModel):
    a: str
    b: str


def _pair_report(a, b):
    for p in _report_files():
        if {a, b} <= set(re.split(r"[^A-Z0-9]+", p.stem.upper())):
            return p.stem
    return None


def _is_limit(msg):
    m = msg.lower()
    return any(k in m for k in ("429", "resource_exhausted", "quota", "rate limit", "rate_limit"))


def _run_job(job_id, a, b):
    job = _jobs[job_id]
    failed, quota = False, False
    try:
        import make_reports
        job.update(status="done", report=make_reports.generate(a, b).stem)
    except Exception as e:
        print("report job failed:", repr(e))           # read this line in the terminal to see the real reason
        failed = True
        quota = _is_limit(str(e))
        job.update(status="error", message=("Today's limit is reached. Please try again tomorrow." if quota
                   else "The report could not be made right now. Please try again later."))
    finally:
        with _rep_lock:
            _rep["current"] = None
            if failed:                                  # the visitor got no report, so give the try back
                _rep["ip"][job["ip"]] = max(0, _rep["ip"].get(job["ip"], 1) - 1)
                # quota error: block everyone for today. Other error: give the site try back too
                _rep["total"] = REPORT_DAILY_CAP if quota else max(0, _rep["total"] - 1)


@app.post("/api/reports/generate")
def report_generate(body: ReportIn, request: Request):
    a, b = body.a.strip().upper(), body.b.strip().upper()
    if a == b or a not in COMPANIES or b not in COMPANIES:
        raise HTTPException(400, "Pick two different companies: " + ", ".join(COMPANIES))
    done = _pair_report(a, b)
    if done:                                            # already saved: free, instant
        return {"status": "done", "report": done}
    ip = _visitor(request)
    with _rep_lock:
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        if _rep["day"] != today:
            _rep.update(day=today, total=0, ip={})
        cur = _rep["current"]
        if cur and time.time() - _jobs[cur]["started"] < 1200:        # one report at a time
            if _jobs[cur]["pair"] == {a, b}:
                return {"status": "running", "job": cur}
            return {"status": "busy", "message": "Another report is being written now. Please try again in a few minutes."}
        if _rep["total"] >= REPORT_DAILY_CAP:
            return {"status": "limit", "message": "Today's report limit is reached. Please try again tomorrow."}
        if _rep["ip"].get(ip, 0) >= REPORT_VISITOR_CAP:
            return {"status": "limit", "message": "You already asked for a report today. Please try again tomorrow."}
        job_id = uuid.uuid4().hex[:10]
        _jobs[job_id] = {"status": "running", "pair": {a, b}, "started": time.time(), "ip": ip}
        _rep.update(current=job_id, total=_rep["total"] + 1)
        _rep["ip"][ip] = _rep["ip"].get(ip, 0) + 1
    threading.Thread(target=_run_job, args=(job_id, a, b), daemon=True).start()
    return {"status": "running", "job": job_id}


@app.get("/api/reports/jobs/{job_id}")
def report_job(job_id: str):
    job = _jobs.get(job_id)
    if not job:
        raise HTTPException(404, "Unknown job")
    return {k: v for k, v in job.items() if k in ("status", "report", "message")}

# ----------------------------------------------------------------- pages
STATIC_DIR.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")