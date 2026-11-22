"""
prefetch.py - Python (NO LLM) collects every number and every annual-report passage
BEFORE the crew starts. The crew then only reasons and writes.

Put this file in: src/investment_analyst_agent/prefetch.py
Build the databases first:  python setup_db.py  and  python ingest.py

Test it alone (prints everything the analysts will see, costs 0 Gemini requests):
    python -m investment_analyst_agent.prefetch TCS INFY

Returns a dict with 5 text blocks. main.py passes them to the crew as inputs:
    financial_data          tables for FY22-FY26 + last 8 quarters + pre-computed numbers
    valuation_data          valuation_now + historical multiple ranges
    commentary_performance  annual-report passages: growth, margins, costs, outlook
    commentary_risk         annual-report passages: risks
    data_notes              missing / odd data points and limitations
"""
import os
import re
import datetime
import sqlite3
import statistics
import sys
from pathlib import Path

# this file -> package folder -> src -> project root
ROOT = Path(__file__).resolve().parents[2]
DB = Path(os.getenv("FINANCIALS_DB", ROOT / "knowledge" / "financials.db"))
OUT_DIR = ROOT / "knowledge"

FIRST_FY, LAST_FY = 2022, 2026          # FY22 to FY26 = 4 years of growth
YEARS = LAST_FY - FIRST_FY
MAX_CHARS = 900                         # characters kept per annual-report passage

ADJUSTED = {  # typed by hand from the annual report, NOT from the database
    ("TCS", 2026): {"net_profit_ex_exceptional": 52820, "eps_ex_exceptional": 145.99,
                    "source": "TCS FY26 annual report p.36 and p.59"},
}


# ----------------------------------------------------------------- small helpers

def _fy(fy):
    return f"FY{str(fy)[-2:]}"


def _fmt(v):
    if v is None:
        return "n/a"
    if isinstance(v, float):
        return f"{v:,.0f}" if abs(v) >= 10000 else f"{v:,.2f}"
    return str(v)


def _md(cols, rows):
    if not rows:
        return "(no rows)"
    out = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    out += ["| " + " | ".join(_fmt(v) for v in r) + " |" for r in rows]
    return "\n".join(out)


def _clean(text):
    """One line, and no { } (CrewAI treats { } in inputs as placeholders)."""
    text = re.sub(r"\s+", " ", str(text)).strip()
    return text.replace("{", "(").replace("}", ")")


def _columns(con, name):
    return [r[1] for r in con.execute(f"PRAGMA table_info({name})")]


def _pct(a, b):
    return round(a / b * 100, 2) if a is not None and b not in (None, 0) else None


def _cagr(a, b):
    if a is None or b is None or a <= 0 or b <= 0:
        return None
    return round(((b / a) ** (1 / YEARS) - 1) * 100, 2)


def _diff(a, b):
    return round(b - a, 2) if a is not None and b is not None else None

def _quarter_fy(period):
    """'Jun-26' -> 'FY27 Q1', 'Mar-26' -> 'FY26 Q4'"""
    d = datetime.strptime(period, "%b-%y")
    fy = d.year + 1 if d.month >= 4 else d.year
    q = ((d.month - 4) % 12) // 3 + 1
    return f"FY{str(fy)[-2:]} Q{q}"


# ----------------------------------------------------------------- SQL part
def _fetch_annual(con, codes, cols):
    """{company: {fiscal_year: {metric: value}}} for FY22-FY26. Skips columns that do not exist."""
    have = set(_columns(con, "annual_wide"))
    use = [c for c in cols if c in have]
    marks = ",".join(["?"] * len(codes))
    sql = (f"SELECT company, fiscal_year, {', '.join(use)} FROM annual_wide "
           f"WHERE company IN ({marks}) AND fiscal_year BETWEEN ? AND ?")
    data = {}
    for row in con.execute(sql, (*codes, FIRST_FY, LAST_FY)):
        data.setdefault(row[0], {})[row[1]] = dict(zip(use, row[2:]))
    return data, use


def _table(data, codes, cols):
    rows = []
    for c in codes:
        for fy in sorted(data.get(c, {})):
            rows.append([c, _fy(fy)] + [data[c][fy].get(col) for col in cols])
    return _md(["company", "fy"] + cols, rows)

def _adjusted_block(codes):
    rows = [[c, _fy(fy), v["net_profit_ex_exceptional"], v["eps_ex_exceptional"], v["source"]]
            for (c, fy), v in ADJUSTED.items() if c in codes]
    return _md(["company", "fy", "net_profit_ex_exceptional", "eps_ex_exceptional", "source"], rows) if rows else "(none)"


def _financial_blocks(con, codes):
    """Returns (financial_data_text, annual_data, used_columns)."""
    cols_all = [
        "sales", "sales_growth_pct", "ebitda", "ebitda_margin_pct", "ebit_margin_pct",
        "net_profit", "net_margin_pct", "roe_pct_year_end_equity", "roce_pct",
        "borrowings", "debt_to_equity", "cash_bank", "cash_from_operating_activity",
        "employee_cost", "other_expenses", "depreciation", "other_income",
        "profit_before_tax", "eps",
    ]
    data, used = _fetch_annual(con, codes, cols_all)

    # cost structure, calculated here (not by the LLM)
    for c in data:
        for fy, r in data[c].items():
            r["employee_cost_pct_sales"] = _pct(r.get("employee_cost"), r.get("sales"))
            r["other_expenses_pct_sales"] = _pct(r.get("other_expenses"), r.get("sales"))
            r["other_income_pct_pbt"] = _pct(r.get("other_income"), r.get("profit_before_tax"))

    growth = ["sales", "sales_growth_pct", "ebitda", "ebitda_margin_pct", "ebit_margin_pct",
              "net_profit", "net_margin_pct", "eps"]
    returns = ["roe_pct_year_end_equity", "roce_pct", "borrowings", "debt_to_equity",
               "cash_bank", "cash_from_operating_activity"]
    cost = ["employee_cost_pct_sales", "other_expenses_pct_sales", "depreciation",
            "other_income", "other_income_pct_pbt"]
    keep = lambda lst: [c for c in lst if c in used or c in cost]

    # pre-computed summary
    summ = []
    for c in codes:
        y = data.get(c, {})
        a, b = y.get(FIRST_FY, {}), y.get(LAST_FY, {})
        avg = lambda k: (round(statistics.mean(v[k] for v in y.values() if v.get(k) is not None), 2)
                         if any(v.get(k) is not None for v in y.values()) else None)
        summ.append([
            c,
            _cagr(a.get("sales"), b.get("sales")),
            _cagr(a.get("net_profit"), b.get("net_profit")),
            _cagr(a.get("eps"), b.get("eps")),
            _diff(a.get("ebitda_margin_pct"), b.get("ebitda_margin_pct")),
            _diff(a.get("net_margin_pct"), b.get("net_margin_pct")),
            _diff(a.get("roce_pct"), b.get("roce_pct")),
            avg("ebitda_margin_pct"), avg("roce_pct"), avg("roe_pct_year_end_equity"),
        ])
    summary = _md(
        ["company", f"sales_CAGR_%_{_fy(FIRST_FY)}-{_fy(LAST_FY)}", "net_profit_CAGR_%", "eps_CAGR_%",
         "ebitda_margin_change_pp", "net_margin_change_pp", "roce_change_pp",
         "avg_ebitda_margin_%_5yrs_FY22-FY26", "avg_roce_%_5yrs_FY22-FY26", "avg_roe_%_5yrs_FY22-FY26"], summ)

    # last 8 quarters
    q_have = set(_columns(con, "quarterly_wide"))
    qcols = [c for c in ("sales", "operating_profit", "net_profit") if c in q_have]
    marks = ",".join(["?"] * len(codes))
    qrows = []
    for c in codes:
        cur = con.execute(
            f"SELECT period, {', '.join(qcols)} FROM quarterly_wide WHERE company=? ORDER BY period_end DESC LIMIT 8",
            (c,))
        for r in reversed(cur.fetchall()):
            d = dict(zip(["period"] + qcols, r))
            qrows.append([c, _quarter_fy(d["period"]) + " (" + d["period"] + ")", d.get("sales"), d.get("operating_profit"),
                          _pct(d.get("operating_profit"), d.get("sales")), d.get("net_profit")])
    quarterly = _md(["company", "quarter", "sales", "operating_profit", "op_margin_%_(operating_profit/sales; this is NOT EBIT)", "net_profit"], qrows)

    text = "\n\n".join([
        "Units: money in INR crore; eps in INR per share; columns ending _pct are percent. n/a = not available.",
        "### A. Growth and profitability (annual)\n" + _table(data, codes, keep(growth)),
        "### B. Returns and balance sheet (annual; ROE uses year-end equity; ROIC is not available)\n"
        + _table(data, codes, keep(returns)),
        "### C. Cost structure (annual; percentages calculated by Python)\n" + _table(data, codes, keep(cost)),
        f"### D. Pre-computed summary ({_fy(FIRST_FY)} to {_fy(LAST_FY)} is {YEARS} years of growth; "
        "change columns are in percentage points; do NOT recalculate)\n" + summary,
        "### E. Last 8 quarters\n" + quarterly,
        "### E2. Net profit EXCLUDING exceptional items (from annual report; the database holds REPORTED profit)\n" + _adjusted_block(codes),
    ])
    return text, data, used

def _sensitivity(eps, price, pes):
    rows = []
    for chg in (-0.10, 0.0, 0.10):
        e = eps * (1 + chg)
        rows.append([f"EPS {chg:+.0%}"] +
                    [f"{round(pe*e)} ({(pe*e/price-1)*100:+.0f}%)" for pe in pes])
    return _md(["EPS change", "Low P/E", "Base P/E", "High P/E"], rows)

def _scenario_block(con, stock):
    pes = [r[0] for r in con.execute(
        "SELECT pe FROM annual_wide WHERE company=? AND fiscal_year BETWEEN ? AND ? AND pe IS NOT NULL",
        (stock, FIRST_FY, LAST_FY))]
    eps = con.execute("SELECT eps FROM annual_wide WHERE company=? AND fiscal_year=?",
                      (stock, LAST_FY)).fetchone()[0]
    price = con.execute("SELECT current_price FROM valuation_now WHERE company=?",
                        (stock,)).fetchone()[0]
    cases = [("Low (min P/E)", min(pes)), ("Base (median P/E)", statistics.median(pes)),
             ("High (max P/E)", max(pes))]
    bases = [("reported EPS", eps)]
    adj = ADJUSTED.get((stock, LAST_FY))
    if adj:
        bases.append(("EPS excl. exceptional items", adj["eps_ex_exceptional"]))
    rows = []
    for bname, e in bases:
        for cname, pe in cases:
            t = pe * e
            rows.append([bname, cname, round(pe, 2), round(e, 2), round(t),
                         round((t / price - 1) * 100, 1)])
    table = _md(["eps_basis", "case", "target_pe", "eps", "target_price_inr", "vs_current_%"], rows)
    sens = _sensitivity(eps, price, [min(pes), statistics.median(pes), max(pes)])
    return ("### I. Scenario prices (Python: target P/E x FY26 EPS; do NOT recalculate)\n" + table
            + f"\nOnly {len(pes)} yearly P/E values exist. Current price: {price}.\n\n"
            + "### J. Sensitivity (reported EPS changes by -10% / 0 / +10%; price and % vs current)\n" + sens)



def _valuation_block(con, codes, stock):
    # historical multiples
    data, used = _fetch_annual(con, codes, ["pe", "pb", "ev_ebitda", "price", "eps", "bvps"])
    hist = []
    for c in codes:
        y = data.get(c, {})
        row = [c]
        for k in ("pe", "pb", "ev_ebitda"):
            vals = [v[k] for v in y.values() if v.get(k) is not None]
            row += ([round(min(vals), 2), round(statistics.median(vals), 2), round(max(vals), 2)]
                    if vals else [None, None, None])
        hist.append(row)
    hist_t = _md(["company", "pe_min", "pe_median", "pe_max", "pb_min", "pb_median", "pb_max",
                  "ev_ebitda_min", "ev_ebitda_median", "ev_ebitda_max"], hist)

    # year-end history table
    year_end = _table(data, codes, [c for c in ("price", "eps", "bvps", "pe", "pb", "ev_ebitda") if c in used])

    # current valuation
    note = ""
    now_t = "(valuation_now table not available)"
    try:
        cols = [c for c in _columns(con, "valuation_now") if c != "note"]
        marks = ",".join(["?"] * len(codes))
        rows = con.execute(
            f"SELECT {', '.join(cols)} FROM valuation_now WHERE company IN ({marks}) "
            "ORDER BY company != ?, company", (*codes, stock)).fetchall()
        now_t = _md(cols, rows)
        n = con.execute("SELECT note FROM valuation_now LIMIT 1").fetchone()
        note = n[0] if n and n[0] else ""
    except sqlite3.Error as e:
        now_t = f"(valuation_now could not be read: {e})"

    return "\n\n".join([
        "Units: price in INR per share; market cap, EV, EBITDA in INR crore; pe, pb, ev_ebitda in times (x).",
        "### F. Current valuation (valuation_now)\n" + now_t,
        "Price source note: " + (_clean(note) or "not stated")
        + " Sector benchmarks are NOT available.",
        f"### G. Historical multiples {_fy(FIRST_FY)}-{_fy(LAST_FY)} (based on year-end price of each fiscal year)\n" + hist_t,
        "### H. Year-end price and multiples by fiscal year\n" + year_end,
    ])


def _data_notes(data, codes, used, missing, rag_status):
    notes = []
    for c in missing:
        notes.append(f"{c}: NOT in the database. Do not substitute another company.")
    for c in codes:
        y = data.get(c, {})
        gaps = []
        for col in used:
            years = [_fy(fy) for fy in sorted(y) if y[fy].get(col) is None]
            if years:
                gaps.append(f"{col} ({', '.join(years)})")
        if gaps:
            notes.append(f"{c}: values not available for: " + "; ".join(gaps))
        neg = [_fy(fy) for fy in sorted(y) if (y[fy].get("other_income") or 0) < 0]
        if neg:
            notes.append(f"{c}: other_income is negative in {', '.join(neg)} (odd; mention it, do not fix it).")
    notes += [
        "ROIC is not in the database: write 'not available'.",
        "Sector averages and trading volume are not available.",
        "Annual-report passages cover FY25 and FY26 only.",
        "Net profit in the database is REPORTED (after exceptional items). The annual report also gives profit EXCLUDING exceptional items (section E2). State which basis you use. Never call 49,210 or 52,820 wrong.",
        "The database other_income for TCS FY26 (-124) appears to be net of exceptional items; the annual report shows other income of 4,402 (p.36). Do not call it an operating anomaly.",
        "Only 5 yearly P/E values exist: the Low case is NOT a floor. Allowed stance: Hold or 'Valuation inconclusive' only. Maximum confidence: Medium.",
        "Quarter labels: Jun-26 and Sep-26 belong to FY27.",
        rag_status,
    ]
    return "\n".join(f"- {n}" for n in notes if n)


# ----------------------------------------------------------------- annual-report (RAG) part
PERF_QUERIES = [
    "revenue growth drivers by segment, geography and clients",
    "reasons for operating margin change, employee cost and other expenses",
    "other income, finance cost, tax and one-time items",
    "outlook, strategy, deal pipeline and demand",
    "exceptional items, one-time charges and profit excluding exceptional items",
]
RISK_QUERIES = [
    "key risks and risk management",
    "macroeconomic, geopolitical, currency and regulatory risks",
]
PEER_PERF = ["revenue growth drivers", "reasons for operating margin change"]
PEER_RISK = ["key risks and risk management"]


def _search(search_reports, query, company, year, n):
    try:
        return search_reports(query, company=company, year=year, n_results=n)
    except Exception as e:  # keep going, record nothing here; caller notes the failure
        raise RuntimeError(str(e))


def _passages(search_reports, jobs):
    """jobs = [(title, query, company, year, n)]. Returns text grouped by title, no duplicates."""
    seen, groups = set(), {}
    for title, query, company, year, n in jobs:
        for h in _search(search_reports, query, company, year, n):
            key = (h["company"], h["year"], h["page"], h["text"][:60])
            if key in seen:
                continue
            seen.add(key)
            text = _clean(h["text"])[:MAX_CHARS]
            groups.setdefault(title, []).append(f"[{h['company']} {h['year']} | PDF page {h['page']}] {text}")
    return "\n\n".join(f"### {t}\n" + "\n\n".join(v) for t, v in groups.items()) or "(no passages found)"


def _commentary(stock, peers):
    """Returns (performance_text, risk_text, status_line)."""
    try:
        from investment_analyst_agent.tools.rag_tool import search_reports
    except Exception as e:
        msg = f"(annual-report search unavailable: {e})"
        return msg, msg, f"Annual-report search failed to load: {e}"
    perf, risk = [], []
    for q in PERF_QUERIES:
        perf += [(f"{stock}: {q}", q, stock, "FY26", 3), (f"{stock}: {q}", q, stock, "FY25", 2)]
    for q in RISK_QUERIES:
        risk.append((f"{stock}: {q}", q, stock, None, 4))
    for p in peers:
        perf += [(f"{p}: {q}", q, p, None, 3) for q in PEER_PERF]
        risk += [(f"{p}: {q}", q, p, None, 3) for q in PEER_RISK]
    try:
        return (_passages(search_reports, perf), _passages(search_reports, risk),
                "Annual-report passages were retrieved by vector search; each passage carries its PDF page number.")
    except Exception as e:
        msg = f"(annual-report search failed: {e})"
        return msg, msg, f"Annual-report search failed: {e}. Say 'reason not found in the annual report'."


# ----------------------------------------------------------------- main entry
def run_prefetch(stock, peers, save=True):
    stock = stock.strip().upper()
    peers = [p.strip().upper() for p in peers]
    if not DB.exists():
        raise FileNotFoundError(f"{DB} not found. Run setup_db.py first.")
    con = sqlite3.connect(DB)
    try:
        existing = {c for (c,) in con.execute("SELECT DISTINCT company FROM financials")}
        wanted = [stock] + [p for p in peers if p != stock]
        codes = [c for c in wanted if c in existing]
        missing = [c for c in wanted if c not in existing]
        if stock not in existing:
            raise ValueError(f"{stock} is not in the database. Companies found: {sorted(existing)}")
        fin_text, data, used = _financial_blocks(con, codes)
        val_text = _valuation_block(con, codes, stock)
    finally:
        con.close()

    print("prefetch: searching annual reports (first run loads the embedding model)...", flush=True)
    perf, risk, rag_status = _commentary(stock, [c for c in codes if c != stock])

    result = {
        "financial_data": fin_text,
        "valuation_data": val_text,
        "commentary_performance": perf,
        "commentary_risk": risk,
        "data_notes": _data_notes(data, codes, used, missing, rag_status),
    }
    if save:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        out = OUT_DIR / f"prefetch_{stock}.md"
        out.write_text("\n\n".join(f"# {k}\n{v}" for k, v in result.items()), encoding="utf-8")
        print(f"prefetch: saved a copy to {out}", flush=True)
    chars = sum(len(v) for v in result.values())
    print(f"prefetch: done, {chars:,} characters (about {chars // 4:,} tokens) for the crew.", flush=True)
    return result


if __name__ == "__main__":
    args = sys.argv[1:] or ["TCS", "INFY"]
    res = run_prefetch(args[0], args[1:])
    for k, v in res.items():
        print(f"\n\n===== {k} =====\n{v}")