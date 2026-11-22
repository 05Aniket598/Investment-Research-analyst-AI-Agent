"""
setup_db.py - Load the "Data Sheet" of each company's Excel file into SQLite.

Run from the project root:   python setup_db.py
Put the Excel files in knowledge/excel/  (names: TCS, Infosys, Hind. Unilever, ITC)
Needs:  pip install pandas openpyxl   (and xlrd only if your files are old .xls)

Creates knowledge/financials.db with:
  - table financials      (tall: one number per row)
  - view  annual_wide     (one row per company per year, includes ratios and valuation multiples)
  - view  quarterly_wide  (one row per company per quarter)
  - table valuation_now   (one row per company: CURRENT price and current multiples)
Every run rebuilds the database from scratch.

All ratios are calculated HERE, in Python, with fixed formulas, so the LLM never has to
calculate or invent them. Formulas (INR crore unless stated):
  ebitda      = profit_before_tax + interest + depreciation - other_income
  ebit        = ebitda - depreciation
  eps         = net_profit / adjusted_equity_shares_in_cr            (INR per share)
  bvps        = (equity_share_capital + reserves) / adjusted shares  (INR per share)
  pe, pb      = year-end price / eps, year-end price / bvps
  market_cap_year_end = year-end price * adjusted shares
  ev          = market_cap_year_end + borrowings - cash_bank
  roe         = net_profit / (equity_share_capital + reserves) * 100   (year-end equity)
  roce        = ebit / (equity_share_capital + reserves + borrowings) * 100
  ROIC is NOT calculated (it needs definitions this data does not have).
A ratio is left empty (NULL) when one of its inputs is blank. Blank is never treated as 0.
"""
import re
import sqlite3
from datetime import datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
EXCEL_DIR = ROOT / "knowledge" / "excel"
DB_PATH = ROOT / "knowledge" / "financials.db"
SHEET_NAME = "Data Sheet"

# Excel file name (lowercase, no extension) -> company code used everywhere
FILES = {
    "tcs": "TCS",
    "infosys": "INFY",
    "hind. unilever": "HUL",
    "itc": "ITC",
}

# Heading in column A -> annual or quarterly
SECTIONS = {
    "profit & loss": "annual",
    "quarters": "quarterly",
    "balance sheet": "annual",
    "cash flow:": "annual",
    "price:": "annual",
    "derived:": "annual",
}

# Rows in the META block at the top of the sheet
META_KEYS = {
    "current price": "current_price",
    "market capitalization": "market_cap_cr",
    "face value": "face_value",
    "number of shares": "number_of_shares",
}


def find_excel(stem):
    for p in EXCEL_DIR.iterdir():
        if p.suffix.lower() in (".xlsx", ".xlsm", ".xls") and p.stem.strip().lower() == stem:
            return p
    return None


def num(v):
    """Return a float, or None for empty / text cells."""
    try:
        if pd.isna(v):
            return None
        return float(v)
    except (TypeError, ValueError):
        return None


def div(a, b, mult=1.0, nd=2):
    """a / b * mult, or None if an input is missing (or b is 0)."""
    if a is None or b is None or b == 0:
        return None
    return round(a / b * mult, nd)


def parse_date(d):
    """Mar-26 -> ('Mar-26', 2026, '2026-03'). Indian FY: Apr-Mar, Mar-26 is FY26."""
    if isinstance(d, str):
        d = datetime.strptime(d.strip(), "%b-%y")
    fy = d.year + 1 if d.month >= 4 else d.year
    return d.strftime("%b-%y"), fy, d.strftime("%Y-%m")


def months_between(a, b):
    """'2025-12', '2026-03' -> 3"""
    ya, ma = map(int, a.split("-"))
    yb, mb = map(int, b.split("-"))
    return (yb - ya) * 12 + (mb - ma)


def unit_for(metric):
    if metric in ("price", "face_value"):
        return "INR"
    if metric == "no_of_equity_shares":
        return "shares"
    if metric == "adjusted_equity_shares_in_cr":
        return "crore shares"
    return "INR crore"   # assumption: the sheet is in INR crore (please verify with a PDF)


def read_company(path, code):
    """Returns (rows, meta). rows go to the financials table, meta = META block values."""
    df = pd.read_excel(path, sheet_name=SHEET_NAME, header=None)
    rows, meta, kind, dates = [], {}, None, None
    for _, r in df.iterrows():
        if pd.isna(r.iloc[0]):
            continue
        label = str(r.iloc[0]).strip()
        key = label.lower()
        values = list(r.iloc[1:])

        if key == "report date":          # the dates row of the current block
            dates = values
            continue
        if key in SECTIONS:               # a block heading
            kind = SECTIONS[key]
            if key != "price:":           # PRICE keeps its numbers on the heading row
                continue
        if dates is None and key in META_KEYS:   # META block, before the first dates row
            meta[META_KEYS[key]] = num(values[0]) if values else None
            continue
        if dates is None or kind is None:  # still in the META area at the top
            continue

        metric = re.sub(r"[^a-z0-9]+", "_", key).strip("_")
        for i, v in enumerate(values):
            val = num(v)
            if val is None or i >= len(dates) or pd.isna(dates[i]):
                continue
            try:
                period, fy, pend = parse_date(dates[i])
            except (ValueError, TypeError, AttributeError):
                continue
            rows.append((code, kind, period, fy, pend, metric, val, unit_for(metric)))
    return rows, meta


def add_derived(con):
    """Ratios and valuation multiples per company per fiscal year (annual section)."""
    q = """SELECT company, period, fiscal_year, period_end, metric, value
           FROM financials WHERE section='annual'"""
    data = {}
    for c, p, fy, pe, m, v in con.execute(q).fetchall():
        data.setdefault(c, {}).setdefault((pe, p, fy), {})[m] = v

    for c, years in data.items():
        prev_sales = None
        for (pe, p, fy), x in sorted(years.items()):
            out = {}
            sales, profit = x.get("sales"), x.get("net_profit")
            pbt, dep = x.get("profit_before_tax"), x.get("depreciation")
            intr, oi = x.get("interest"), x.get("other_income")
            borrow, cash = x.get("borrowings"), x.get("cash_bank")
            sh, price = x.get("adjusted_equity_shares_in_cr"), x.get("price")
            eq = None
            if x.get("equity_share_capital") is not None and x.get("reserves") is not None:
                eq = x["equity_share_capital"] + x["reserves"]

            def put(metric, val, unit):
                if val is not None:
                    out[metric] = (val, unit)

            put("net_margin_pct", div(profit, sales, 100), "%")
            if prev_sales and sales:
                put("sales_growth_pct", round((sales / prev_sales - 1) * 100, 2), "%")
            put("roe_pct_year_end_equity", div(profit, eq, 100), "%")
            if eq is not None and borrow is not None:
                put("debt_to_equity", div(borrow, eq), "ratio")

            ebitda = ebit = None
            if None not in (pbt, dep, intr, oi):
                ebitda = pbt + intr + dep - oi
                ebit = ebitda - dep
                put("ebitda", round(ebitda, 2), "INR crore")
                put("ebit", round(ebit, 2), "INR crore")
                put("ebitda_margin_pct", div(ebitda, sales, 100), "%")
                put("ebit_margin_pct", div(ebit, sales, 100), "%")
                if eq is not None and borrow is not None:
                    put("roce_pct", div(ebit, eq + borrow, 100), "%")

            eps = div(profit, sh)
            bvps = div(eq, sh)
            put("eps", eps, "INR per share")
            put("bvps", bvps, "INR per share")
            put("pe", div(price, eps), "x")
            put("pb", div(price, bvps), "x")

            if price is not None and sh:
                mcap = price * sh
                put("market_cap_year_end", round(mcap, 2), "INR crore")
                if borrow is not None and cash is not None:
                    ev = mcap + borrow - cash
                    put("ev", round(ev, 2), "INR crore")
                    put("ev_ebitda", div(ev, ebitda), "x")
                    put("ev_sales", div(ev, sales), "x")

            prev_sales = sales or prev_sales
            for m, (val, unit) in out.items():
                con.execute("INSERT OR REPLACE INTO financials VALUES (?,?,?,?,?,?,?,?)",
                            (c, "annual", p, fy, pe, m, val, unit))


def build_valuation_now(con, meta_all):
    """One row per company with the CURRENT price (from the Excel META block)."""
    con.execute("DROP TABLE IF EXISTS valuation_now")
    con.execute("""CREATE TABLE valuation_now(
        company TEXT PRIMARY KEY, current_price REAL, market_cap_cr REAL,
        latest_fiscal_year INTEGER, eps_latest_fy REAL, eps_ttm REAL,
        pe_latest_fy REAL, pe_ttm REAL, bvps_latest_fy REAL, pb_latest_fy REAL,
        ev_cr REAL, ebitda_ttm REAL, ev_ebitda_ttm REAL, ttm_quarters TEXT, note TEXT)""")

    for c, meta in meta_all.items():
        price = meta.get("current_price")
        latest_end = con.execute(
            "SELECT MAX(period_end) FROM financials WHERE company=? AND section='annual' AND metric='eps'",
            (c,)).fetchone()[0]
        if not price or not latest_end:
            print(f"  valuation_now: skipped {c} (no current price or no annual data)")
            continue
        latest = dict(con.execute(
            "SELECT metric, value FROM financials WHERE company=? AND section='annual' AND period_end=?",
            (c, latest_end)).fetchall())
        fy = con.execute(
            "SELECT fiscal_year FROM financials WHERE company=? AND section='annual' AND period_end=? LIMIT 1",
            (c, latest_end)).fetchone()[0]
        sh = latest.get("adjusted_equity_shares_in_cr")

        # last 4 consecutive quarters -> trailing twelve months (TTM)
        by_end = {}
        for period, pend, m, v in con.execute(
                """SELECT period, period_end, metric, value FROM financials
                   WHERE company=? AND section='quarterly'
                   AND metric IN ('net_profit','operating_profit')""", (c,)).fetchall():
            by_end.setdefault(pend, {"period": period})[m] = v
        ends = sorted(by_end)[-4:]
        ttm_np = ttm_ebitda = ttm_label = None
        if len(ends) == 4 and all(months_between(a, b) == 3 for a, b in zip(ends, ends[1:])):
            nps = [by_end[e].get("net_profit") for e in ends]
            ops = [by_end[e].get("operating_profit") for e in ends]
            if None not in nps:
                ttm_np = sum(nps)
            if None not in ops:
                ttm_ebitda = sum(ops)
            ttm_label = f"{by_end[ends[0]]['period']} to {by_end[ends[-1]]['period']}"

        eps_fy, bvps = latest.get("eps"), latest.get("bvps")
        eps_ttm = div(ttm_np, sh)
        mcap = meta.get("market_cap_cr") or (round(price * sh, 2) if sh else None)
        borrow, cash = latest.get("borrowings"), latest.get("cash_bank")
        ev = round(mcap + borrow - cash, 2) if None not in (mcap, borrow, cash) else None

        con.execute("INSERT INTO valuation_now VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", (
            c, price, mcap, fy, eps_fy, eps_ttm,
            div(price, eps_fy), div(price, eps_ttm), bvps, div(price, bvps),
            ev, ttm_ebitda, div(ev, ttm_ebitda), ttm_label,
            "Current price and market cap come from the Excel META block (the file does not state "
            "the price date; assumed to be the download date). EV uses borrowings and cash of the "
            "latest fiscal year. TTM = sum of the last 4 quarters in the Excel."))


def create_views(con):
    """Wide views: one column per metric. Makes the agent's SQL very simple."""
    for section, view in [("annual", "annual_wide"), ("quarterly", "quarterly_wide")]:
        metrics = [m for (m,) in con.execute(
            "SELECT DISTINCT metric FROM financials WHERE section=? ORDER BY metric", (section,))]
        cols = ",\n  ".join(
            f"MAX(CASE WHEN metric='{m}' THEN value END) AS \"{m}\"" for m in metrics)
        con.execute(f"DROP VIEW IF EXISTS {view}")
        con.execute(f"""CREATE VIEW {view} AS
            SELECT company, period, fiscal_year, period_end,
              {cols}
            FROM financials WHERE section='{section}'
            GROUP BY company, period, fiscal_year, period_end""")


def main():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(DB_PATH)
    con.execute("DROP VIEW IF EXISTS annual_wide")
    con.execute("DROP VIEW IF EXISTS quarterly_wide")
    con.execute("DROP TABLE IF EXISTS financials")
    con.execute("""CREATE TABLE financials(
        company TEXT, section TEXT, period TEXT, fiscal_year INTEGER,
        period_end TEXT, metric TEXT, value REAL, unit TEXT,
        PRIMARY KEY(company, section, period, metric))""")

    meta_all = {}
    for stem, code in FILES.items():
        path = find_excel(stem)
        if path is None:
            print(f"MISSING: no Excel file named '{stem}' in {EXCEL_DIR}")
            continue
        rows, meta = read_company(path, code)
        con.executemany("INSERT OR REPLACE INTO financials VALUES (?,?,?,?,?,?,?,?)", rows)
        meta_all[code] = meta
        print(f"{code}: {len(rows)} numbers loaded from {path.name}; META: {meta}")

    add_derived(con)
    build_valuation_now(con, meta_all)
    create_views(con)
    con.commit()

    print("\nRows per company and section:")
    for row in con.execute(
            "SELECT company, section, COUNT(*) FROM financials GROUP BY company, section"):
        print("  ", row)
    print("\nannual_wide columns:")
    print("  ", [r[1] for r in con.execute("PRAGMA table_info(annual_wide)")])
    print("quarterly_wide columns:")
    print("  ", [r[1] for r in con.execute("PRAGMA table_info(quarterly_wide)")])
    print("\nvaluation_now:")
    cols = [r[1] for r in con.execute("PRAGMA table_info(valuation_now)")][:-1]  # skip long note
    for row in con.execute(f"SELECT {', '.join(cols)} FROM valuation_now"):
        print("  ", dict(zip(cols, row)))
    con.close()


if __name__ == "__main__":
    main()