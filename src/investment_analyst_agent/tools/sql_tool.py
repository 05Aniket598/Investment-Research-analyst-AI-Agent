"""
sql_tool.py - CrewAI tool: the agent writes ONE read-only SQL SELECT query and
gets exact numbers from the financial database.
Put this file in: src/investment_analyst_agent/tools/sql_tool.py

Build the database first with:  python setup_db.py
The column lists in the tool description are read from the real database, so they
always match what is stored.
"""
import sqlite3
from pathlib import Path
from typing import Type

from crewai.tools import BaseTool
from pydantic import BaseModel, Field

# this file -> tools -> package folder -> src -> project root
ROOT = Path(__file__).resolve().parents[3]
DB = ROOT / "knowledge" / "financials.db"
MAX_ROWS = 100
OBJECTS = ("annual_wide", "quarterly_wide", "valuation_now")


def _columns(con, name):
    return [r[1] for r in con.execute(f"PRAGMA table_info({name})")]


def _build_description():
    base = (
        "Runs ONE read-only SQLite SELECT query on the financial database and returns the rows. "
        "Use it for ALL numbers (sales, profit, debt, ratios, growth, multiples, price). "
        "Never guess numbers and never calculate in your head: ratios and multiples are already "
        "in the database; do any extra calculation inside the SQL.\n"
    )
    if not DB.exists():
        return base + "The database is not built yet. Run setup_db.py."
    try:
        con = sqlite3.connect(DB)
        companies = [c for (c,) in con.execute("SELECT DISTINCT company FROM financials ORDER BY company")]
        years = con.execute("SELECT MIN(fiscal_year), MAX(fiscal_year) FROM financials WHERE section='annual'").fetchone()
        quarters = con.execute("SELECT MIN(period_end), MAX(period_end) FROM financials WHERE section='quarterly'").fetchone()
        annual_cols = _columns(con, "annual_wide")
        quarterly_cols = _columns(con, "quarterly_wide")
        now_cols = _columns(con, "valuation_now")
        con.close()
    except sqlite3.Error:
        return base + "The database exists but could not be read. Run setup_db.py again."

    return (
        base
        + f"Company codes (column company): {', '.join(companies)}.\n"
        + f"VIEW annual_wide: one row per company per year, fiscal_year {years[0]} to {years[1]} "
          "(fiscal_year 2026 = FY26 = year ended March 2026). "
          f"Columns: {', '.join(annual_cols)}.\n"
        + f"VIEW quarterly_wide: one row per company per quarter, period_end {quarters[0]} to {quarters[1]} "
          f"(format YYYY-MM). Columns: {', '.join(quarterly_cols)}.\n"
        + f"TABLE valuation_now: one row per company with the CURRENT share price and current multiples. "
          f"Columns: {', '.join(now_cols)}. Use annual_wide for history (its price, pe, pb, ev_ebitda use "
          "the YEAR-END price of each fiscal year) and valuation_now for today's valuation.\n"
        + "Units: money columns are INR crore. Exceptions: price, current_price = INR per share; "
          "eps, bvps = INR per share; no_of_equity_shares = count; adjusted_equity_shares_in_cr = crore "
          "shares; columns ending in _pct are percent; pe, pb, ev_ebitda, ev_sales are multiples (x); "
          "debt_to_equity is a ratio. NULL means the value is not available.\n"
        + "Formulas already calculated (do not recalculate): ebitda = profit_before_tax + interest + "
          "depreciation - other_income; ebit = ebitda - depreciation; eps = net_profit / adjusted shares; "
          "bvps = (equity_share_capital + reserves) / adjusted shares; ev = market cap + borrowings - "
          "cash_bank; roe_pct_year_end_equity uses year-end equity (not average); roce_pct = ebit / "
          "(equity_share_capital + reserves + borrowings). ROIC is NOT available: say 'not available'.\n"
        + "Example: SELECT company, fiscal_year, sales, net_profit, ebitda_margin_pct, roce_pct, pe FROM "
          "annual_wide WHERE company IN ('TCS','INFY') AND fiscal_year >= 2022 ORDER BY company, fiscal_year\n"
        + "Example: SELECT company, current_price, pe_ttm, pb_latest_fy, ev_ebitda_ttm FROM valuation_now"
    )


_DESCRIPTION = _build_description()


def _schema_help():
    try:
        con = sqlite3.connect(DB)
        parts = [f"{name} columns: {', '.join(_columns(con, name))}" for name in OBJECTS]
        con.close()
        return "\n".join(parts)
    except sqlite3.Error:
        return ""


def _connect_read_only():
    try:
        con = sqlite3.connect(DB.resolve().as_uri() + "?mode=ro", uri=True)
    except sqlite3.Error:
        con = sqlite3.connect(DB)   # fallback; query_only below still blocks writes
    con.execute("PRAGMA query_only = ON")
    return con


class SQLInput(BaseModel):
    query: str = Field(..., description="One SQLite SELECT query on annual_wide, quarterly_wide or valuation_now")


class FinancialsSQLTool(BaseTool):
    name: str = "financials_sql"
    description: str = _DESCRIPTION
    args_schema: Type[BaseModel] = SQLInput

    def _run(self, query: str) -> str:
        if not DB.exists():
            return "Database not found. Run setup_db.py first."
        q = query.strip().rstrip(";").strip()
        if ";" in q or not q.lower().startswith(("select", "with")):
            return "Only ONE SELECT query is allowed (no semicolons inside)."

        con = None
        try:
            con = _connect_read_only()
            cur = con.execute(q)
            cols = [c[0] for c in cur.description]
            rows = cur.fetchmany(MAX_ROWS + 1)
        except sqlite3.Error as e:
            return f"SQL error: {e}\n{_schema_help()}"
        finally:
            if con:
                con.close()

        if not rows:
            return "Query ran but returned no rows. Check the company code and fiscal_year."
        truncated = len(rows) > MAX_ROWS
        rows = rows[:MAX_ROWS]
        lines = [" | ".join(cols)] + [
            " | ".join("" if v is None else str(v) for v in r) for r in rows]
        note = f"\n(Showing first {MAX_ROWS} rows only)" if truncated else ""
        return "\n".join(lines) + note + "\n(Source: financial database built from Excel data)"