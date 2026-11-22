"""
make_reports.py - make ONE saved peer report offline, for the website.

Run from the project root (it uses Gemini: about 4 requests per report):
    uv run python make_reports.py TCS INFY
    uv run python make_reports.py HUL ITC

What it does:
  1. runs prefetch (SQL + annual-report search, no LLM)
  2. runs the crew (same steps as main.py) and gets report.md
  3. copies the result to knowledge/reports/TCS_vs_INFY.md and adds a "please cross-check" note at the end

It never overwrites a saved report. To rebuild one, delete its file first.
READ the new report before you commit it to GitHub (check the numbers against the screener tables).
"""
import os
import sys
import warnings
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPORT = ROOT / "report.md"                    # the crew writes here (output_file in crew.py)
OUT_DIR = ROOT / "knowledge" / "reports"
CODES = {"TCS", "INFY", "HUL", "ITC"}
MIN_CHARS = 3000                                # a real report is about 2000 words

FOOTER = (
    "\n\n---\n\n"
    "*Note: this report was written by AI agents (the crew). The crew can make mistakes. "
    "Please cross-check every number and claim with the original annual reports and company filings "
    "before you use it. This is not investment advice.*\n"
)

warnings.filterwarnings("ignore", category=SyntaxWarning, module="pysbd")
os.environ.setdefault("LITELLM_LOG", "ERROR")


def generate(stock, peer):
    """Run prefetch + crew and save the report. Returns the saved Path. Raises an error if it fails."""
    from investment_analyst_agent.crew import InvestmentResearchAnalyst
    from investment_analyst_agent.prefetch import run_prefetch

    data = run_prefetch(stock, [peer])
    inputs = {
        "stock": stock,
        "peer_companies": peer,
        "report_date": datetime.now().strftime("%d %B %Y"),
        **data,
    }
    REPORT.unlink(missing_ok=True)
    InvestmentResearchAnalyst().crew().kickoff(inputs=inputs)
    if not REPORT.exists():
        raise RuntimeError("The crew finished but report.md was not written.")
    text = REPORT.read_text(encoding="utf-8").strip()
    if len(text) < MIN_CHARS:
        raise RuntimeError(f"report.md is very short ({len(text)} characters).")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"{stock}_vs_{peer}.md"
    out.write_text(text + FOOTER, encoding="utf-8")
    return out


def main():
    if len(sys.argv) != 3:
        print("Usage: uv run python make_reports.py TCS INFY")
        return 1
    stock, peer = sys.argv[1].upper(), sys.argv[2].upper()
    if stock == peer or stock not in CODES or peer not in CODES:
        print(f"Use two different codes from: {', '.join(sorted(CODES))}")
        return 1

    out = OUT_DIR / f"{stock}_vs_{peer}.md"
    # the website finds a report by the two codes in any order, so do not make a duplicate
    for p in OUT_DIR.glob("*.md") if OUT_DIR.exists() else []:
        if {stock, peer} <= set(p.stem.upper().replace("-", "_").split("_")):
            print(f"A report for this pair already exists: {p.name}. Delete it first if you want a new one.")
            return 1

    os.chdir(ROOT)
    try:
        out = generate(stock, peer)
    except Exception as e:
        print(f"Failed: {e}. If this is a 429 (quota) or 503 error, wait and try again later. No report was saved.")
        return 1
    print(f"\nSaved: {out}")
    print("Next: READ it, check the numbers against the screener, then git add + commit + push.")
    return 0


if __name__ == "__main__":
    sys.exit(main())