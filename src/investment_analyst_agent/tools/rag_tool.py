"""
rag_tool.py - CrewAI tool that searches the annual-report vector database.
Put this file in: src/investment_research_analyst/tools/rag_tool.py

Build the database first with:  python ingest.py
"""
import re
from pathlib import Path
from typing import Optional, Type

from crewai.tools import BaseTool
from pydantic import BaseModel, Field

import threading

# this file -> tools -> investment_research_analyst -> src -> project root
ROOT = Path(__file__).resolve().parents[3]
CHROMA_DIR = ROOT / "knowledge" / "chroma_db"
COLLECTION = "annual_reports"
MODEL_NAME = "all-MiniLM-L6-v2"   # must match ingest.py

ALIASES = {
    "INF": "INFY", "INFOSYS": "INFY",
    "TATA CONSULTANCY SERVICES": "TCS",
    "HINDUSTAN UNILEVER": "HUL", "HIND. UNILEVER": "HUL",
}

_model = None
_collection = None


_lock = threading.Lock()

def _load():
    """Load the model and the database only once (on first use)."""
    global _model, _collection
    if _collection is None:
        with _lock:
            if _collection is None:   # check again inside the lock
                import chromadb
                from sentence_transformers import SentenceTransformer
                _model = SentenceTransformer(MODEL_NAME)
                client = chromadb.PersistentClient(path=str(CHROMA_DIR))
                _collection = client.get_collection(COLLECTION)
    return _model, _collection


def _norm_company(company):
    if not company:
        return None
    c = company.strip().upper()
    return ALIASES.get(c, c)


def _norm_year(year):
    """'fy26', 'FY26', '2026' -> 'FY26'"""
    if not year:
        return None
    y = str(year).strip().upper()
    m = re.search(r"(\d{2,4})$", y)
    return f"FY{m.group(1)[-2:]}" if m else y


def search_reports(query, company=None, year=None, n_results=4):
    """Return a list of dicts: text, company, year, page, distance.
    (Also useful later for measuring retrieval hit rate.)"""
    model, col = _load()
    conds = []
    if _norm_company(company):
        conds.append({"company": _norm_company(company)})
    if _norm_year(year):
        conds.append({"year": _norm_year(year)})
    where = None if not conds else (conds[0] if len(conds) == 1 else {"$and": conds})

    n = max(1, min(int(n_results), 8))
    emb = model.encode([query], normalize_embeddings=True).tolist()
    res = col.query(query_embeddings=emb, n_results=n, where=where)

    out = []
    for doc, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0]):
        out.append({"text": doc, "company": meta["company"], "year": meta["year"],
                    "page": meta["page"], "distance": round(dist, 3)})
    return out


class RagInput(BaseModel):
    query: str = Field(..., description="What to find, e.g. 'reasons for margin decline' or 'key risks'")
    company: Optional[str] = Field(None, description="One of: TCS, INFY, HUL, ITC. Leave empty to search all.")
    year: Optional[str] = Field(None, description="Report year like 'FY26' or 'FY25'. Leave empty for both.")
    n_results: int = Field(4, description="How many passages to return (1 to 8)")


class AnnualReportSearchTool(BaseTool):
    name: str = "annual_report_search"
    description: str = (
        "Searches the FY25 and FY26 annual reports of TCS, INFY (Infosys), HUL and ITC. "
        "Use it for management commentary, reasons behind results, risks, outlook and strategy. "
        "Returns text passages with company, year and PDF page number. "
        "Do NOT use it for exact financial numbers: use the financials_sql tool for those."
    )
    args_schema: Type[BaseModel] = RagInput

    def _run(self, query: str, company: Optional[str] = None,
             year: Optional[str] = None, n_results: int = 4) -> str:
        if not CHROMA_DIR.exists():
            return "Vector database not found. Run ingest.py first."
        try:
            hits = search_reports(query, company, year, n_results)
        except Exception as e:
            return f"Search error: {e}"
        if not hits:
            return "No passages found. Check the company code (TCS, INFY, HUL, ITC) and year (FY25, FY26)."
        blocks = [f"[{h['company']} {h['year']} | PDF page {h['page']} | distance {h['distance']}]\n{h['text']}"
                  for h in hits]
        return "\n\n---\n\n".join(blocks)