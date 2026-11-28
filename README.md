# Investment Research Analyst: RAG + SQL + Multi-Agent Stock Screener

An AI research assistant for four Indian companies: **TCS, Infosys (INFY), Hindustan Unilever (HUL) and ITC**.
Exact numbers come from a SQL database. Explanations come from the companies' annual reports through RAG. A CrewAI team of agents writes a full peer-comparison report.

**Live demo:** https://mini-screener-0rq4.onrender.com/
*(Hosted on a free tier. It sleeps when idle, so the first load can take a minute.)*

---

## The main idea

> **Numbers from SQL. Reasons from annual reports. The LLM only explains.**

LLMs make up numbers. So in this project the LLM never calculates or remembers a number. Python and SQL fetch every figure first, and the LLM only reads and explains what it was given.

| Question type | Where the answer comes from | LLM used? |
|---|---|---|
| "TCS sales FY22 to FY26" | SQLite database | No |
| "Which company has the highest ROCE?" | SQLite database | No |
| "Compare TCS and INFY" | Table + saved crew report | No |
| "Why did TCS margin change in FY26?" | Database numbers + annual report passages | Yes (Gemini) |
| "Generate detailed report" | Full CrewAI run | Yes (Gemini) |

---

## Features

- **Stock screener page** with annual numbers, quarterly trend, valuation multiples, and live price with 52-week range (yfinance).
- **Chat bot** that answers number questions straight from the database and uses AI only when the question needs explanation.
- **Cost-aware routing:** cache, then Python router, then compare, then AI, then fallback. The cheapest path is always tried first, so the free Gemini limit lasts.
- **Citations:** AI answers cite the annual report as `(Company FYxx, p. N)` using the PDF page number.
- **Peer reports:** a CrewAI team (financial, risk and valuation analysts plus a manager) writes a 2,000 to 2,500 word report.
- **Daily caps** for the whole site and for each visitor, so the free API quota is not used up.
- **Safety rules:** no buy, sell or hold advice. A "please cross-check" note is added to every AI report.

---

## How it works

```
Excel financials ──► setup_db.py ──► financials.db (SQLite: annual_wide, quarterly_wide, valuation_now)
Annual report PDFs ─► ingest.py ───► Chroma vector DB (all-MiniLM-L6-v2 embeddings)

Chat question
   │
   ├─ 1. cache hit?                          → free
   ├─ 2. number question?  → SQL tables      → free, exact
   ├─ 3. "compare A and B" → table + report  → free
   ├─ 4. "why / explain"   → DB snapshot + annual report pages → 1 Gemini call
   └─ 5. daily cap reached → show passages without AI → free

Detailed report button
   prefetch.py (Python: SQL + RAG search, no LLM)
        → Financial analyst → Risk analyst → Valuation analyst → Manager
        → report.md → saved in knowledge/reports/
```

**Key design choices**

- **Prefetch, then reason.** `prefetch.py` runs all SQL and RAG searches in Python before the crew starts. The agents have no tools and only reason over the data given. This keeps Gemini use to about 4 requests per report.
- **Python does the math.** Growth rates (CAGR), margin changes, scenario prices and sensitivity tables are calculated in Python. The agents are told not to recalculate.
- **Reported vs adjusted profit.** Reported profit and profit excluding exceptional items are kept apart and labelled, so the report does not call a correct number wrong.
- **Guardrails in the prompts.** "Reason not found in the annual report" is required when no passage explains a change. Missing metrics (such as ROIC) must be written as "not available". The investment stance is limited when the evidence is thin.

---

## Tech stack

| Part | Tool |
|---|---|
| Agents | CrewAI |
| LLM | Google Gemini (model names set in `.env`) |
| Vector search | ChromaDB + `sentence-transformers` (all-MiniLM-L6-v2) |
| Numbers | SQLite |
| Backend | FastAPI + Uvicorn |
| Frontend | Plain HTML and JavaScript |
| Live prices | yfinance |
| Hosting | Render (free tier) |

---

## Project structure

```
├── app.py                  # FastAPI: screener, saved reports, chat and report-job routes
├── chat_engine.py          # Chat router, cache, caps, Gemini call
├── make_reports.py         # Runs prefetch + crew, saves a peer report
├── requirements.txt
├── static/index.html       # The web page
├── knowledge/
│   ├── financials.db       # SQLite database built from Excel
│   ├── chroma_db/          # Vector database built from annual reports
│   └── reports/            # Saved peer reports (.md)
└── src/investment_analyst_agent/
    ├── crew.py             # Agents and tasks
    ├── main.py             # Run one report from the terminal
    ├── prefetch.py         # SQL + RAG data collection (no LLM)
    ├── config/             # agents.yaml, tasks.yaml
    └── tools/              # rag_tool.py, sql_tool.py
```

The raw Excel files and annual report PDFs are not in this repository.

---

## Run it locally

Use Python 3.12 (CrewAI does not support 3.14 at the time of writing).

```bash
git clone https://github.com/05Aniket598/Investment-Research-analyst-AI-Agent
cd Investment-Research-analyst-AI-Agent
pip install -r requirements.txt
```

Create a `.env` file:

```
GEMINI_API_KEY=your_key_here
```

Set the import path and start the app:

```bash
# Windows PowerShell
$env:PYTHONPATH = "src"
uvicorn app:app --reload

# macOS / Linux
PYTHONPATH=src uvicorn app:app --reload
```

Open http://127.0.0.1:8000

Make a peer report from the terminal (uses about 4 Gemini requests):

```bash
python make_reports.py TCS INFY
```

---

## Retrieval quality (honest results)

I tested retrieval on 10 hand-written questions about the annual reports.

| Check | Result |
|---|---|
| Right chunk found in top 5 | 5 of 10 |
| Right chunk found in top 20 | 6 of 10 |
| Right **page** found in top 5 | 7 of 10 |
| Right page found in top 20 | 8 of 10 |

**What I learned:** the right page is usually found, but the exact number often sits in a neighbouring chunk. PDF extraction also splits or joins numbers in tables. That is why exact numbers come from SQL and not from the PDF text.

**Planned improvements:** neighbour-chunk expansion, keyword filtering, and a reranker.

---

## Limitations

- Only four companies and two annual reports (FY25, FY26) per company.
- The price date in the Excel file is not stated. Live prices come from Yahoo Finance (unofficial) and can be delayed.
- ROE and ROCE use year-end values, so they can differ from other websites.
- The retrieval test above is small (10 questions) and uses strict matching.
- AI reports can make mistakes. Check the numbers against the original annual reports.
- On the free host, reports made by the button are lost on restart. Reports saved in `knowledge/reports/` stay.
- **This is not investment advice.**

---

## Author

**Aniket Yadav**
B.Sc. Data Science & AI, Mumbai
[LinkedIn](https://linkedin.com/in/aniketyadavofficial/) | [GitHub](https://github.com/05Aniket598)
