"""
ingest.py - Read annual report PDFs, cut them into small chunks, and store them
in a local ChromaDB vector database (with company, year and page number).

Run from the project root:   python ingest.py
PDF names must look like:    TCS_FY26.pdf, INFY_FY25.pdf, HUL_FY26.pdf, ITC_FY25.pdf
(Put the PDFs in knowledge/pdfs/)

Every run deletes the old collection and rebuilds it, so it is safe to re-run.
"""
import re
from pathlib import Path

import chromadb
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parent
PDF_DIR = ROOT / "knowledge" / "pdfs"
CHROMA_DIR = ROOT / "knowledge" / "chroma_db"
COLLECTION = "annual_reports"
MODEL_NAME = "all-MiniLM-L6-v2"   # rag_tool.py must use the SAME model


CHUNK_WORDS = 120      # upper limit; the code below shrinks it if needed
OVERLAP_WORDS = 20
MIN_WORDS = 20
MAX_TOKENS = 230       # leaves room for the header and the model limit of 256

# Old file names -> official company code
ALIASES = {"INF": "INFY", "INFOSYS": "INFY"}


def parse_name(pdf_path):
    """TCS_FY26.pdf -> ('TCS', 'FY26'). Returns None if the name is not valid."""
    m = re.match(r"^([A-Za-z]+)_(FY\d{2})$", pdf_path.stem.strip(), re.IGNORECASE)
    if not m:
        return None
    company = m.group(1).upper()
    return ALIASES.get(company, company), m.group(2).upper()


def make_chunks(text, tokenizer):
    """Cut a page into chunks that fit the model's token limit."""
    words = text.split()
    start = 0
    while start < len(words):
        end = min(start + CHUNK_WORDS, len(words))
        # make the chunk shorter until it fits the token limit
        while end - start > MIN_WORDS and \
                len(tokenizer(" ".join(words[start:end]))["input_ids"]) > MAX_TOKENS:
            end -= 10
        piece = words[start:end]
        if len(piece) >= MIN_WORDS:
            yield " ".join(piece)
        if end >= len(words):
            break
        start = max(end - OVERLAP_WORDS, start + 1)   # always move forward


def main():
    pdfs = sorted(PDF_DIR.glob("*.pdf"))
    if not pdfs:
        print(f"No PDFs found in {PDF_DIR}")
        return

    print("Loading embedding model (first run downloads it, please wait)...")
    model = SentenceTransformer(MODEL_NAME)

    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    try:
        client.delete_collection(COLLECTION)
    except Exception:
        pass  # collection did not exist yet
    col = client.create_collection(COLLECTION, metadata={"hnsw:space": "cosine"})

    for pdf in pdfs:
        parsed = parse_name(pdf)
        if not parsed:
            print(f"SKIPPED {pdf.name}: name must look like TCS_FY26.pdf")
            continue
        company, year = parsed

        reader = PdfReader(str(pdf))
        ids, docs, metas = [], [], []
        empty_pages = 0
        for page_no, page in enumerate(reader.pages, start=1):  # page_no = PDF page number
            try:
                text = page.extract_text() or ""
            except Exception:
                text = ""
            if not text.strip():
                empty_pages += 1
                continue
            for j, chunk in enumerate(make_chunks(text, model.tokenizer)):
                ids.append(f"{company}_{year}_p{page_no}_c{j}")
                docs.append(f"{company} {year} PDF p.{page_no}: {chunk}")
                metas.append({"company": company, "year": year,
                              "page": page_no, "source": pdf.name})

        for k in range(0, len(docs), 256):
            batch = docs[k:k + 256]
            embeddings = model.encode(batch, normalize_embeddings=True).tolist()
            col.add(ids=ids[k:k + 256], documents=batch,
                    metadatas=metas[k:k + 256], embeddings=embeddings)

        total = len(reader.pages)
        print(f"{pdf.name}: {len(docs)} chunks from {total} pages "
              f"({empty_pages} pages had no text)")
        if total and empty_pages / total > 0.3:
            print(f"  WARNING: many pages have no text. {pdf.name} may be a scanned PDF "
                  f"(images). Text search will be weak for it.")

    print("Done. Total chunks in database:", col.count())


if __name__ == "__main__":
    main()