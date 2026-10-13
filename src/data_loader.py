"""
Data loader – reads .txt and .pdf documents from the data/ directory
and the project root.
"""

import os
import glob
from src.config import DATA_DIR, PROJECT_ROOT


def _extract_pdf_text(filepath: str) -> str:
    """Extract text content from a PDF file using PyPDF2."""
    try:
        from PyPDF2 import PdfReader
        reader = PdfReader(filepath)
        pages = []
        for page in reader.pages:
            text = page.extract_text()
            if text:
                pages.append(text.strip())
        return "\n\n".join(pages)
    except Exception as e:
        print(f"[DataLoader] Error extracting PDF {filepath}: {e}")
        return ""


def load_documents(data_dir: str = DATA_DIR) -> list[dict]:
    """
    Load all .txt files from data_dir and .pdf files from both
    data_dir and the project root.

    Returns a list of dicts:
        {
            "document_name": <filename without ext>,
            "source": <full file path>,
            "text": <file content>
        }
    """
    documents = []

    # ── Load .txt files from data/ ──
    txt_files = sorted(glob.glob(os.path.join(data_dir, "*.txt")))
    for filepath in txt_files:
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                text = f.read().strip()
            if text:
                doc_name = os.path.splitext(os.path.basename(filepath))[0]
                documents.append({
                    "document_name": doc_name,
                    "source": filepath,
                    "text": text,
                })
        except Exception as e:
            print(f"[DataLoader] Error reading {filepath}: {e}")

    # ── Load .pdf files from data/ ──
    pdf_files_data = sorted(glob.glob(os.path.join(data_dir, "*.pdf")))
    for filepath in pdf_files_data:
        text = _extract_pdf_text(filepath)
        if text:
            doc_name = os.path.splitext(os.path.basename(filepath))[0]
            documents.append({
                "document_name": doc_name,
                "source": filepath,
                "text": text,
            })

    # ── Load .pdf files from project root ──
    pdf_files_root = sorted(glob.glob(os.path.join(PROJECT_ROOT, "*.pdf")))
    for filepath in pdf_files_root:
        # Avoid duplicates if data/ is in root
        if filepath in [p for p in pdf_files_data]:
            continue
        text = _extract_pdf_text(filepath)
        if text:
            doc_name = os.path.splitext(os.path.basename(filepath))[0]
            documents.append({
                "document_name": doc_name,
                "source": filepath,
                "text": text,
            })

    if not documents:
        print(f"[DataLoader] WARNING: No documents found in {data_dir} or {PROJECT_ROOT}")
    else:
        print(f"[DataLoader] Loaded {len(documents)} documents "
              f"({len(txt_files)} txt + {len(pdf_files_data) + len(pdf_files_root)} pdf)")

    return documents
