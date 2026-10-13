"""
Text chunking module – splits documents into overlapping chunks with metadata.
"""

from src.config import CHUNK_SIZE, CHUNK_OVERLAP


def chunk_document(document: dict,
                   chunk_size: int = CHUNK_SIZE,
                   chunk_overlap: int = CHUNK_OVERLAP) -> list[dict]:
    """
    Split a single document dict into a list of chunk dicts.

    Each chunk dict contains:
        document_name, chunk_id, source, text
    """
    text = document["text"]
    doc_name = document["document_name"]
    source = document["source"]

    chunks = []
    start = 0
    chunk_idx = 0

    while start < len(text):
        end = start + chunk_size

        # Try to break at sentence / paragraph boundary
        if end < len(text):
            # Look for a period, newline, or space near the boundary
            for sep in ["\n\n", "\n", ". ", " "]:
                boundary = text.rfind(sep, start + chunk_size // 2, end + 50)
                if boundary != -1:
                    end = boundary + len(sep)
                    break

        chunk_text = text[start:end].strip()
        if chunk_text:
            chunks.append({
                "document_name": doc_name,
                "chunk_id": f"{doc_name}_chunk_{chunk_idx}",
                "source": source,
                "text": chunk_text,
            })
            chunk_idx += 1

        start = end - chunk_overlap if end < len(text) else len(text)

    return chunks


def chunk_all_documents(documents: list[dict],
                        chunk_size: int = CHUNK_SIZE,
                        chunk_overlap: int = CHUNK_OVERLAP) -> list[dict]:
    """Chunk every document and return a flat list of chunks."""
    all_chunks = []
    for doc in documents:
        all_chunks.extend(chunk_document(doc, chunk_size, chunk_overlap))
    print(f"[Chunker] Created {len(all_chunks)} chunks from {len(documents)} documents")
    return all_chunks
