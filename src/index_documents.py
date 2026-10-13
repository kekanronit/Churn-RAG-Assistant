
from pathlib import Path
import hashlib

import chromadb
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer


ROOT_DIR = Path(__file__).resolve().parent.parent
DOCS_DIR = ROOT_DIR / "docs"
VECTORSTORE_DIR = ROOT_DIR / "vectorstore"

COLLECTION_NAME = "retention_knowledge_base"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150


def extract_pdf_text(pdf_path: Path) -> list[dict]:
    """Extract text page by page from a PDF."""
    reader = PdfReader(str(pdf_path))
    pages = []

    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""

        if text.strip():
            pages.append({
                "text": text.strip(),
                "page": page_number,
            })

    return pages


def split_text(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> list[str]:
    """Split text into overlapping character-based chunks."""
    if chunk_size <= 0 or overlap < 0 or overlap >= chunk_size:
        raise ValueError("Require chunk_size > overlap >= 0.")

    text = " ".join(text.split())
    chunks = []
    start = 0

    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end == len(text):
            break

        start = end - overlap

    return chunks


def main():
    if not DOCS_DIR.exists():
        raise FileNotFoundError(
            f"Documents folder not found: {DOCS_DIR}"
        )

    pdf_files = sorted(DOCS_DIR.glob("*.pdf"))

    if not pdf_files:
        print(f"No PDF files found in {DOCS_DIR}")
        print("Add a retention-policy PDF and run this script again.")
        return

    print(f"Found {len(pdf_files)} PDF file(s).")
    print(f"Loading embedding model: {EMBEDDING_MODEL}")

    model = SentenceTransformer(EMBEDDING_MODEL)

    client = chromadb.PersistentClient(
        path=str(VECTORSTORE_DIR)
    )

    # Use a dedicated collection for this indexing run.
    # Recreate it to avoid stale or duplicate chunks on reruns.
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass

    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    total_chunks = 0

    for pdf_path in pdf_files:
        pages = extract_pdf_text(pdf_path)
        document_chunks = []

        for page_info in pages:
            chunks = split_text(page_info["text"])

            for chunk_index, chunk in enumerate(chunks):
                document_chunks.append({
                    "text": chunk,
                    "page": page_info["page"],
                    "chunk_index": chunk_index,
                })

        if not document_chunks:
            print(f"Warning: no extractable text in {pdf_path.name}")
            continue

        texts = [item["text"] for item in document_chunks]
        embeddings = model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=True,
        )

        ids = []
        metadatas = []

        for item in document_chunks:
            identity = (
                f"{pdf_path.name}|{item['page']}|"
                f"{item['chunk_index']}|{item['text']}"
            )
            chunk_id = hashlib.sha256(
                identity.encode("utf-8")
            ).hexdigest()

            ids.append(chunk_id)
            metadatas.append({
                "source": pdf_path.name,
                "page": item["page"],
                "chunk_index": item["chunk_index"],
            })

        collection.add(
            ids=ids,
            documents=texts,
            embeddings=embeddings.tolist(),
            metadatas=metadatas,
        )

        total_chunks += len(texts)
        print(
            f"Indexed {len(texts)} chunk(s) from {pdf_path.name}"
        )

    print("\n--- PDF Indexing Complete ---")
    print(f"PDF files processed: {len(pdf_files)}")
    print(f"Chunks indexed: {total_chunks}")
    print(f"ChromaDB collection: {COLLECTION_NAME}")
    print(f"Vector store: {VECTORSTORE_DIR}")


if __name__ == "__main__":
    main()
