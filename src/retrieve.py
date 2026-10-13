
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer


ROOT_DIR = Path(__file__).resolve().parent.parent
VECTORSTORE_DIR = ROOT_DIR / "vectorstore"

COLLECTION_NAME = "retention_knowledge_base"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"


def retrieve_documents(
    question: str,
    top_k: int = 3,
) -> list[dict]:
    """Retrieve relevant retention-policy passages for a question."""

    if not question.strip():
        raise ValueError("Question cannot be empty.")

    if top_k < 1:
        raise ValueError("top_k must be at least 1.")

    client = chromadb.PersistentClient(
        path=str(VECTORSTORE_DIR)
    )

    try:
        collection = client.get_collection(COLLECTION_NAME)
    except Exception as exc:
        raise RuntimeError(
            "Knowledge base not found. Run "
            "'python src/index_documents.py' first."
        ) from exc

    if collection.count() == 0:
        return []

    model = SentenceTransformer(EMBEDDING_MODEL)
    query_embedding = model.encode(
        question,
        normalize_embeddings=True,
    ).tolist()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(top_k, collection.count()),
        include=["documents", "metadatas", "distances"],
    )

    matches = []

    for document, metadata, distance in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        matches.append({
            "text": document,
            "source": metadata["source"],
            "page": metadata["page"],
            "distance": float(distance),
        })

    return matches


if __name__ == "__main__":
    question = (
        "What can we do when a customer has a month-to-month "
        "contract and a high risk of churning?"
    )

    print("\n--- Retention Knowledge Search ---")
    print(f"Question: {question}\n")

    matches = retrieve_documents(question, top_k=3)

    if not matches:
        print("No matching passages found.")
    else:
        for index, match in enumerate(matches, start=1):
            print(f"Result {index}")
            print(f"Source: {match['source']}")
            print(f"PDF page: {match['page']}")
            print(f"Distance: {match['distance']:.4f}")
            print(f"Passage: {match['text']}\n")
