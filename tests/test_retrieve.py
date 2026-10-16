
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"

sys.path.insert(0, str(SRC_DIR))

from rag_assistant import retrieve_context


def test_retrieval_returns_passages():
    passages = retrieve_context(
        "month-to-month contract and high churn risk"
    )

    assert isinstance(passages, list)
    assert len(passages) > 0


def test_retrieved_passages_have_required_fields():
    passages = retrieve_context(
        "What should we review for a high-risk customer?"
    )

    assert passages

    for passage in passages:
        assert "text" in passage
        assert "source" in passage
        assert "page" in passage
        assert "distance" in passage
        assert isinstance(passage["text"], str)
        assert passage["text"].strip()


def test_retrieval_respects_top_k():
    passages = retrieve_context(
        "customer retention policy",
        top_k=2,
    )

    assert len(passages) <= 2


def test_retrieval_returns_relevant_source_metadata():
    passages = retrieve_context(
        "month-to-month contract and high churn risk"
    )

    assert passages

    for passage in passages:
        assert passage["source"]
        assert passage["page"] is not None
        assert passage["distance"] >= 0
