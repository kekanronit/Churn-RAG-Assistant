
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"

sys.path.insert(0, str(SRC_DIR))

from rag_assistant import ask_retention_assistant


def test_empty_question_is_rejected():
    with pytest.raises(ValueError, match="Question cannot be empty"):
        ask_retention_assistant("")


def test_whitespace_question_is_rejected():
    with pytest.raises(ValueError, match="Question cannot be empty"):
        ask_retention_assistant("   ")


@patch("rag_assistant.retrieve_context", return_value=[])
def test_no_retrieved_passages_returns_clear_message(mock_retrieve):
    result = ask_retention_assistant(
        "What retention action should we take?"
    )

    assert result["answer"] == (
        "No relevant retention-policy passages were found."
    )
    assert result["sources"] == []
    mock_retrieve.assert_called_once()


@patch("rag_assistant.retrieve_context", return_value=[])
def test_empty_retrieval_does_not_call_groq(mock_retrieve):
    with patch("rag_assistant.Groq") as mock_groq:
        result = ask_retention_assistant(
            "Suggest a retention action."
        )

    mock_groq.assert_not_called()
    assert result["sources"] == []
