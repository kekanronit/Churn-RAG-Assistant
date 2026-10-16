
import sys
from pathlib import Path

import pytest

# Add the project's src directory to Python's import path.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"

sys.path.insert(0, str(SRC_DIR))

from explain import explain_customer


def test_explanation_returns_valid_probability():
    result = explain_customer("7590-VHVEG")

    assert 0.0 <= result["churn_probability"] <= 1.0


def test_explanation_returns_valid_risk_level():
    result = explain_customer("7590-VHVEG")

    probability = result["churn_probability"]

    if probability < 0.35:
        expected = "Low"
    elif probability <= 0.65:
        expected = "Medium"
    else:
        expected = "High"

    assert result["risk_level"] == expected


def test_explanation_has_expected_structure():
    result = explain_customer("7590-VHVEG")

    assert result["customer_id"] == "7590-VHVEG"
    assert "top_drivers" in result
    assert "explanation_method" in result

    for driver in result["top_drivers"]:
        assert "label" in driver
        assert "contribution" in driver
        assert driver["contribution"] > 0


def test_explanation_respects_top_n():
    result = explain_customer("7590-VHVEG", top_n=3)

    assert len(result["top_drivers"]) <= 3
