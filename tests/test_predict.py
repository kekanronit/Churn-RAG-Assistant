
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"

sys.path.insert(0, str(SRC_DIR))

from predict import score_customer


def test_prediction_returns_valid_probability():
    result = score_customer("7590-VHVEG")

    assert 0.0 <= result["churn_probability"] <= 1.0


def test_prediction_returns_valid_risk_level():
    result = score_customer("7590-VHVEG")

    probability = result["churn_probability"]

    if probability < 0.35:
        expected = "Low"
    elif probability <= 0.65:
        expected = "Medium"
    else:
        expected = "High"

    assert result["risk_level"] == expected


def test_prediction_contains_required_fields():
    result = score_customer("7590-VHVEG")

    assert result["customer_id"] == "7590-VHVEG"
    assert "churn_probability" in result
    assert "risk_level" in result
    assert "model_version" in result


def test_prediction_rejects_unknown_customer():
    with pytest.raises(ValueError, match="Customer not found"):
        score_customer("INVALID-CUSTOMER-ID")
