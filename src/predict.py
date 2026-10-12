
import sqlite3
from pathlib import Path

import joblib
import pandas as pd


ROOT_DIR = Path(__file__).resolve().parent.parent
DB_PATH = ROOT_DIR / "data" / "churn.db"
MODEL_PATH = ROOT_DIR / "models" / "churn_model.joblib"

MODEL_VERSION = "logistic-regression-v1"


def score_customer(customer_id):
    """Predict churn risk and save the prediction to SQLite."""

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "Model not found. Run python src/train.py first."
        )

    model = joblib.load(MODEL_PATH)

    with sqlite3.connect(DB_PATH) as conn:
        customer = pd.read_sql_query(
            "SELECT * FROM customers WHERE customer_id = ?",
            conn,
            params=(customer_id,),
        )

        if customer.empty:
            raise ValueError(
                f"Customer not found: {customer_id}"
            )

        # Use exactly the features used during training.
        features = customer.drop(
            columns=["customer_id", "churn_label"]
        )

        probability = float(
            model.predict_proba(features)[0, 1]
        )

        # Thresholds from the project design document.
        if probability < 0.35:
            risk_level = "Low"
        elif probability <= 0.65:
            risk_level = "Medium"
        else:
            risk_level = "High"

        conn.execute("""
            INSERT INTO predictions (
                customer_id,
                churn_probability,
                risk_level,
                top_drivers,
                model_version
            )
            VALUES (?, ?, ?, ?, ?)
        """, (
            customer_id,
            probability,
            risk_level,
            None,
            MODEL_VERSION,
        ))

    result = {
        "customer_id": customer_id,
        "churn_probability": round(probability, 4),
        "risk_level": risk_level,
        "model_version": MODEL_VERSION,
    }

    return result


if __name__ == "__main__":
    result = score_customer("7590-VHVEG")

    print("\n--- Customer Churn Prediction ---")
    for key, value in result.items():
        print(f"{key}: {value}")

