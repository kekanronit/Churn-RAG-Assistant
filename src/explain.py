
import sqlite3
from pathlib import Path

import joblib
import pandas as pd


ROOT_DIR = Path(__file__).resolve().parent.parent
DB_PATH = ROOT_DIR / "data" / "churn.db"
MODEL_PATH = ROOT_DIR / "models" / "churn_model.joblib"


FEATURE_LABELS = {
    "numeric__tenure_months": "Customer tenure",
    "categorical__contract_Month-to-month":
        "Contract type: Month-to-month",
    "categorical__payment_method_Electronic check":
        "Payment method: Electronic check",
    "numeric__monthly_charges": "Monthly charges",
    "categorical__paperless_billing_Yes":
        "Paperless billing enabled",
}


def explain_customer(customer_id: str, top_n: int = 5) -> dict:
    """
    Explain the strongest positive feature contributions
    for a customer using the trained Logistic Regression model.
    """

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Trained model not found: {MODEL_PATH}. "
            "Run src/train.py first."
        )

    if top_n < 1:
        raise ValueError("top_n must be at least 1.")

    model = joblib.load(MODEL_PATH)

    with sqlite3.connect(DB_PATH) as conn:
        customer = pd.read_sql_query(
            "SELECT * FROM customers WHERE customer_id = ?",
            conn,
            params=(customer_id,),
        )

    if customer.empty:
        raise ValueError(f"Customer not found: {customer_id}")

    # Use the same input features that were used during training.
    features = customer.drop(
        columns=["customer_id", "churn_label"],
        errors="ignore",
    )

    preprocessor = model.named_steps["preprocessor"]
    classifier = model.named_steps["classifier"]

    # Transform the customer's data using the fitted preprocessing pipeline.
    transformed = preprocessor.transform(features)

    if hasattr(transformed, "toarray"):
        transformed = transformed.toarray()

    feature_names = preprocessor.get_feature_names_out()
    coefficients = classifier.coef_[0]

    # Each contribution is in log-odds space.
    contributions = transformed[0] * coefficients

    positive_drivers = []

    for name, contribution in zip(feature_names, contributions):
        contribution = float(contribution)

        # Only include contributions that push the prediction
        # toward the churn class.
        if contribution > 0:
            feature_name = str(name)

            positive_drivers.append(
                {
                    "feature": feature_name,
                    "label": FEATURE_LABELS.get(
                        feature_name,
                        feature_name,
                    ),
                    "contribution": contribution,
                }
            )

    positive_drivers.sort(
        key=lambda item: item["contribution"],
        reverse=True,
    )

    probability = float(model.predict_proba(features)[0][1])

    if probability < 0.35:
        risk_level = "Low"
    elif probability <= 0.65:
        risk_level = "Medium"
    else:
        risk_level = "High"

    return {
        "customer_id": customer_id,
        "churn_probability": probability,
        "risk_level": risk_level,
        "top_drivers": positive_drivers[:top_n],
        "explanation_method": "Logistic Regression feature contributions",
    }


if __name__ == "__main__":
    result = explain_customer("7590-VHVEG", top_n=5)

    print("\n--- Customer Risk Explanation ---")
    print(f"Customer ID: {result['customer_id']}")
    print(
        f"Churn probability: "
        f"{result['churn_probability']:.2%}"
    )
    print(f"Risk level: {result['risk_level']}")
    print(f"Explanation method: {result['explanation_method']}")

    print("\nTop churn risk drivers:")

    if not result["top_drivers"]:
        print("No positive feature contributions were found.")
    else:
        for i, driver in enumerate(result["top_drivers"], start=1):
            print(f"{i}. {driver['label']}")
            print(
                f"   Model contribution: "
                f"+{driver['contribution']:.4f}"
            )
