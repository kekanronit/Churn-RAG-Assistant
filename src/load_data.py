
import sqlite3
from pathlib import Path

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT_DIR / "data" / "telco.csv"
DB_PATH = ROOT_DIR / "data" / "churn.db"


def load_customers():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)

    # Match the customer columns to our database design.
    
    df = df.rename(columns={
    "customerID": "customer_id",
    "SeniorCitizen": "senior_citizen",
    "Partner": "partner",
    "Dependents": "dependents",
    "tenure": "tenure_months",
    "Contract": "contract",
    "PaymentMethod": "payment_method",
    "PaperlessBilling": "paperless_billing",
    "InternetService": "internet_service",
    "MonthlyCharges": "monthly_charges",
    "TotalCharges": "total_charges",
    "Churn": "churn_label",
})


    required_columns = [
        "customer_id", "gender", "senior_citizen",
        "partner", "dependents", "tenure_months",
        "contract", "payment_method", "paperless_billing",
        "internet_service", "monthly_charges",
        "total_charges", "churn_label",
    ]

    missing = set(required_columns) - set(df.columns)
    if missing:
        raise ValueError(f"Missing CSV columns: {sorted(missing)}")

    # Convert numeric values safely.
    df["tenure_months"] = pd.to_numeric(
        df["tenure_months"], errors="raise"
    ).astype(int)

    df["monthly_charges"] = pd.to_numeric(
        df["monthly_charges"], errors="raise"
    )

    df["total_charges"] = pd.to_numeric(
        df["total_charges"], errors="coerce"
    ).fillna(0.0)

    if df["customer_id"].isna().any():
        raise ValueError("Customer IDs cannot be empty.")

    if df["customer_id"].duplicated().any():
        raise ValueError("Duplicate customer IDs found.")

    with sqlite3.connect(DB_PATH) as conn:
        df[required_columns].to_sql(
            "customers",
            conn,
            if_exists="replace",
            index=False,
        )

        count = conn.execute(
            "SELECT COUNT(*) FROM customers"
        ).fetchone()[0]

    print(f"Customers loaded successfully: {count}")
    print(f"Database: {DB_PATH}")


if __name__ == "__main__":
    load_customers()

