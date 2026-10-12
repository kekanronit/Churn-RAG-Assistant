
import sqlite3
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

ROOT_DIR = Path(__file__).resolve().parent.parent
DB_PATH = ROOT_DIR / "data" / "churn.db"


def run_eda():
    with sqlite3.connect(DB_PATH) as conn:
        df = pd.read_sql_query(
            "SELECT * FROM customers", conn
        )

    print("\n--- Dataset overview ---")
    print("Rows:", df.shape[0])
    print("Columns:", df.shape[1])

    print("\n--- Column names ---")
    print(df.columns.tolist())

    print("\n--- Missing values ---")
    print(df.isnull().sum())

    print("\n--- Churn distribution ---")
    print(df["churn_label"].value_counts())

    churn_rate = (df["churn_label"] == "Yes").mean()
    print(f"\nChurn rate: {churn_rate:.2%}")

    print("\n--- Contract vs churn ---")
    print(pd.crosstab(
        df["contract"],
        df["churn_label"],
        normalize="index"
    ).round(3))

    print("\n--- Monthly charges by churn ---")
    print(df.groupby("churn_label")["monthly_charges"].agg(
        ["mean", "median", "min", "max"]
    ).round(2))

    print("\n--- Tenure by churn ---")
    print(df.groupby("churn_label")["tenure_months"].agg(
        ["mean", "median", "min", "max"]
    ).round(2))

    # Plot churn distribution
    counts = df["churn_label"].value_counts().reindex(
        ["No", "Yes"]
    )

    counts.plot(kind="bar", rot=0)
    plt.title("Customer Churn Distribution")
    plt.xlabel("Churn")
    plt.ylabel("Number of Customers")
    plt.tight_layout()

    output_path = ROOT_DIR / "data" / "churn_distribution.png"
    plt.savefig(output_path)
    plt.close()

    print(f"\nChart saved to: {output_path}")


if __name__ == "__main__":
    run_eda()
