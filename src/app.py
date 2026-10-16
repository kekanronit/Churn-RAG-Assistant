
import sqlite3
import sys
from pathlib import Path

import pandas as pd
import streamlit as st


# =========================================================
# 1. PROJECT CONFIGURATION
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DB_PATH = PROJECT_ROOT / "data" / "churn.db"

# Make modules in src importable
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from predict import score_customer
from explain import explain_customer
from rag_assistant import ask_retention_assistant


st.set_page_config(
    page_title="Customer 360 | Churn Intelligence",
    page_icon="📊",
    layout="wide",
)

st.title("📊 Customer 360")
st.caption(
    "Customer Churn Predictor + RAG Retention Assistant"
)


# =========================================================
# 2. LOAD CUSTOMER DATA
# =========================================================

@st.cache_data
def load_customers():
    """Load customer records from the SQLite database."""

    if not DB_PATH.exists():
        raise FileNotFoundError(
            f"Database not found: {DB_PATH}"
        )

    with sqlite3.connect(DB_PATH) as connection:
        return pd.read_sql_query(
            """
            SELECT *
            FROM customers
            ORDER BY customer_id
            """,
            connection,
        )


# =========================================================
# 3. CUSTOMER PROFILE
# =========================================================

try:
    customers = load_customers()

    if customers.empty:
        st.warning("No customers were found in the database.")
        st.stop()

    customer_ids = customers["customer_id"].astype(str).tolist()

    selected_id = st.sidebar.selectbox(
        "Select Customer ID",
        customer_ids,
    )

    customer = customers.loc[
        customers["customer_id"].astype(str) == selected_id
    ].iloc[0]

    st.sidebar.markdown("---")
    st.sidebar.caption("Customer Churn Intelligence")
    st.sidebar.caption(f"Selected ID: {selected_id}")

    st.header("👤 Customer Profile")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Customer ID",
        selected_id,
    )

    col2.metric(
        "Tenure",
        f"{int(customer['tenure_months'])} months",
    )

    col3.metric(
        "Monthly Charges",
        f"${float(customer['monthly_charges']):,.2f}",
    )

    col4.metric(
        "Contract",
        str(customer["contract"]),
    )

    with st.expander("View More Customer Details"):
        detail_col1, detail_col2 = st.columns(2)

        with detail_col1:
            st.write("**Payment Method**")
            st.write(
                customer.get("payment_method", "Not available")
            )

            st.write("**Internet Service**")
            st.write(
                customer.get("internet_service", "Not available")
            )

        with detail_col2:
            st.write("**Paperless Billing**")
            st.write(
                customer.get("paperless_billing", "Not available")
            )

            st.write("**Total Charges**")
            st.write(
                customer.get("total_charges", "Not available")
            )

    st.divider()


    # =====================================================
    # 4. CHURN RISK ASSESSMENT
    # =====================================================

    st.header("📈 Churn Risk Assessment")

    with st.spinner("Calculating customer churn risk..."):
        prediction = score_customer(selected_id)
        explanation = explain_customer(
            selected_id,
            top_n=5,
        )

    probability = float(
        prediction["churn_probability"]
    )

    risk_level = prediction["risk_level"]

    metric1, metric2, metric3 = st.columns(3)

    metric1.metric(
        "Churn Probability",
        f"{probability:.2%}",
    )

    metric2.metric(
        "Risk Level",
        risk_level,
    )

    metric3.metric(
        "Model Version",
        prediction["model_version"],
    )

    st.write("**Churn Probability Indicator**")

    st.progress(
        min(max(probability, 0.0), 1.0)
    )

    if risk_level == "High":
        st.error(
            "High risk: review this customer's retention needs."
        )

    elif risk_level == "Medium":
        st.warning(
            "Medium risk: review customer concerns and "
            "consider appropriate retention actions."
        )

    else:
        st.success(
            "Low risk: continue regular customer engagement."
        )

    st.caption(
        "Risk thresholds: Low < 35%; "
        "Medium 35%–65%; High > 65%."
    )

    st.divider()


    # =====================================================
    # 5. RISK DRIVER EXPLANATION
    # =====================================================

    st.header("🔍 Why Is This Customer at Risk?")

    st.caption(
        "The following are positive Logistic Regression feature "
        "contributions in log-odds space. They are not causal "
        "explanations or percentage-point increases."
    )

    drivers = explanation["top_drivers"]

    if drivers:
        for index, driver in enumerate(drivers, start=1):
            st.write(
                f"**{index}. {driver['label']}**"
                f" — contribution: "
                f"+{driver['contribution']:.4f}"
            )

    else:
        st.info(
            "No positive churn feature contributions were identified."
        )

    st.caption(
        f"Explanation method: "
        f"{explanation['explanation_method']}"
    )

    st.divider()


    # =====================================================
    # 6. RAG RETENTION ASSISTANT
    # =====================================================

    st.header("🤖 RAG Retention Assistant")

    st.write(
        "Ask questions about retention actions using your "
        "indexed retention policy documents."
    )

    st.info(
        "The assistant uses the selected customer's profile "
        "and churn assessment to understand your question. "
        "Recommendations must be supported by retrieved policy "
        "documents."
    )

    # Customer information supplied as scenario context
    customer_context = (
        f"Customer ID: {selected_id}\n"
        f"Contract: {customer['contract']}\n"
        f"Tenure: {customer['tenure_months']} months\n"
        f"Monthly charges: {customer['monthly_charges']}\n"
        f"Payment method: {customer['payment_method']}\n"
        f"Internet service: {customer['internet_service']}\n"
        f"Churn probability: {probability:.2%}\n"
        f"Risk level: {risk_level}"
    )

    question = st.text_area(
        "Ask a retention question",
        placeholder=(
            "Example: What should we review for this "
            "high-risk customer before choosing a retention action?"
        ),
        key="retention_question",
    )

    ask_button = st.button(
        "Get Retention Recommendations",
        type="primary",
    )

    if ask_button:
        if not question.strip():
            st.warning(
                "Please enter a question before continuing."
            )

        else:
            combined_question = (
                "Customer profile and churn assessment:\n"
                f"{customer_context}\n\n"
                "User's retention question:\n"
                f"{question.strip()}\n\n"
                "Instructions for this scenario: use the customer "
                "profile to understand the situation. Base policy "
                "recommendations only on retrieved retention-policy "
                "passages. Do not invent discounts, refunds, "
                "eligibility rules, or company policies. If the "
                "retrieved policy does not support a recommendation, "
                "state that limitation clearly."
            )

            try:
                with st.spinner(
                    "Retrieving policy documents and generating "
                    "recommendations..."
                ):
                    result = ask_retention_assistant(
                        combined_question
                    )

                st.subheader("💡 Retention Recommendations")

                st.write(result["answer"])

                st.subheader("📚 Retrieved Sources")

                # Deduplicate source filename and page combinations
                unique_sources = sorted({
                    (
                        source["source"],
                        source["page"],
                    )
                    for source in result.get("sources", [])
                })

                if unique_sources:
                    for source_name, page_number in unique_sources:
                        st.markdown(
                            f"- `{source_name}` — page {page_number}"
                        )

                else:
                    st.info(
                        "No relevant policy sources were retrieved. "
                        "Do not treat the response as a verified "
                        "policy recommendation."
                    )

            except Exception as rag_error:
                st.error(
                    "The retention assistant could not generate "
                    f"a response: {rag_error}"
                )

    st.divider()

    st.caption(
        "Customer 360 | Churn predictions are model estimates, "
        "not guarantees of customer behavior."
    )


# =========================================================
# 7. APPLICATION ERROR HANDLING
# =========================================================

except Exception as error:
    st.error(
        f"Unable to display the Customer 360 dashboard: {error}"
    )

    st.info(
        "Check that data/churn.db exists, the customers table "
        "contains the expected columns, and "
        "models/churn_model.joblib is available."
    )
