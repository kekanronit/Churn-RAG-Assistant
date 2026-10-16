# Customer Churn Predictor + RAG Retention Assistant

An end-to-end machine learning and Generative AI project that predicts customer churn, explains the factors contributing to churn risk, and retrieves relevant retention guidance using Retrieval-Augmented Generation (RAG).

The application combines **Machine Learning, SQL, RAG, Large Language Models (LLMs), and Streamlit** in a unified Customer 360 dashboard.

## Overview

Customer churn is a major challenge for subscription-based businesses. Identifying customers who are likely to leave helps retention teams prioritize their efforts and make informed decisions.

This project provides a workflow to:

- Predict customer churn probability using a trained machine learning model.
- Classify customers into Low, Medium, and High risk categories.
- Explain the model features contributing positively to churn risk.
- Retrieve relevant passages from retention policy documents using semantic search.
- Generate policy-grounded responses using an LLM.
- Display customer information, risk assessments, and retention assistance in a Streamlit dashboard.

## Key Features

### 1. Customer Churn Prediction

- Trains a Logistic Regression classification pipeline using the Telco Customer Churn dataset.
- Uses preprocessing for numerical and categorical features.
- Generates a churn probability for an individual customer.
- Saves prediction records and model metadata in SQLite.

### 2. Risk Driver Explanation

- Analyzes feature contributions from the trained Logistic Regression model.
- Identifies the strongest positive contributions toward the churn prediction.
- Displays readable feature labels for easier interpretation.

### 3. Retention Knowledge Base

- Extracts text from PDF documents using PyPDF.
- Generates text embeddings using Sentence Transformers.
- Stores document chunks and metadata in ChromaDB.
- Retrieves relevant passages based on semantic similarity.

### 4. RAG Retention Assistant

- Combines retrieved policy passages with a user's question.
- Uses the Groq API to generate an answer.
- Returns supporting source filenames and page numbers.
- Instructs the model not to invent discounts, refunds, or eligibility rules unsupported by the retrieved policy.

### 5. Customer 360 Dashboard

Built with Streamlit, the dashboard brings the project components together.

It displays:
- Customer profile information.
- Churn probability and risk category.
- Model version.
- Top positive churn-risk feature contributions.
- A retention question interface.
- Generated answers and retrieved document sources.

## Technology Stack

| Category | Technologies |
|---|---|
| Programming Language | Python |
| Data Processing | Pandas |
| Database | SQLite |
| Machine Learning | Scikit-learn, Joblib |
| Data Visualization | Matplotlib |
| PDF Processing | PyPDF |
| Embeddings | Sentence Transformers |
| Vector Database | ChromaDB |
| Generative AI | Groq API |
| LLM Integration | Groq Python SDK |
| Frontend / Dashboard | Streamlit |
| Environment Configuration | python-dotenv |
| Version Control | Git and GitHub |

## System Architecture

```text
                  Customer Dataset
                         |
                         v
                  SQLite Database
                         |
                         v
                Customer Data Pipeline
                         |
                         v
               Churn Prediction Model
                 (Scikit-learn)
                         |
              +----------+----------+
              |                     |
              v                     v
       Churn Probability      Risk Driver Explanation
              |                     |
              +----------+----------+
                         |
                         v
                 Customer 360 UI
                    (Streamlit)
                         |
                         v
                 Retention Question
                         |
                         v
                 Semantic Retrieval
              (Sentence Transformers)
                         |
                         v
                     ChromaDB
                         |
                  Relevant Passages
                         |
                         v
                    Groq LLM API
                         |
                         v
              Policy-Grounded Response
                         |
                         v
                Answer with Sources
```

## Machine Learning Results

The initial Logistic Regression model was evaluated on a held-out test set.

| Metric | Result |
|---|---:|
| ROC-AUC | 0.8356 |
| Accuracy | 0.73 |
| Churn Precision | 0.49 |
| Churn Recall | 0.79 |
| Churn F1-score | 0.61 |

The test set contained 1,409 records, including 374 churned customers.

### Interpretation

- **ROC-AUC of 0.8356:** The model demonstrates useful discrimination between churned and retained customers.
- **Churn recall of 0.79:** The model identified approximately 79% of actual churn cases in the test set.
- **Churn precision of 0.49:** Some customers flagged as potential churners did not churn in the test set.

These results reflect the initial model evaluation and should not be interpreted as a guarantee of performance on future customer data.

### Risk Classification

The project uses the following thresholds:

| Churn Probability | Risk Level |
|---|---|
| Below 35% | Low |
| 35% to 65%, inclusive | Medium |
| Above 65% | High |

## Project Structure

```text
churn-rag-assistant/
│
├── data/
│   └── telco.csv                 # Local dataset; not committed
│
├── docs/
│   └── Sample_retention_policy.pdf
│
├── models/
│   └── churn_model.joblib        # Generated model; not committed
│
├── src/
│   ├── app.py                    # Streamlit Customer 360 dashboard
│   ├── db.py                     # Database utilities
│   ├── load_data.py              # Dataset loading
│   ├── repair_db.py              # Database repair utilities
│   ├── inspect_db.py             # Database inspection
│   ├── check_db.py               # Database validation
│   ├── generate_tickets.py       # Synthetic support ticket generation
│   ├── eda.py                    # Exploratory data analysis
│   ├── train.py                  # Model training and evaluation
│   ├── predict.py                # Churn prediction
│   ├── explain.py                # Risk driver explanation
│   ├── index_documents.py        # PDF indexing
│   ├── retrieve.py               # Semantic retrieval
│   └── rag_assistant.py          # RAG and LLM integration
│
├── tests/                        # Automated tests, if present
├── vectorstore/                  # Local ChromaDB data; not committed
├── requirements.txt
├── .gitignore
└── README.md
```

Some generated files and optional folders may be created locally when running the project. The trained model and vector database can be regenerated using the appropriate scripts.

## Getting Started

### Prerequisites

- Python 3.10 or later, subject to compatibility of the installed dependencies.
- Git.
- A Groq API key for generating LLM responses.
- The Telco Customer Churn dataset.
- A retention policy PDF for indexing.

### 1. Clone the Repository

```bash
git clone https://github.com/KekanRonit/churn-rag-assistant.git
cd churn-rag-assistant
```

Replace the repository URL if your GitHub repository uses a different name or URL.

### 2. Create a Virtual Environment

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Install Dependencies

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Ensure `requirements.txt` includes the dependencies used by the application, including:

```text
pandas
scikit-learn
matplotlib
joblib
pypdf
sentence-transformers
chromadb
groq
python-dotenv
streamlit
```

### 4. Configure Environment Variables

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_api_key_here
```

Replace the placeholder with your own API key.

**Security:** Never commit `.env` or publish your API key. The application loads this value through `python-dotenv`.

### 5. Prepare the Dataset and Database

Place the Telco Customer Churn CSV file at:

```text
data/telco.csv
```

Use the project's database and data-loading scripts to create and populate the SQLite database. For example:

```powershell
python src/load_data.py
```

If the database has not been initialized, follow the setup instructions implemented in `src/db.py` and the relevant scripts.

### 6. Train the Churn Model

```powershell
python src/train.py
```

The trained model is saved locally as:

```text
models/churn_model.joblib
```

The prediction and explanation modules require this model file.

### 7. Index the Retention Documents

Place your permitted retention policy PDF in the `docs/` directory.

Then run:

```powershell
python src/index_documents.py
```

This creates or updates the persistent ChromaDB knowledge base used for retrieval.

### 8. Run the Streamlit Dashboard

From the project root:

```powershell
python -m streamlit run src/app.py
```

Open the local URL displayed in the terminal, usually:

```text
http://localhost:8501
```

Select a customer, inspect the churn assessment, and ask the retention assistant a question.

## Example Use Case

Consider a customer with a month-to-month contract, relatively short tenure, and elevated monthly charges.

The application can:

1. Retrieve the customer's record from SQLite.
2. Estimate churn probability using the trained model.
3. Assign a risk category using the configured thresholds.
4. Display the strongest positive model feature contributions.
5. Retrieve relevant retention guidance from indexed documents.
6. Generate an answer using the retrieved context and the Groq API.
7. Display the response and its supporting source references.

The system supports retention analysis; it does not guarantee that a customer will churn or that a particular intervention will prevent churn.

## Data and Evaluation Notes

- The customer churn model uses the Telco Customer Churn dataset.
- Synthetic support tickets are generated for development and demonstration.
- The sample retention policy is synthetic and intended for testing the document retrieval and RAG workflow.
- The application should not be presented as using a real company's confidential retention policy unless such a policy has actually been supplied and authorized for use.
- Feature contributions are model-based associations, not proof of causation.
- The initial evaluation metrics are based on a held-out test set. Further validation and monitoring are needed before real-world deployment.

## Limitations and Future Improvements

Potential next steps include:

- Add automated tests for predictions, database operations, and retrieval.
- Prevent duplicate prediction records during Streamlit reruns.
- Cache expensive model and embedding initialization.
- Evaluate retrieval relevance and answer faithfulness systematically.
- Improve the dashboard with customer search, analytics, and historical prediction views.
- Add robust logging and API error handling.
- Explore model calibration and threshold selection based on business costs.
- Add authentication and access control before exposing customer data in a deployed application.
- Deploy the application using an appropriate hosting environment.

## What I Learned

This project brings together several areas of software and AI engineering:

- Preparing and querying structured data with SQL and SQLite.
- Building and evaluating a machine learning classification pipeline.
- Explaining model predictions through feature contributions.
- Processing PDF documents and creating semantic embeddings.
- Implementing retrieval-augmented generation with a vector database and an LLM API.
- Integrating machine learning and generative AI components into a Streamlit application.
- Managing environment variables, dependencies, and source code using Git.

## Author

**Ronit Kekan**

BCA Graduate | MCA Student | Software Development and AI/ML Enthusiast

- GitHub: [KekanRonit](https://github.com/KekanRonit)
- LinkedIn: [Ronit Kekan](https://www.linkedin.com/in/ronitkekan79/)

---

*Built as an end-to-end learning project exploring Machine Learning, Retrieval-Augmented Generation, and customer retention analytics.*
