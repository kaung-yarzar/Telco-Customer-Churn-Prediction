# Telco Customer Churn Prediction

Predicts which telecom customers are likely to **churn**
using the [Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
dataset (7,043 customers, 21 columns).

The project covers the full workflow: **data analysis → model training → serving
the model via an API and a web UI.**

---

## Repository structure

```
.
├── Data                  # Dataset
├── Churn_Predict.ipynb   # Data analysis & model training notebook
├── app.py                # FastAPI prediction API (JSON endpoint)
├── streamlit_app.py      # Streamlit web UI (interactive form)
├── churn_model.joblib    # Trained model (created by the notebook)
├── requirements.txt      # Python dependencies
└── README.md
```

---

## Quickstart

```bash
# 1. Create a virtual environment and install dependencies
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 2. (Optional) Re-run analysis.ipynb to regenerate churn_model.joblib
#    The notebook loads the data, cleans it, trains the model, and saves it.

# 3. Run the API
uvicorn app:app --reload           # http://127.0.0.1:8000/docs

# 4. Or run the web UI
streamlit run streamlit_app.py     # http://localhost:8501
```

---

## 1. Data Analysis (`analysis.ipynb`)

The notebook inspects the data and identifies the factors that correlate most
with churn.

**Cleaning steps**
- `customerID` dropped (an identifier, no predictive value).
- `TotalCharges` was stored as text with 11 blank values (new customers,
  `tenure = 0`) — converted to numeric and those rows dropped.
- 22 duplicate rows removed.
- Target `Churn` mapped from `Yes`/`No` to `1`/`0`.

**Key findings — what drives churn**

| Factor | Effect |
|--------|--------|
| **Contract = Month-to-month** | Strongest driver — ~43% churn vs ~3% for two-year contracts |
| **InternetService = Fiber optic** | ~42% churn vs ~19% for DSL |
| **Low tenure** | New customers churn most (avg 18 vs 38 months) |
| **PaymentMethod = Electronic check** | ~45% churn vs ~15% for automatic methods |
| **Higher MonthlyCharges** | Churners pay ~$13 more per month on average |

**Class imbalance:** only ~27% of customers churn, so the project evaluates with
**F1-score and ROC-AUC**, not accuracy.

Visualizations include: churn distribution bar chart, churn-rate-by-category bar
charts, tenure/charges histograms split by churn, and a correlation chart.

---

## 2. Model Training

A scikit-learn **Pipeline** is used so preprocessing and the model
together:

```
ColumnTransformer( OneHotEncoder + StandardScaler )  ->  XGBClassifier
```

- **Encoding:** categorical features one-hot encoded; numeric features scaled.
- **Class imbalance** handled with XGBoost's `scale_pos_weight` (ratio of
  negative to positive classes ≈ 2.8).
- **Validation:** 5-fold `StratifiedKFold` cross-validation on the training set.
- The whole pipeline is saved with `joblib` as `churn_model.joblib`.

**Performance (held-out test set)**

| Metric | Score |
|--------|-------|
| ROC-AUC | **0.84** |
| F1 (churn class) | 0.62 |
| Recall (churn class) | **0.77** |
| Precision (churn class) | 0.51 |

Recall is prioritized: missing a customer who is about to churn costs more than a
false alarm, so the model is tuned to catch ~77% of actual churners.

---

## 3. Prediction API (`app.py`)

A **FastAPI** service loads `churn_model.joblib` and serves predictions.

**Run it**
```bash
uvicorn app:app --reload
```

**Endpoints**

| Method | Path | Description |
|--------|------|-------------|
| `GET`  | `/health`  | Liveness check |
| `POST` | `/predict` | Predict churn for one customer |

Interactive docs: **http://127.0.0.1:8000/docs**

**Example request**
```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "gender": "Female", "SeniorCitizen": 0, "Partner": "No", "Dependents": "No",
    "tenure": 1, "PhoneService": "Yes", "MultipleLines": "No",
    "InternetService": "Fiber optic", "OnlineSecurity": "No", "OnlineBackup": "No",
    "DeviceProtection": "No", "TechSupport": "No", "StreamingTV": "No",
    "StreamingMovies": "No", "Contract": "Month-to-month", "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check", "MonthlyCharges": 85.0, "TotalCharges": 85.0
  }'
```

**Example response**
```json
{
  "churn": true,
  "churn_probability": 0.9521,
  "risk_band": "high"
}
```

Input is validated with **Pydantic** — invalid payloads are rejected with a
clear `422` error before reaching the model.

---

## 4. Web UI (`streamlit_app.py`)

A **Streamlit** app for testing the model through a form.

```bash
streamlit run streamlit_app.py
```

Fill in the customer's details, click **Predict**, and see the churn probability risk band (low / medium / high).

---

## Tech stack

`pandas` · `scikit-learn` · `xgboost` · `matplotlib` / `seaborn` ·
`FastAPI` · `Streamlit` · `joblib`

---

## Future improvements

In Future, the following would be added:
- Model versioning and a model registry.
- Monitoring for **data drift** (churn patterns change over time) and scheduled
  retraining.
- Request logging, authentication, and rate limiting.
- Containerization (Docker) with multiple workers.
