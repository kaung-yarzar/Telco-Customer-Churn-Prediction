import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

model = joblib.load("churn_model.joblib")

app = FastAPI(title="Telco Customer Churn API", version="1.0")


class Customer(BaseModel):
    """19 features the model"""
    gender: str = Field(..., examples=["Female"])
    SeniorCitizen: int = Field(..., ge=0, le=1, examples=[0])
    Partner: str = Field(..., examples=["Yes"])
    Dependents: str = Field(..., examples=["No"])
    tenure: int = Field(..., ge=0, examples=[1])
    PhoneService: str = Field(..., examples=["No"])
    MultipleLines: str = Field(..., examples=["No phone service"])
    InternetService: str = Field(..., examples=["DSL"])
    OnlineSecurity: str = Field(..., examples=["No"])
    OnlineBackup: str = Field(..., examples=["Yes"])
    DeviceProtection: str = Field(..., examples=["No"])
    TechSupport: str = Field(..., examples=["No"])
    StreamingTV: str = Field(..., examples=["No"])
    StreamingMovies: str = Field(..., examples=["No"])
    Contract: str = Field(..., examples=["Month-to-month"])
    PaperlessBilling: str = Field(..., examples=["Yes"])
    PaymentMethod: str = Field(..., examples=["Electronic check"])
    MonthlyCharges: float = Field(..., ge=0, examples=[29.85])
    TotalCharges: float = Field(..., ge=0, examples=[29.85])


@app.get("/health")
def health():
    """Simple liveness check."""
    return {"status": "ok"}


@app.post("/predict")
def predict(customer: Customer):
    """Accept customer details and return the churn prediction."""
    df = pd.DataFrame([customer.model_dump()])
    proba = float(model.predict_proba(df)[0, 1])
    band = "high" if proba >= 0.66 else "medium" if proba >= 0.33 else "low"
    return {
        "churn": bool(proba >= 0.5),
        "churn_probability": round(proba, 4),
        "risk_band": band,
    }
