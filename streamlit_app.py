import joblib
import pandas as pd
import streamlit as st

# Load the trained pipeline once (cached so it doesn't reload on every click).
@st.cache_resource
def load_model():
    return joblib.load("churn_model.joblib")

model = load_model()

st.title("Telco Customer Churn Predictoion")
st.write("Fill in the customer's details and click **Predict** to see their churn risk.")

# Input form
col1, col2, col3 = st.columns(3)

with col1:
    gender = st.selectbox("Gender", ["Female", "Male"])
    SeniorCitizen = st.selectbox("Senior Citizen", ["No", "Yes"])
    Partner = st.selectbox("Partner", ["Yes", "No"])
    Dependents = st.selectbox("Dependents", ["Yes", "No"])
    tenure = st.slider("Tenure (months)", 0, 72, 12)
    PhoneService = st.selectbox("Phone Service", ["Yes", "No"])
    MultipleLines = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])

with col2:
    InternetService = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    OnlineSecurity = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
    OnlineBackup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
    DeviceProtection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
    TechSupport = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
    StreamingTV = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
    StreamingMovies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])

with col3:
    Contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    PaperlessBilling = st.selectbox("Paperless Billing", ["Yes", "No"])
    PaymentMethod = st.selectbox("Payment Method", [
        "Electronic check", "Mailed check",
        "Bank transfer (automatic)", "Credit card (automatic)",
    ])
    MonthlyCharges = st.number_input("Monthly Charges", 0.0, 200.0, 70.0)
    TotalCharges = st.number_input("Total Charges", 0.0, 10000.0, 1000.0)

# Predict
if st.button("Predict", type="primary"):
    customer = {
        # model was trained on 0/1, convert to Yes/No back
        "gender": gender, "SeniorCitizen": 1 if SeniorCitizen == "Yes" else 0,
        "Partner": Partner,
        "Dependents": Dependents, "tenure": tenure, "PhoneService": PhoneService,
        "MultipleLines": MultipleLines, "InternetService": InternetService,
        "OnlineSecurity": OnlineSecurity, "OnlineBackup": OnlineBackup,
        "DeviceProtection": DeviceProtection, "TechSupport": TechSupport,
        "StreamingTV": StreamingTV, "StreamingMovies": StreamingMovies,
        "Contract": Contract, "PaperlessBilling": PaperlessBilling,
        "PaymentMethod": PaymentMethod, "MonthlyCharges": MonthlyCharges,
        "TotalCharges": TotalCharges,
    }
    df = pd.DataFrame([customer])
    proba = float(model.predict_proba(df)[0, 1])

    st.subheader("Result")
    st.metric("Churn probability", f"{proba:.1%}")
    st.progress(proba)

    if proba >= 0.66:
        st.error("🔴 HIGH risk — this customer is likely to churn.")
    elif proba >= 0.33:
        st.warning("🟡 MEDIUM risk — keep an eye on this customer.")
    else:
        st.success("🟢 LOW risk — this customer is likely to stay.")
