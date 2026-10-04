"""
Streamlit UI: collects raw customer details, sends them to the FastAPI
/predict endpoint, and shows the churn probability and decision.

The API address comes from the API_URL environment variable
(default http://127.0.0.1:9123), so the same file works locally,
in Docker, and on AWS.
"""
import os

import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://127.0.0.1:9123").rstrip("/")

st.set_page_config(page_title="Telco Churn Predictor", page_icon="📉", layout="centered")
st.title("📉 Telco Customer Churn Predictor")
st.caption("Enter a customer's details to estimate the chance they will leave.")

YES_NO = ["No", "Yes"]

# ---------- Account ----------
st.subheader("Account")
c1, c2 = st.columns(2)
with c1:
    contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    tenure = st.number_input("Tenure (months)", min_value=0, max_value=100, value=12)
    paperless = st.selectbox("Paperless billing", YES_NO, index=1)
with c2:
    payment = st.selectbox("Payment method", [
        "Electronic check", "Mailed check",
        "Bank transfer (automatic)", "Credit card (automatic)"])
    monthly = st.number_input("Monthly charges", min_value=0.0, max_value=500.0,
                              value=70.0, step=0.05)
    know_total = st.checkbox("I know the total charges")
    total = st.number_input("Total charges", min_value=0.0, value=float(tenure * monthly),
                            step=1.0, disabled=not know_total)

# ---------- Demographics ----------
st.subheader("Demographics")
d1, d2, d3, d4 = st.columns(4)
gender = d1.selectbox("Gender", ["Female", "Male"])
senior = d2.selectbox("Senior citizen", YES_NO)
partner = d3.selectbox("Partner", YES_NO)
dependents = d4.selectbox("Dependents", YES_NO)

# ---------- Services ----------
st.subheader("Services")
s1, s2 = st.columns(2)
with s1:
    phone = st.selectbox("Phone service", ["Yes", "No"])
    if phone == "No":
        multiple = "No phone service"
        st.text_input("Multiple lines", value=multiple, disabled=True)
    else:
        multiple = st.selectbox("Multiple lines", YES_NO)
with s2:
    internet = st.selectbox("Internet service", ["Fiber optic", "DSL", "No"])

addon_names = ["OnlineSecurity", "OnlineBackup", "DeviceProtection",
               "TechSupport", "StreamingTV", "StreamingMovies"]
addons = {}
cols = st.columns(3)
for i, name in enumerate(addon_names):
    label = name.replace("Online", "Online ").replace("Device", "Device ") \
                .replace("Tech", "Tech ").replace("Streaming", "Streaming ")
    with cols[i % 3]:
        if internet == "No":
            addons[name] = "No internet service"
            st.text_input(label, value="No internet service", disabled=True, key=name)
        else:
            addons[name] = st.selectbox(label, YES_NO, key=name)

# ---------- Predict ----------
st.divider()
if st.button("Predict churn", type="primary", use_container_width=True):
    payload = {
        "gender": gender,
        "SeniorCitizen": 1 if senior == "Yes" else 0,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": int(tenure),
        "PhoneService": phone,
        "MultipleLines": multiple,
        "InternetService": internet,
        **addons,
        "Contract": contract,
        "PaperlessBilling": paperless,
        "PaymentMethod": payment,
        "MonthlyCharges": float(monthly),
        "TotalCharges": float(total) if know_total else None,
    }
    try:
        r = requests.post(f"{API_URL}/predict", json=payload, timeout=15)
    except requests.exceptions.ConnectionError:
        st.error(f"Cannot reach the API at {API_URL}. Is it running?")
        st.stop()
    except requests.exceptions.Timeout:
        st.error("The API took too long to respond.")
        st.stop()

    if r.status_code != 200:
        st.error(f"API error {r.status_code}")
        st.json(r.json())
        st.stop()

    res = r.json()
    prob = res["churn_probability"]

    m1, m2 = st.columns(2)
    m1.metric("Churn probability", f"{prob:.1%}")
    m2.metric("Decision threshold", f"{res['threshold']:.1%}")
    st.progress(min(max(prob, 0.0), 1.0))

    if res["prediction"] == "Churn":
        st.error("⚠️ Likely to churn. Consider a retention offer.")
    else:
        st.success("✅ Unlikely to churn.")

    if res["total_charges_estimated"]:
        st.info("Total charges were not provided, so they were estimated as "
                "tenure × monthly charges.")