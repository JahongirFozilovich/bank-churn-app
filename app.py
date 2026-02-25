import streamlit as st
import numpy as np
import joblib

# Page config
st.set_page_config(page_title="Bank Customer Churn Predictor", layout="centered")

st.title("🏦 Bank Customer Churn Prediction")
st.markdown("Enter the customer details below:")

# Load model (make sure these files are in the same folder)
model = joblib.load("churn_model.pkl")

# Optional: load scaler if you used one
# scaler = joblib.load("scaler.pkl")

# -----------------------
# User Inputs
# -----------------------
credit_score = st.number_input("Credit Score", 300, 900, 600)
age = st.number_input("Age", 18, 100, 30)
balance = st.number_input("Balance", 0.0, 250000.0, 50000.0)
num_products = st.number_input("Number of Products", 1, 4, 1)
tenure = st.number_input("Tenure (Years with Bank)", 0, 20, 5)
has_cr_card = st.selectbox("Has Credit Card?", [0, 1])
is_active = st.selectbox("Is Active Member?", [0, 1])
estimated_salary = st.number_input("Estimated Salary", 0.0, 250000.0, 50000.0)

gender = st.selectbox("Gender", ["Male", "Female"])
geography = st.selectbox("Geography", ["France", "Spain", "Germany"])

# -----------------------
# One-Hot Encoding
# -----------------------
gender_male = 1 if gender == "Male" else 0
gender_female = 1 if gender == "Female" else 0

geo_france = 1 if geography == "France" else 0
geo_spain = 1 if geography == "Spain" else 0
geo_germany = 1 if geography == "Germany" else 0

# -----------------------
# Prepare input array
# -----------------------
input_data = np.array([[credit_score, geo_france, geo_spain, geo_germany,
                        gender_male, gender_female, age, tenure, balance,
                        num_products, has_cr_card, is_active, estimated_salary]])

# If you used scaler:
# input_data = scaler.transform(input_data)

# -----------------------
# Prediction
# -----------------------
if st.button("Predict"):
    prediction = model.predict(input_data)
    probability = model.predict_proba(input_data)[0][1]

    st.subheader("Prediction Result")

    if probability > 0.7:
        st.error(f"🔴 High Risk Customer (Probability: {probability:.2f})")
    elif probability > 0.4:
        st.warning(f"🟡 Medium Risk Customer (Probability: {probability:.2f})")
    else:
        st.success(f"🟢 Low Risk Customer (Probability: {probability:.2f})")

    st.markdown("### 📌 Business Insight")
    if prediction[0] == 1:
        st.write("Customer is likely to churn. Consider retention strategies: discounts, loyalty programs, or personalized offers.")
    else:
        st.write("Customer is likely to stay. Maintain engagement and upsell opportunities.")

# -----------------------
# Sidebar Info
# -----------------------
st.sidebar.title("Project Info")
st.sidebar.write("""
Model: Random Forest Classifier  
Features: 13 (including one-hot encoded categorical)  
Developer: Jahongir Bekmurodov  
Use Case: Customer Churn Prediction
""")