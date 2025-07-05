import streamlit as st
import pandas as pd
import pickle

# Load the saved model and transformer
model = pickle.load(open("result/best_churn_model.pkl", "rb"))
transformer = pickle.load(open("result/transformer.pkl", "rb"))

# Streamlit app
st.set_page_config(page_title="Customer Churn Prediction", layout="wide")
st.title("📊 Customer Churn Prediction App")
st.markdown("""
Enter the customer details below to predict whether they are likely to churn.
The model uses advanced machine learning to provide accurate predictions based on the Telco dataset.
""")
# Display model type for verification
st.write(f"Loaded Model: {type(model).__name__}")

# Organize inputs in two columns for better layout
col1, col2 = st.columns(2)

# Input fields for categorical features
with col1:
    st.subheader("Customer Information")
    gender = st.selectbox("Gender", ["Female", "Male"])
    senior_citizen = st.selectbox("Senior Citizen", ["0", "1"])
    partner = st.selectbox("Partner", ["Yes", "No"])
    dependents = st.selectbox("Dependents", ["Yes", "No"])
    phone_service = st.selectbox("Phone Service", ["Yes", "No"])
    multiple_lines = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
    internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    online_security = st.selectbox("Online Security", ["Yes", "No", "No internet service"])

with col2:
    st.subheader("Service and Billing Details")
    online_backup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
    device_protection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
    tech_support = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
    streaming_tv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
    streaming_movies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])
    contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
    payment_method = st.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])

# Input fields for numerical features
st.subheader("Numerical Inputs")
tenure = st.slider("Tenure (months)", 0, 72, 32)
monthly_charges = st.number_input("Monthly Charges ($)", min_value=0.0, max_value=200.0, value=93.95, step=0.01)
total_charges = st.number_input("Total Charges ($)", min_value=0.0, max_value=10000.0, value=2861.45, step=0.01)

# Predict button
if st.button("Predict Churn"):
    # Create a dictionary with input data
    input_data = {
        'gender': gender,
        'seniorcitizen': senior_citizen,
        'partner': partner,
        'dependents': dependents,
        'tenure': tenure,
        'phoneservice': phone_service,
        'multiplelines': multiple_lines,
        'internetservice': internet_service,
        'onlinesecurity': online_security,
        'onlinebackup': online_backup,
        'deviceprotection': device_protection,
        'techsupport': tech_support,
        'streamingtv': streaming_tv,
        'streamingmovies': streaming_movies,
        'contract': contract,
        'paperlessbilling': paperless_billing,
        'paymentmethod': payment_method,
        'monthlycharges': monthly_charges,
        'totalcharges': total_charges
    }
    
    # Convert input to DataFrame
    input_df = pd.DataFrame([input_data])
    
    # Transform input data
    input_transformed = transformer.transform(input_df)
    
    # Make prediction
    prediction = model.predict(input_transformed)[0]
    probability = model.predict_proba(input_transformed)[0][1]
    
    # Display results
    st.subheader("Prediction Result")
    result = "Churn" if prediction == 1 else "Not Churn"
    st.markdown(f"**Prediction**: {result}")
    st.markdown(f"**Churn Probability**: {probability:.2f}")
    
    # Add visual indicator
    if prediction == 1:
        st.error("⚠️ This customer is likely to churn!")
    else:
        st.success("✅ This customer is likely to stay.")

# Add footer
st.markdown("---")
st.markdown("Developed with Streamlit | Powered by Machine Learning")