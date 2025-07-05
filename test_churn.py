import pandas as pd
import pickle

# Load the saved model and transformer
model = pickle.load(open("result/best_churn_model.pkl", "rb"))
transformer = pickle.load(open("result/transformer.pkl", "rb"))

# Test customer data
test_data = {
    'gender': 'Male',
    'seniorcitizen': '0',
    'partner': 'No',
    'dependents': 'No',
    'tenure': 2,
    'phoneservice': 'Yes',
    'multiplelines': 'No',
    'internetservice': 'Fiber optic',
    'onlinesecurity': 'No',
    'onlinebackup': 'No',
    'deviceprotection': 'No',
    'techsupport': 'No',
    'streamingtv': 'Yes',
    'streamingmovies': 'Yes',
    'contract': 'Month-to-month',
    'paperlessbilling': 'Yes',
    'paymentmethod': 'Electronic check',
    'monthlycharges': 100.0,
    'totalcharges': 200.0
}

# Convert to DataFrame
test_df = pd.DataFrame([test_data])

# Transform data
test_transformed = transformer.transform(test_df)

# Predict
prediction = model.predict(test_transformed)[0]
probability = model.predict_proba(test_transformed)[0][1]

# Print result
print(f"Prediction: {'Churn' if prediction == 1 else 'Not Churn'}")
print(f"Churn Probability: {probability:.2f}")