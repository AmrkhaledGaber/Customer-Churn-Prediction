import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report, confusion_matrix, roc_curve
from imblearn.over_sampling import SMOTE
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
import os
import json

# Create result folder if it doesn't exist
if not os.path.exists('result'):
    os.makedirs('result')

# Load dataset
df = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')

# Preprocessing
df['TotalCharges'] = df.apply(lambda row: row['MonthlyCharges'] if row['TotalCharges'] == ' ' else row['TotalCharges'], axis=1)
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df['TotalCharges'] = df['TotalCharges'].fillna(df['MonthlyCharges'])
df.drop('customerID', axis=1, inplace=True)
df['SeniorCitizen'] = df['SeniorCitizen'].astype('str')  # Ensure SeniorCitizen is string
df.columns = df.columns.str.lower()
df['churn'] = (df['churn'] == 'Yes').astype(int)

# Split data
df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=1)
df_train, df_valid = train_test_split(df_full_train, test_size=0.2, random_state=1)

# Define features
categorical = df_train.select_dtypes(include=['object']).columns.tolist()
numerical = ['tenure', 'monthlycharges', 'totalcharges']
y_train = df_train['churn']
y_valid = df_valid['churn']
y_test = df_test['churn']

# Preprocessing function
def preprocess(df_train, df_valid, df_test, num, cat):
    ohe = OneHotEncoder(drop='first', handle_unknown='ignore', sparse_output=False)
    scaler = StandardScaler()
    transformer = make_column_transformer(
        (scaler, num),
        (ohe, cat),
        remainder='passthrough',
        verbose_feature_names_out=False
    )
    X_train = transformer.fit_transform(df_train[cat + num])
    X_valid = transformer.transform(df_valid[cat + num])
    X_test = transformer.transform(df_test[cat + num])
    feature_names = transformer.get_feature_names_out()
    return X_train, X_valid, X_test, transformer, feature_names

X_train, X_valid, X_test, transformer, feature_names = preprocess(df_train, df_valid, df_test, numerical, categorical)

# Apply SMOTE to handle class imbalance
smote = SMOTE(random_state=1)
X_train, y_train = smote.fit_resample(X_train, y_train)

# Define models and hyperparameter grids
models = {
    'Logistic Regression': LogisticRegression(solver='liblinear', random_state=1),
    'Random Forest': RandomForestClassifier(random_state=1),
    'XGBoost': XGBClassifier(random_state=1, eval_metric='logloss'),
    'SVM': SVC(random_state=1, probability=True)
}

param_grids = {
    'Logistic Regression': {'C': [0.1, 1, 10]},
    'Random Forest': {'n_estimators': [100, 200], 'max_depth': [10, 20, None]},
    'XGBoost': {'n_estimators': [100, 200], 'learning_rate': [0.01, 0.1], 'max_depth': [3, 5]},
    'SVM': {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf']}
}

# Train and tune models
best_models = {}
results = {}
best_params = {}
for name, model in models.items():
    grid_search = GridSearchCV(model, param_grids[name], cv=5, scoring='f1')
    grid_search.fit(X_train, y_train)
    best_models[name] = grid_search.best_estimator_
    best_params[name] = grid_search.best_params_
    
    # Evaluate on test set
    y_pred = best_models[name].predict(X_test)
    y_prob = best_models[name].predict_proba(X_test)[:, 1]
    
    results[name] = {
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred),
        'F1-Score': f1_score(y_test, y_pred),
        'AUC': roc_auc_score(y_test, y_prob)
    }
    print(f"\n{name} Classification Report:")
    print(classification_report(y_test, y_pred))
    
    # Save classification report
    with open(f'result/{name.lower().replace(" ", "_")}_classification_report.txt', 'w') as f:
        f.write(classification_report(y_test, y_pred))
    
    # Generate and save confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='0.0f', cmap='Blues')
    plt.title(f'{name} Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(f'result/{name.lower().replace(" ", "_")}_cm.png')
    plt.close()
    
    # Generate and save ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    plt.figure(figsize=(6, 4))
    plt.plot(fpr, tpr, label=f'{name} (AUC = {results[name]["AUC"]:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'{name} ROC Curve')
    plt.legend()
    plt.savefig(f'result/{name.lower().replace(" ", "_")}_roc_curve.png')
    plt.close()
    
    # Save feature importance for Random Forest and XGBoost
    if name in ['Random Forest', 'XGBoost']:
        importance = best_models[name].feature_importances_
        feature_importance = pd.DataFrame({'Feature': feature_names, 'Importance': importance})
        feature_importance.sort_values(by='Importance', ascending=False, inplace=True)
        feature_importance.to_csv(f'result/{name.lower().replace(" ", "_")}_feature_importance.csv', index=False)

# Save model comparison metrics
results_df = pd.DataFrame(results).T
results_df.to_csv('result/model_comparison_metrics.csv')

# Save best hyperparameters
with open('result/best_hyperparameters.json', 'w') as f:
    json.dump(best_params, f, indent=4)

# Display results
print("\nModel Comparison:")
for name, metrics in results.items():
    print(f"\n{name}:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.2f}")

# Select and save best model (based on F1-Score)
best_model_name = max(results, key=lambda x: results[x]['F1-Score'])
best_model = best_models[best_model_name]
pickle.dump(best_model, open("result/best_churn_model.pkl", 'wb'))
pickle.dump(transformer, open("result/transformer.pkl", 'wb'))

# Example prediction
cust = {
    'gender': 'Female',
    'seniorcitizen': '0',  # String to match training data
    'partner': 'No',
    'dependents': 'No',
    'tenure': 32,
    'phoneservice': 'Yes',
    'multiplelines': 'No',
    'internetservice': 'Fiber optic',
    'onlinesecurity': 'No',
    'onlinebackup': 'Yes',
    'deviceprotection': 'No',
    'techsupport': 'No',
    'streamingtv': 'No',
    'streamingmovies': 'No',
    'contract': 'Month-to-month',
    'paperlessbilling': 'Yes',
    'paymentmethod': 'Electronic check',
    'monthlycharges': 93.95,
    'totalcharges': 2861.45
}
cust_df = pd.DataFrame([cust])
cust_transformed = transformer.transform(cust_df)
prediction = best_model.predict(cust_transformed)[0]
probability = best_model.predict_proba(cust_transformed)[0][1]
print(f"\nPrediction for new customer: {'Churn' if prediction == 1 else 'Not Churn'} (Probability: {probability:.2f})")