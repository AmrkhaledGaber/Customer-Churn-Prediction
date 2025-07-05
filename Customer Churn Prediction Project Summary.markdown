# Customer Churn Prediction Project Summary

## Requirements Fulfillment

1. **Data Collection**  
   - Dataset: Loaded from 'WA_Fn-UseC_-Telco-Customer-Churn.csv'.

2. **Data Preprocessing**  
   - Handled missing values in 'TotalCharges'.  
   - Converted data types and dropped unnecessary columns.  
   - Encoded categorical variables using OneHotEncoder.  
   - Scaled numerical features with StandardScaler.  
   - Applied SMOTE to balance the dataset.

3. **Feature Extraction**  
   - Used a column transformer to scale numerical features and one-hot encode categorical features.

4. **Model Training**  
   - Trained four models: Logistic Regression, Random Forest, XGBoost, and SVM.  
   - Performed hyperparameter tuning with GridSearchCV.

5. **Testing and Evaluation**  
   - Evaluated models on the test set with accuracy, precision, recall, F1-score, and AUC.  
   - Generated and saved classification reports, confusion matrices, and ROC curves.

6. **Deployment**  
   - Saved the best model as 'best_churn_model.pkl' and transformer as 'transformer.pkl'.  
   - Demonstrated prediction for a new customer.

The project fully meets all specified requirements and is ready for use.