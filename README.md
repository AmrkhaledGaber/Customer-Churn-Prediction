# Customer-Churn-Prediction


## Overview

This project focuses on predicting customer churn for a telecommunications company using machine learning. The dataset used is the [Telco Customer Churn dataset](https://www.kaggle.com/blastchar/telco-customer-churn). The goal of this project is to develop a machine learning model to predict whether a customer will churn (leave the company) or not, and provide a web-based user interface for easy predictions.

The project involves data collection, preprocessing, feature extraction, model training, testing and evaluation, and deployment through a Streamlit web application.

## Requirements

* Python 3.6+
* Libraries:

  * Pandas
  * NumPy
  * Scikit-learn
  * XGBoost
  * Streamlit
  * imbalanced-learn (for SMOTE)

## Installation

### 1. Clone the repository:

```bash
git clone https://github.com/AmrkhaledGaber/Customer-Churn-Prediction.git
```

### 2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Project Structure

* `Customer Churn Prediction Project Report.pdf`: Final project report.
* `Customer Churn Prediction Project Summary.markdown`: Summary of the project.
* `WA_Fn-UseC_-Telco-Customer-Churn.csv`: The dataset used for model training.
* `app.py`: Streamlit application for churn prediction.
* `churn_prediction.py`: Main script for training and evaluating machine learning models.
* `result/`: Folder containing saved models, visualizations, and other results.
* `test_churn.py`: Testing script for model predictions.

## Key Steps and Implementation

### 1. Data Collection

The `WA_Fn-UseC_-Telco-Customer-Churn.csv` dataset is loaded and preprocessed for model training.

### 2. Data Preprocessing

* Handled missing values.
* Converted categorical features to numerical.
* Applied SMOTE to address class imbalance.

### 3. Feature Extraction

* Encoded categorical variables.
* Scaled numerical features.
* Combined preprocessing steps into a pipeline.

### 4. Model Training

Trained four models:

* Logistic Regression
* Random Forest
* XGBoost
* Support Vector Machine (SVM)

### 5. Testing and Evaluation

* Evaluated models on various metrics like accuracy, precision, recall, F1-score, and AUC.
* The best models were Logistic Regression and XGBoost based on F1-score and recall.

### 6. Deployment

* Saved the best model (Logistic Regression or XGBoost).
* Developed a Streamlit application for user input and churn prediction.

## How to Use

### Running the Streamlit App

To run the Streamlit app, use the following command:

```bash
streamlit run app.py
```

* Enter customer details such as tenure, contract type, payment method, etc.
* Get churn prediction (Yes or No) along with the probability.

### Example Input:

* **Tenure**: 10
* **Monthly Charges**: 75.00
* **Total Charges**: 750.00
* **Contract**: Month-to-month
* **Payment Method**: Electronic check

### Example Output:

* **Prediction**: Churn
* **Probability**: 0.75

## Challenges and Solutions

* **Class Imbalance**: Applied SMOTE to balance the classes and improve model performance.
* **Handling Missing Values**: Replaced missing `TotalCharges` with `MonthlyCharges`.
* **Inconsistent Categorical Encoding**: Ensured consistency with `OneHotEncoder` using `handle_unknown='ignore'`.
* **Streamlit Integration**: Successfully integrated model into the web app with proper debugging and testing.

## Future Improvements

* Experiment with deep learning models.
* Add feature importance and confusion matrix visualizations to the Streamlit app.
* Deploy the app to a cloud platform (e.g., AWS or Streamlit Cloud).
* Monitor model performance over time for data drift.

## Conclusion

This project successfully implemented a machine learning model to predict customer churn, and the Streamlit app provides an easy-to-use interface for users. The project highlights key steps in the machine learning pipeline, including data preprocessing, model evaluation, and deployment.

