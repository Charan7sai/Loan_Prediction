Project Overview

This project aims to predict whether a loan application will be Approved (Y) or Rejected (N) using Machine Learning techniques.

The project covers the complete ML pipeline:

Exploratory Data Analysis (EDA)

Data Cleaning & Preprocessing

Statistical Hypothesis Testing

Feature Engineering

Model Training & Evaluation

Final Model Selection

Deployment-ready Pipeline Script

 Project Structure
Loan_Prediction/
│
├── data/
│   ├── loan_data.csv
│   └── loan_data_cleaned.csv
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_data_cleaning.ipynb
│   ├── 03_hypothesis_testing.ipynb
│   ├── 04_feature_engineering_modeling.ipynb
│   └── 05_model_evaluation.ipynb
│
├── models/
│   └── final_model.pkl
│
├── src/
│   └── loan_prediction_pipeline.py
│
├── reports/
│   └── Loan_Approval_Prediction_ML_Report.pdf
│
└── README.md

📊 Dataset Description

The dataset contains applicant information such as:

Gender

Marital Status

Dependents

Education

Self Employment

Applicant Income

Coapplicant Income

Loan Amount

Loan Term

Credit History

Property Area

Loan Status (Target Variable)

📈 Statistical Tests Performed
1️⃣ Chi-Square Test

Tested relationship between Education Level and Loan Status

Result: p-value < 0.05

Conclusion: Significant relationship exists.

2️⃣ Two-Sample T-Test

Compared mean Applicant Income between approved and rejected loans

Result: p-value > 0.05

Conclusion: No significant difference in income.

3️⃣ ANOVA (Conceptual Explanation)

Discussed for multi-group comparison scenarios.

🤖 Models Implemented
🔹 Logistic Regression

Accuracy: 0.79

Precision: 0.76

Recall: 0.99

F1-Score: 0.86

ROC-AUC: 0.75

🔹 Decision Tree Classifier

Accuracy: 0.70

Precision: 0.76

Recall: 0.79

F1-Score: 0.77

ROC-AUC: 0.66

 Final Model Selection

Logistic Regression was selected as the final model because:

Higher ROC-AUC score (0.75)

Better overall accuracy

Very high recall for approved loans

Better generalization

More interpretable than Decision Tree

📉 ROC-AUC Curve

The ROC curve comparison demonstrates that Logistic Regression provides better discrimination between approved and rejected applications.

🛠 Technologies Used

Python

Pandas

NumPy

Matplotlib

Scikit-Learn

Joblib

Jupyter Notebook
