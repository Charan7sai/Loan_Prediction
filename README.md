#  Loan Approval Prediction using Machine Learning

##  Project Overview
This project builds a Machine Learning model to predict whether a loan application will be Approved (Y) or Rejected (N) based on applicant demographic and financial information. The project follows a complete end-to-end Machine Learning pipeline including Exploratory Data Analysis (EDA), Data Cleaning & Preprocessing, Statistical Hypothesis Testing, Feature Engineering & Selection, Model Training & Evaluation, Final Model Selection, and a Deployment-ready Python pipeline.

##  Project Structure
Loan_Prediction/
├── data/
│   ├── loan_data.csv
│   └── loan_data_cleaned.csv
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_data_cleaning.ipynb
│   ├── 03_hypothesis_testing.ipynb
│   ├── 04_feature_engineering_modeling.ipynb
│   └── 05_model_evaluation.ipynb
├── models/
│   └── final_model.pkl
├── src/
│   └── loan_prediction_pipeline.py
├── reports/
│   └── Loan_Prediction_ML_Report.pdf
└── README.md

##  Dataset Description
The dataset contains applicant details such as Gender, Marital Status, Dependents, Education, Self Employment, Applicant Income, Coapplicant Income, Loan Amount, Loan Term, Credit History, Property Area, and Loan Status (Target Variable). The target variable Loan_Status indicates whether a loan is approved or rejected.

##  Statistical Analysis
Chi-Square Test:
- Tested relationship between Education Level and Loan Status
- P-value = 0.043
- Conclusion: Significant relationship exists (Reject Null Hypothesis)

Two-Sample T-Test:
- Compared Applicant Income between approved and rejected loans
- P-value = 0.907
- Conclusion: No significant income difference (Fail to Reject Null Hypothesis)

ANOVA (Conceptual):
- Explained as appropriate for multi-group comparisons

##  Models Implemented
Logistic Regression:
- Accuracy: 0.79
- Precision: 0.76
- Recall: 0.99
- F1-score: 0.86
- ROC-AUC: 0.75

Decision Tree:
- Accuracy: 0.70
- Precision: 0.76
- Recall: 0.79
- F1-score: 0.77
- ROC-AUC: 0.66

##  Final Model Selection
Logistic Regression was selected as the final model because it achieved a higher ROC-AUC score (0.75), demonstrated strong generalization capability, achieved very high recall for approved loans, and provided better interpretability compared to the Decision Tree model.
