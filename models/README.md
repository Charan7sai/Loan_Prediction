# Trained Model – Loan Prediction

This directory contains the final trained machine learning model used
for predicting loan approval status in the Loan Prediction project.


## Model File

- **Filename:** `final_model.pkl`
- **Model Type:** Logistic Regression
- **Selection Criteria:** ROC–AUC performance
- **Framework:** scikit-learn

The Logistic Regression model was selected as the final model based on
its superior ROC–AUC score compared to other evaluated models, indicating
better discrimination between approved and rejected loan applications.


## Model Description

The model was trained using a cleaned and preprocessed dataset with:
- Encoded categorical variables
- Selected top features using Chi-Square feature selection
- 80–20 train-test split
- Evaluation using Accuracy, Precision, Recall, F1-score, and ROC–AUC

The final ROC–AUC score achieved by the model is approximately **0.75**,
demonstrating good predictive performance.


## How to Load the Model

The model can be loaded using the `joblib` library as shown below:

```python
import joblib

model = joblib.load('models/final_model.pkl')
