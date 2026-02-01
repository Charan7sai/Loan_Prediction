"""
Loan Prediction Pipeline
------------------------
This script performs preprocessing, feature selection,
model training, evaluation, and final model saving
for the Loan Prediction project.
"""

import os
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, roc_auc_score


# -----------------------------
# Load Dataset (PATH SAFE)
# -----------------------------
def load_data(relative_path):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base_dir, relative_path)
    return pd.read_csv(data_path)


# -----------------------------
# Preprocessing
# -----------------------------
def preprocess_data(df):
    le = LabelEncoder()

    categorical_cols = [
        'Gender', 'Married', 'Dependents',
        'Education', 'Self_Employed', 'Property_Area'
    ]

    for col in categorical_cols:
        df[col] = le.fit_transform(df[col])

    df['Loan_Status'] = le.fit_transform(df['Loan_Status'])

    X = df.drop(['Loan_Status', 'Loan_ID'], axis=1)
    y = df['Loan_Status']

    return X, y


# -----------------------------
# Feature Selection
# -----------------------------
def select_features(X, y, k=10):
    selector = SelectKBest(score_func=chi2, k=k)
    X_selected = selector.fit_transform(X, y)
    selected_features = X.columns[selector.get_support()]
    return X_selected, selected_features


# -----------------------------
# Model Training
# -----------------------------
def train_models(X_train, y_train):
    log_reg = LogisticRegression(max_iter=2000)
    dt_model = DecisionTreeClassifier(random_state=42)

    log_reg.fit(X_train, y_train)
    dt_model.fit(X_train, y_train)

    return log_reg, dt_model


# -----------------------------
# Model Evaluation
# -----------------------------
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)

    return accuracy, auc


# -----------------------------
# Main Pipeline
# -----------------------------
def main():
    print("Loading cleaned dataset...")
    df = load_data("../data/loan_data_cleaned.csv")

    print("Preprocessing data...")
    X, y = preprocess_data(df)

    print("Selecting best features...")
    X_selected, features = select_features(X, y)
    print("Selected Features:", list(features))

    print("Splitting dataset...")
    X_train, X_test, y_train, y_test = train_test_split(
        X_selected, y, test_size=0.2, random_state=42
    )

    print("Training models...")
    log_reg, dt_model = train_models(X_train, y_train)

    print("Evaluating Logistic Regression...")
    lr_acc, lr_auc = evaluate_model(log_reg, X_test, y_test)

    print("Evaluating Decision Tree...")
    dt_acc, dt_auc = evaluate_model(dt_model, X_test, y_test)

    print("\nModel Performance:")
    print(f"Logistic Regression -> Accuracy: {lr_acc:.2f}, AUC: {lr_auc:.2f}")
    print(f"Decision Tree       -> Accuracy: {dt_acc:.2f}, AUC: {dt_auc:.2f}")

    # Create models directory safely
    base_dir = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(base_dir, "../models")
    os.makedirs(models_dir, exist_ok=True)

    # Save best model
    if lr_auc > dt_auc:
        print("\nSaving Logistic Regression as final model...")
        joblib.dump(log_reg, os.path.join(models_dir, "final_model.pkl"))
    else:
        print("\nSaving Decision Tree as final model...")
        joblib.dump(dt_model, os.path.join(models_dir, "final_model.pkl"))

    print("Pipeline completed successfully.")


if __name__ == "__main__":
    main()