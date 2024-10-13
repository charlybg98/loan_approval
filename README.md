# Loan Approval Prediction

This project is part of the **Kaggle Playground Series - Season 4, Episode 10**. The goal is to predict whether a loan applicant will be approved using machine learning models.

## Implementation

- **Data Loading**: Handled with `DataLoader`, including preprocessing and splitting the data.
- **Modeling**: XGBoost is used for classification, with training managed by the `Trainer` class.
- **Hyperparameter Tuning**: Performed using Optuna with the `HyperparameterTuner` class.
- **Evaluation**: The model is evaluated using accuracy, precision, recall, F1-score, and AUC-ROC.

Run the project using the provided `main.py` script for training, tuning, and inference.