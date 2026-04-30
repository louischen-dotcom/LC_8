# train_model.py — Complete model training script

import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, roc_auc_score
import joblib
import os

# --- Generate synthetic data ---
np.random.seed(42)
n_samples = 5000

data = pd.DataFrame({
    'age': np.random.randint(21, 70, n_samples),
    'annual_income': np.random.lognormal(mean=10.5, sigma=0.8, size=n_samples).astype(int),
    'debt_to_income_ratio': np.random.uniform(0, 1.5, n_samples).round(3),
    'credit_history_length': np.random.randint(0, 30, n_samples),
    'num_open_accounts': np.random.randint(1, 20, n_samples),
    'num_late_payments': np.random.poisson(lam=1.5, size=n_samples),
    'loan_amount': np.random.randint(1000, 50000, n_samples),
})

# --- Create target ---
default_probability = (
    0.15 * data['debt_to_income_ratio']
    + 0.1 * (data['num_late_payments'] / 10)
    - 0.05 * (data['credit_history_length'] / 30)
    + 0.05 * (data['loan_amount'] / 50000)
    - 0.05 * (data['annual_income'] / data['annual_income'].max())
)
default_probability = default_probability.clip(0.05, 0.95)
data['default'] = np.random.binomial(1, default_probability)

print(f"Dataset shape: {data.shape}")
print(f"Default rate: {data['default'].mean():.2%}")

# --- Split ---
feature_columns = [
    'age', 'annual_income', 'debt_to_income_ratio',
    'credit_history_length', 'num_open_accounts',
    'num_late_payments', 'loan_amount'
]
X = data[feature_columns]
y = data['default']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# --- Train ---
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', GradientBoostingClassifier(
        n_estimators=100, max_depth=4, learning_rate=0.1, random_state=42
    ))
])
pipeline.fit(X_train, y_train)

# --- Evaluate ---
y_pred = pipeline.predict(X_test)
y_proba = pipeline.predict_proba(X_test)[:, 1]
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print(f"ROC AUC Score: {roc_auc_score(y_test, y_proba):.4f}")

# --- Save ---
os.makedirs('model', exist_ok=True)
os.makedirs('data', exist_ok=True)
joblib.dump(pipeline, 'model/credit_model.pkl')
X_train.to_csv('data/reference_data.csv', index=False)
X_test.to_csv('data/test_data.csv', index=False)
print("\nModel and reference data saved.")