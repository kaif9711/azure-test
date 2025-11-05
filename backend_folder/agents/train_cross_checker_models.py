"""Training script for CrossCheckerAgent models using health insurance dataset.

Usage:
  python backend/agents/train_cross_checker_models.py \
      --data data/health_insurance_dataset.csv \
      --output models/ \
      --target charges \
      --synthetic-fraud

Notes:
- If your dataset lacks an explicit fraud label, you can create a synthetic label with --synthetic-fraud.
- Produces: lgbm_fraud_model.txt, iso_forest.joblib, scaler.joblib
"""
from __future__ import annotations
import argparse
import os
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, f1_score

try:
    import lightgbm as lgb
except ImportError:
    lgb = None  # type: ignore

RANDOM_STATE = 42


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    features = df.copy()
    # Basic cleaning
    for col in features.select_dtypes(include=['object']).columns:
        features[col] = features[col].astype('category')
    # Simple numeric encoding for categories
    for col in features.select_dtypes(include=['category']).columns:
        features[col] = features[col].cat.codes
    # Interaction examples
    if {'bmi', 'age'}.issubset(features.columns):
        features['bmi_age_interaction'] = features['bmi'] * features['age']
    return features


def create_synthetic_fraud_label(df: pd.DataFrame, base_col: str) -> pd.Series:
    # Heuristic: top decile of charges considered positive class
    threshold = df[base_col].quantile(0.9)
    return (df[base_col] > threshold).astype(int)


def train_models(data_path: str, output: str, target: str, synthetic_fraud: bool):
    os.makedirs(output, exist_ok=True)
    df = pd.read_csv(data_path)

    if synthetic_fraud:
        if target not in df.columns:
            raise ValueError(f"Target column '{target}' not found for synthetic fraud generation.")
        y = create_synthetic_fraud_label(df, target)
    else:
        if target not in df.columns:
            raise ValueError(f"Target column '{target}' not in dataset; supply explicit label or use --synthetic-fraud.")
        y = df[target]

    X = build_features(df.drop(columns=[target])) if target in df.columns else build_features(df)

    # Align shapes
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)

    # Scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # LightGBM
    if lgb is not None:
        lgb_train = lgb.Dataset(X_train_scaled, label=y_train)
        params = {
            'objective': 'binary',
            'learning_rate': 0.05,
            'num_leaves': 31,
            'feature_fraction': 0.9,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'metric': 'auc',
            'seed': RANDOM_STATE
        }
        model = lgb.train(params, lgb_train, num_boost_round=200)
        model.save_model(os.path.join(output, 'lgbm_fraud_model.txt'))
        y_pred_prob = model.predict(X_test_scaled)
        auc = roc_auc_score(y_test, y_pred_prob)
        y_pred_label = (y_pred_prob > 0.5).astype(int)
        f1 = f1_score(y_test, y_pred_label)
        print(f"LightGBM AUC={auc:.4f} F1={f1:.4f}")
    else:
        print("LightGBM not installed; skipping gradient boosting model.")

    # IsolationForest (unsupervised)
    iso = IsolationForest(n_estimators=200, contamination=0.05, random_state=RANDOM_STATE)
    iso.fit(X_train_scaled)
    joblib.dump(iso, os.path.join(output, 'iso_forest.joblib'))

    # Persist scaler
    joblib.dump(scaler, os.path.join(output, 'scaler.joblib'))
    print(f"Artifacts written to {output}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data', required=True, help='Path to CSV dataset')
    p.add_argument('--output', default='models', help='Output directory for artifacts')
    p.add_argument('--target', default='charges', help='Target column (ignored if --synthetic-fraud is set but needed to derive features)')
    p.add_argument('--synthetic-fraud', action='store_true', help='Create synthetic fraud label from target decile')
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    train_models(args.data, args.output, args.target, args.synthetic_fraud)
