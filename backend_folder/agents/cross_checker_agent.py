"""Cross-Checker Agent

Engineers features from extracted document fields + raw claim metadata and applies:
- LightGBM classifier (fraud risk)
- IsolationForest anomaly detection

Outputs model_risk plus intermediate diagnostics.
This is a scaffold that can be expanded with persistence, retraining, and calibration.
"""
from __future__ import annotations
import logging
from dataclasses import dataclass
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import joblib
import os

try:
    import lightgbm as lgb
except ImportError:  # pragma: no cover
    lgb = None  # type: ignore

logger = logging.getLogger(__name__)

MODEL_DIR = os.getenv("MODEL_PATH", "./models")

@dataclass
class CrossCheckResult:
    model_risk: float
    fraud_probability: float
    anomaly_score: float
    features: Dict[str, float]
    feature_importance: Dict[str, float]

class CrossCheckerAgent:
    def __init__(self):
        self.scaler = StandardScaler()
        self.lgbm_model = None
        self.isolation_forest = None
        self._load_models()

    def _load_models(self):
        try:
            lgb_path = os.path.join(MODEL_DIR, "lgbm_fraud_model.txt")
            if lgb and os.path.exists(lgb_path):
                self.lgbm_model = lgb.Booster(model_file=lgb_path)
                logger.info("Loaded LightGBM model")
            iso_path = os.path.join(MODEL_DIR, "iso_forest.joblib")
            if os.path.exists(iso_path):
                self.isolation_forest = joblib.load(iso_path)
                logger.info("Loaded IsolationForest model")
            scaler_path = os.path.join(MODEL_DIR, "scaler.joblib")
            if os.path.exists(scaler_path):
                self.scaler = joblib.load(scaler_path)
                logger.info("Loaded feature scaler")
        except Exception as e:
            logger.warning(f"Model load warning: {e}")

    def cross_check(self, extracted_fields: List[Dict[str, Any]], claim_meta: Dict[str, Any]) -> CrossCheckResult:
        # Build feature dict
        feat = self._engineer_features(extracted_fields, claim_meta)
        X = np.array([list(feat.values())])
        self._fit_placeholder(X)  # Fit scalers / fallback models if needed

        fraud_prob = self._predict_lightgbm(X)
        anomaly_score = self._predict_isolation(X)

        # Combine
        model_risk = self._combine(fraud_prob, anomaly_score)

        importance = self._fake_importance(feat, fraud_prob, anomaly_score)

        return CrossCheckResult(
            model_risk=model_risk,
            fraud_probability=fraud_prob,
            anomaly_score=anomaly_score,
            features=feat,
            feature_importance=importance
        )

    # --------- Feature Engineering ---------
    def _engineer_features(self, fields: List[Dict[str, Any]], meta: Dict[str, Any]) -> Dict[str, float]:
        f_map = {f['name']: f for f in fields if 'name' in f}
        def _num(name, default=0.0):
            try:
                return float(f_map.get(name, {}).get('value', meta.get(name, default)))
            except Exception:
                return default

        features = {
            'claim_amount': _num('claim_amount'),
            'policy_duration': float(meta.get('policy_duration', meta.get('policy_duration_months', 12))),
            'previous_claims': float(meta.get('previous_claims', 0)),
            'extracted_field_count': float(len(f_map)),
            'avg_confidence': float(np.mean([f.get('confidence', 0) for f in f_map.values()]) if f_map else 0.0),
            'amount_per_month': 0.0,
            'high_amount_flag': 0.0,
            'low_confidence_flag': 0.0,
        }
        if features['policy_duration'] > 0:
            features['amount_per_month'] = features['claim_amount'] / features['policy_duration']
        features['high_amount_flag'] = 1.0 if features['claim_amount'] > 50000 else 0.0
        features['low_confidence_flag'] = 1.0 if features['avg_confidence'] < 0.6 else 0.0
        return features

    # --------- Model Predictions ---------
    def _fit_placeholder(self, X: np.ndarray):
        # Fit scaler if not already fitted (placeholder training)
        if not hasattr(self.scaler, 'n_features_in_'):
            self.scaler.fit(X)
        if self.isolation_forest is None:
            self.isolation_forest = IsolationForest(n_estimators=50, contamination=0.05, random_state=42)
            self.isolation_forest.fit(X)

    def _predict_lightgbm(self, X: np.ndarray) -> float:
        # Scale if scaler fitted
        try:
            if hasattr(self.scaler, 'n_features_in_'):
                Xs = self.scaler.transform(X)
            else:
                Xs = X
        except Exception as e:
            logger.warning(f"Scaler transform failed: {e}")
            Xs = X

        if self.lgbm_model is None or lgb is None:
            # Simplistic heuristic fallback
            return float(min(1.0, max(0.0, Xs[0][0] / 100000 + 0.1 * Xs[0][-1])))
        try:
            return float(self.lgbm_model.predict(Xs)[0])
        except Exception as e:
            logger.warning(f"LightGBM prediction failed: {e}")
            return 0.5

    def _predict_isolation(self, X: np.ndarray) -> float:
        try:
            if hasattr(self.scaler, 'n_features_in_'):
                Xs = self.scaler.transform(X)
            else:
                Xs = X
            score = self.isolation_forest.decision_function(Xs)[0]
            # Normalize to 0-1 (lower means more anomalous in IsolationForest convention)
            norm = (1 - score) / 2
            return float(min(1.0, max(0.0, norm)))
        except Exception as e:
            logger.warning(f"IsolationForest prediction failed: {e}")
            return 0.3

    def _combine(self, fraud_prob: float, anomaly_score: float) -> float:
        return round(0.6 * fraud_prob + 0.4 * anomaly_score, 4)

    def _fake_importance(self, feat: Dict[str, float], fraud: float, anomaly: float) -> Dict[str, float]:
        # Placeholder scaled values indicating relative influence
        total = sum(abs(v) for v in feat.values()) or 1.0
        return {k: round(abs(v) / total, 3) for k, v in feat.items()}
