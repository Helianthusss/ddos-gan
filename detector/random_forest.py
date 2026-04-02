"""
Random Forest Detector
Dùng để test transferability: GAN train để bypass MLP
nhưng cũng bypass được RF không?
"""

import numpy as np
import pickle, os
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score, confusion_matrix

DATA_DIR     = "data/processed"
DETECTOR_DIR = "detector"
os.makedirs(DETECTOR_DIR, exist_ok=True)


def train_rf():
    print("\n[RF] Training Random Forest detector ...")
    X_train = np.load(f"{DATA_DIR}/X_train.npy")
    y_train = np.load(f"{DATA_DIR}/y_train.npy")
    X_test  = np.load(f"{DATA_DIR}/X_test.npy")
    y_test  = np.load(f"{DATA_DIR}/y_test.npy")

    rf = RandomForestClassifier(
        n_estimators=100, max_depth=20,
        n_jobs=-1, random_state=42, class_weight="balanced"
    )
    rf.fit(X_train, y_train)

    preds = rf.predict(X_test)
    probs = rf.predict_proba(X_test)[:, 1]

    cm = confusion_matrix(y_test, preds)
    tn, fp, fn, tp = cm.ravel()
    metrics = {
        "accuracy": accuracy_score(y_test, preds),
        "f1":       f1_score(y_test, preds),
        "auc":      roc_auc_score(y_test, probs),
        "fnr":      fn / (fn + tp),
    }

    print(f"  Accuracy : {metrics['accuracy']:.4f}")
    print(f"  F1       : {metrics['f1']:.4f}")
    print(f"  AUC      : {metrics['auc']:.4f}")
    print(f"  FNR      : {metrics['fnr']:.4f}")

    path = f"{DETECTOR_DIR}/rf_detector.pkl"
    with open(path, "wb") as f:
        pickle.dump(rf, f)
    print(f"  Saved → {path}")
    return rf, metrics


def predict_rf(X: np.ndarray) -> tuple:
    with open(f"{DETECTOR_DIR}/rf_detector.pkl", "rb") as f:
        rf = pickle.load(f)
    preds = rf.predict(X)
    probs = rf.predict_proba(X)[:, 1]
    return probs, preds


if __name__ == "__main__":
    train_rf()