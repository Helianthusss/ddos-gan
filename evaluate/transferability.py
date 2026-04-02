"""
Transferability Evaluation
GAN được train để bypass MLP (white-box).
Test xem fake samples có bypass được RF (black-box) không.
Nếu có → transferability claim cho paper.
"""

import numpy as np
import json, os
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score, confusion_matrix
import torch
from torch.utils.data import DataLoader, TensorDataset

from detector.mlp import MLPDetector, evaluate
from detector.random_forest import predict_rf

import pickle

DATA_DIR     = "data/processed"
DETECTOR_DIR = "detector"
GAN_DIR      = "gan"
EVAL_DIR     = "evaluate"
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs(EVAL_DIR, exist_ok=True)


def eval_detector_on_fake(detector_name: str, X_benign: np.ndarray,
                           fake_ddos: np.ndarray) -> dict:
    n     = min(len(X_benign), len(fake_ddos))
    X_adv = np.concatenate([X_benign[:n], fake_ddos[:n]])
    y_adv = np.concatenate([np.zeros(n), np.ones(n)]).astype(int)

    if detector_name == "mlp_v1":
        with open(f"{DETECTOR_DIR}/model_config.pkl", "rb") as f:
            cfg = pickle.load(f)
        model = MLPDetector(cfg["input_dim"], cfg["hidden_dims"], cfg["dropout"])
        model.load_state_dict(torch.load(
            f"{DETECTOR_DIR}/detector_best.pt", map_location=DEVICE, weights_only=True))
        model.to(DEVICE).eval()
        loader = DataLoader(TensorDataset(
            torch.tensor(X_adv, dtype=torch.float32),
            torch.tensor(y_adv.astype(np.float32))), batch_size=512)
        m = evaluate(model, loader)
        probs, preds = m["probs"], (m["probs"] >= 0.5).astype(int)

    elif detector_name == "mlp_v2":
        with open(f"{DETECTOR_DIR}/model_config.pkl", "rb") as f:
            cfg = pickle.load(f)
        model = MLPDetector(cfg["input_dim"], cfg["hidden_dims"], cfg["dropout"])
        model.load_state_dict(torch.load(
            f"{DETECTOR_DIR}/detector_adv_r1.pt", map_location=DEVICE, weights_only=True))
        model.to(DEVICE).eval()
        loader = DataLoader(TensorDataset(
            torch.tensor(X_adv, dtype=torch.float32),
            torch.tensor(y_adv.astype(np.float32))), batch_size=512)
        m = evaluate(model, loader)
        probs, preds = m["probs"], (m["probs"] >= 0.5).astype(int)

    elif detector_name == "rf":
        probs, preds = predict_rf(X_adv)

    else:
        raise ValueError(f"Unknown detector: {detector_name}")

    cm = confusion_matrix(y_adv, preds)
    tn, fp, fn, tp = cm.ravel()
    return {
        "detector": detector_name,
        "f1":       float(f1_score(y_adv, preds)),
        "accuracy": float(accuracy_score(y_adv, preds)),
        "auc":      float(roc_auc_score(y_adv, probs)),
        "fnr":      float(fn / (fn + tp)) if (fn + tp) > 0 else 0.0,
        "fool_rate": float((preds[y_adv == 1] == 0).mean()),
    }


def run_transferability(round_id: int = 1):
    print(f"\n{'='*60}")
    print(f"  Transferability Evaluation — GAN Round {round_id}")
    print(f"{'='*60}")

    X_test   = np.load(f"{DATA_DIR}/X_test.npy")
    y_test   = np.load(f"{DATA_DIR}/y_test.npy")
    fake     = np.load(f"{GAN_DIR}/fake_ddos_r{round_id}.npy")
    X_benign = X_test[y_test == 0]

    detectors = ["mlp_v1", "rf"]
    if os.path.exists(f"{DETECTOR_DIR}/detector_adv_r1.pt"):
        detectors.append("mlp_v2")

    results = []
    print(f"\n  {'Detector':<12} {'F1':>8} {'FoolRate':>10} {'FNR':>8}")
    print(f"  {'-'*42}")
    for det in detectors:
        try:
            r = eval_detector_on_fake(det, X_benign, fake)
            results.append(r)
            print(f"  {r['detector']:<12} {r['f1']:>8.4f} {r['fool_rate']:>10.4f} {r['fnr']:>8.4f}")
        except Exception as e:
            print(f"  {det:<12} ERROR: {e}")

    print(f"\n  Transferability: GAN trained on MLP — FoolRate on RF = "
          f"{next((r['fool_rate'] for r in results if r['detector']=='rf'), 'N/A'):.4f}")

    path = f"{EVAL_DIR}/transferability_r{round_id}.json"
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Saved → {path}")
    return results


if __name__ == "__main__":
    import sys
    round_id = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    run_transferability(round_id)