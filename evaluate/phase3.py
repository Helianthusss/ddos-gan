"""
Phase 3 Evaluation
Test Detector v{round_id} với GAN Round {round_id} adversarial samples.
  round_id=1 → detector_best.pt    vs fake_ddos_r1.npy
  round_id=2 → detector_adv_r1.pt  vs fake_ddos_r2.npy
Metrics: F1, AUC, FNR + KL Divergence + KS Statistic
"""

import torch
import numpy as np
import pandas as pd
import pickle
import os
from scipy import stats as scipy_stats
from sklearn.metrics import (
    f1_score, accuracy_score, roc_auc_score,
    confusion_matrix, classification_report
)
from torch.utils.data import DataLoader, TensorDataset
from evaluate.metrics_utils import calculate_distribution_metrics

from detector.mlp import MLPDetector

DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_DIR     = "data/processed"
DETECTOR_DIR = "detector"
GAN_DIR      = "gan"
EVAL_DIR     = "evaluate"
os.makedirs(EVAL_DIR, exist_ok=True)


# ── Load Detector ─────────────────────────────────────────────────────────────

def load_detector(round_id: int = 1):
    """
    Load đúng detector theo round:
      round_id=1 → detector_best.pt    (Detector v1 — baseline)
      round_id=2 → detector_adv_r1.pt  (Detector v2 — sau adversarial training)
    """
    with open(f"{DETECTOR_DIR}/model_config.pkl", "rb") as f:
        cfg = pickle.load(f)
    model = MLPDetector(cfg["input_dim"], cfg["hidden_dims"], cfg["dropout"])
    det_path = (
        f"{DETECTOR_DIR}/detector_best.pt" if round_id == 1
        else f"{DETECTOR_DIR}/detector_adv_r1.pt"
    )
    model.load_state_dict(
        torch.load(det_path, map_location=DEVICE, weights_only=True)
    )
    model.to(DEVICE).eval()
    print(f"[INFO] Loaded detector → {det_path}")
    return model


@torch.no_grad()
def predict(model, X: np.ndarray):
    loader = DataLoader(
        TensorDataset(torch.tensor(X, dtype=torch.float32)),
        batch_size=512
    )
    probs = []
    for (xb,) in loader:
        logits = model(xb.to(DEVICE))
        probs.extend(torch.sigmoid(logits).cpu().numpy())
    probs = np.array(probs)
    preds = (probs >= 0.5).astype(int)
    return probs, preds


# ── Metrics ───────────────────────────────────────────────────────────────────

def compute_metrics(labels, preds, probs, phase_name: str) -> dict:
    cm  = confusion_matrix(labels, preds)
    tn, fp, fn, tp = cm.ravel()
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0

    metrics = {
        "phase"    : phase_name,
        "accuracy" : accuracy_score(labels, preds),
        "f1"       : f1_score(labels, preds),
        "auc"      : roc_auc_score(labels, probs),
        "fnr"      : fnr,
        "tn": tn, "fp": fp, "fn": fn, "tp": tp,
    }
    return metrics


def print_metrics(m: dict):
    print(f"\n{'='*58}")
    print(f"  {m['phase']}")
    print(f"{'='*58}")
    print(f"  Accuracy : {m['accuracy']:.4f}")
    print(f"  F1 Score : {m['f1']:.4f}")
    print(f"  AUC-ROC  : {m['auc']:.4f}")
    print(f"  FNR      : {m['fnr']:.4f}  ← DDoS bị bỏ sót")
    print(f"\n  Confusion Matrix:")
    print(f"    TN={m['tn']:,}  FP={m['fp']:,}")
    print(f"    FN={m['fn']:,}  TP={m['tp']:,}")
    print(f"{'='*58}")


# ── KL Divergence + KS Test ───────────────────────────────────────────────────

def kl_divergence(p: np.ndarray, q: np.ndarray, bins: int = 50) -> float:
    """KL(P || Q) — P=real, Q=fake. Càng nhỏ fake càng giống real."""
    min_v = min(p.min(), q.min())
    max_v = max(p.max(), q.max()) + 1e-8
    p_hist, _ = np.histogram(p, bins=bins, range=(min_v, max_v), density=True)
    q_hist, _ = np.histogram(q, bins=bins, range=(min_v, max_v), density=True)
    # Smooth để tránh log(0)
    eps = 1e-10
    p_hist = p_hist + eps
    q_hist = q_hist + eps
    p_hist /= p_hist.sum()
    q_hist /= q_hist.sum()
    return float(np.sum(p_hist * np.log(p_hist / q_hist)))


def evaluate_gan_quality(real_ddos: np.ndarray, fake_ddos: np.ndarray,
                          feature_names: list, round_id: int = 1) -> dict:
    """
    So sánh phân phối real DDoS vs fake DDoS:
    - KL Divergence (mean over all features)
    - KS Statistic  (mean over all features)
    """
    print(f"\n[GAN Quality] Comparing real vs fake DDoS distributions ...")
    print(f"  Real DDoS shape : {real_ddos.shape}")
    print(f"  Fake DDoS shape : {fake_ddos.shape}")

    kl_scores = []
    ks_scores = []
    ks_pvals  = []

    for i in range(real_ddos.shape[1]):
        r = real_ddos[:, i]
        f = fake_ddos[:, i]
        kl_scores.append(kl_divergence(r, f))
        ks_stat, ks_pval = scipy_stats.ks_2samp(r, f)
        ks_scores.append(ks_stat)
        ks_pvals.append(ks_pval)

    mean_kl = float(np.mean(kl_scores))
    mean_ks = float(np.mean(ks_scores))

    # New advanced metrics: Wasserstein & MMD
    dist_metrics = calculate_distribution_metrics(real_ddos, fake_ddos)
    mean_wd = dist_metrics["mean_wd"]
    mmd = dist_metrics["mmd"]

    # Top 10 features với KL thấp nhất (fake giống real nhất)
    kl_per_feat = pd.DataFrame({
        "feature": feature_names,
        "kl_div" : kl_scores,
        "ks_stat": ks_scores,
        "ks_pval": ks_pvals,
    }).sort_values("kl_div")

    print(f"\n  Mean KL Divergence : {mean_kl:.4f}  (thấp = fake giống real)")
    print(f"  Mean KS Statistic  : {mean_ks:.4f}  (thấp = phân phối gần nhau)")
    print(f"  Mean Wasserstein   : {mean_wd:.4f}  (khoảng cách Wasserstein)")
    print(f"  MMD (RBF Kernel)   : {mmd:.4f}  (Maximum Mean Discrepancy)")
    print(f"\n  Top 10 features fake giống real nhất (KL thấp nhất):")
    print(kl_per_feat.head(10)[["feature","kl_div","ks_stat"]].to_string(index=False))

    # Save
    kl_per_feat.to_csv(f"{EVAL_DIR}/gan_quality_r{round_id}.csv", index=False)
    print(f"\n  Saved → {EVAL_DIR}/gan_quality_r{round_id}.csv")

    return {
        "mean_kl": mean_kl,
        "mean_ks": mean_ks,
        "mean_wd": mean_wd,
        "mmd": mmd,
        "per_feature": kl_per_feat,
    }


# ── Phase 3: Test Detector với adversarial samples ────────────────────────────

def phase3_evaluation(round_id: int = 1):
    print(f"\n{'='*58}")
    print(f"  Phase 3 — Detector vs GAN Adversarial Samples (Round {round_id})")
    print(f"{'='*58}")

    # Load đúng detector theo round
    detector = load_detector(round_id=round_id)
    det_label = "Detector v1 (Baseline)" if round_id == 1 else f"Detector v{round_id} (Adv. Round {round_id-1})"

    # Load test set thật
    X_test = np.load(f"{DATA_DIR}/X_test.npy")
    y_test = np.load(f"{DATA_DIR}/y_test.npy")

    # Load fake DDoS từ GAN
    fake_ddos = np.load(f"{GAN_DIR}/fake_ddos_r{round_id}.npy")

    # Load real DDoS từ test set để so sánh distribution
    real_ddos_test = X_test[y_test == 1]

    # Load feature names
    with open(f"{DATA_DIR}/feature_names.pkl", "rb") as f:
        feature_names = pickle.load(f)

    # ── 1. Baseline: Detector trên test set thật ──────────────────
    print("\n[Step 1] Detector on REAL test set (baseline check) ...")
    probs_real, preds_real = predict(detector, X_test)
    m_baseline = compute_metrics(y_test, preds_real, probs_real,
                                  f"Baseline — {det_label} on Real Test Set")
    print_metrics(m_baseline)

    # ── 2. Adversarial test set: real BENIGN + fake DDoS ──────────
    # Lấy BENIGN từ test set thật + fake DDoS từ GAN
    X_benign = X_test[y_test == 0]
    n_adv    = min(len(X_benign), len(fake_ddos))

    X_adv = np.concatenate([X_benign[:n_adv], fake_ddos[:n_adv]], axis=0)
    y_adv = np.concatenate([
        np.zeros(n_adv, dtype=np.int64),   # BENIGN = 0
        np.ones(n_adv,  dtype=np.int64),   # DDoS = 1
    ])

    print(f"\n[Step 2] Adversarial test set: {n_adv:,} BENIGN + {n_adv:,} fake DDoS")
    probs_adv, preds_adv = predict(detector, X_adv)
    m_adv = compute_metrics(y_adv, preds_adv, probs_adv,
                             f"Phase 3 — Detector vs GAN Round {round_id}")
    print_metrics(m_adv)

    # ── 3. GAN Quality (KL + KS) ──────────────────────────────────
    print(f"\n[Step 3] GAN Quality Metrics ...")
    quality = evaluate_gan_quality(
        real_ddos_test, fake_ddos[:len(real_ddos_test)],
        feature_names, round_id
    )

    # ── 4. Summary ────────────────────────────────────────────────
    print(f"\n{'='*58}")
    print(f"  SUMMARY — Round {round_id}")
    print(f"{'='*58}")
    print(f"  {'Metric':<20} {'Real Test':>12} {'Adversarial':>12}")
    print(f"  {'-'*44}")
    print(f"  {'Accuracy':<20} {m_baseline['accuracy']:>12.4f} {m_adv['accuracy']:>12.4f}")
    print(f"  {'F1 Score':<20} {m_baseline['f1']:>12.4f} {m_adv['f1']:>12.4f}")
    print(f"  {'AUC-ROC':<20} {m_baseline['auc']:>12.4f} {m_adv['auc']:>12.4f}")
    print(f"  {'FNR':<20} {m_baseline['fnr']:>12.4f} {m_adv['fnr']:>12.4f}")
    print(f"  {'Mean KL Div':<20} {'—':>12} {quality['mean_kl']:>12.4f}")
    print(f"  {'Mean KS Stat':<20} {'—':>12} {quality['mean_ks']:>12.4f}")
    print(f"  {'Wasserstein':<20} {'—':>12} {quality['mean_wd']:>12.4f}")
    print(f"  {'MMD (RBF)':<20} {'—':>12} {quality['mmd']:>12.4f}")
    print(f"{'='*58}")

    # Drop in F1
    f1_drop = m_baseline['f1'] - m_adv['f1']
    print(f"\n  📉 F1 Drop : {f1_drop:.4f} ({f1_drop*100:.1f}%)")
    if m_adv['f1'] < 0.80:
        print(f"  ✅ GAN bypass thành công — F1 giảm xuống {m_adv['f1']:.4f}")
    elif m_adv['f1'] < 0.90:
        print(f"  ⚠️  GAN bypass một phần — F1 = {m_adv['f1']:.4f}")
    else:
        print(f"  ❌ GAN chưa bypass hiệu quả — F1 vẫn cao: {m_adv['f1']:.4f}")
        print(f"     → Có thể FoolRate cao nhưng fake samples vẫn bị detect")
        print(f"     → Xem KL/KS để hiểu tại sao")

    # Save tất cả metrics để dùng cho arms race chart
    all_metrics = {
        "baseline": m_baseline,
        "phase3"  : m_adv,
        "quality" : {
            "mean_kl": quality["mean_kl"], 
            "mean_ks": quality["mean_ks"],
            "mean_wd": quality["mean_wd"],
            "mmd": quality["mmd"]
        },
    }
    np.save(f"{EVAL_DIR}/phase3_metrics_r{round_id}.npy", all_metrics)
    print(f"\n  Saved → {EVAL_DIR}/phase3_metrics_r{round_id}.npy")

    return all_metrics


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    round_id = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    phase3_evaluation(round_id=round_id)