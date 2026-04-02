"""
Phase 4 — Adversarial Training
Trộn GAN-generated fake DDoS vào train set → Retrain Detector mới.
Mục tiêu: F1 recover lên ~90-95% sau khi học cách nhận ra fake DDoS.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pickle
import os
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score, confusion_matrix

from detector.mlp import MLPDetector, evaluate, final_eval

DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_DIR     = "data/processed"
DETECTOR_DIR = "detector"
GAN_DIR      = "gan"
EVAL_DIR     = "evaluate"

BATCH_SIZE   = 512
EPOCHS       = 30
LR           = 1e-3
RANDOM_SEED  = 42

torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


# ── Build Adversarial Train Set ───────────────────────────────────────────────

def build_adv_trainset(round_id: int = 1, mix_ratio: float = 0.5):
    """
    Trộn fake DDoS vào train set gốc.
    mix_ratio: tỉ lệ fake DDoS so với real DDoS trong train set mới.
               0.5 = số fake bằng 50% số real DDoS
               1.0 = số fake bằng 100% số real DDoS (double DDoS samples)
    """
    X_train = np.load(f"{DATA_DIR}/X_train.npy")
    y_train = np.load(f"{DATA_DIR}/y_train.npy")
    fake_ddos = np.load(f"{GAN_DIR}/fake_ddos_r{round_id}.npy")

    n_real_ddos = (y_train == 1).sum()
    n_fake_use  = int(n_real_ddos * mix_ratio)
    n_fake_use  = min(n_fake_use, len(fake_ddos))

    # Sample fake DDoS
    idx = np.random.choice(len(fake_ddos), n_fake_use, replace=False)
    fake_selected = fake_ddos[idx]
    fake_labels   = np.ones(n_fake_use, dtype=np.int64)

    # Concat
    X_new = np.concatenate([X_train, fake_selected], axis=0).astype(np.float32)
    y_new = np.concatenate([y_train, fake_labels],   axis=0).astype(np.int64)

    # Shuffle
    perm  = np.random.permutation(len(X_new))
    X_new = X_new[perm]
    y_new = y_new[perm]

    print(f"[INFO] Adversarial train set:")
    print(f"       Original  : {len(X_train):,} samples")
    print(f"       Fake DDoS added: {n_fake_use:,} (mix_ratio={mix_ratio})")
    print(f"       New total : {len(X_new):,} samples")
    print(f"       BENIGN    : {(y_new==0).sum():,}")
    print(f"       DDoS(real): {n_real_ddos:,}")
    print(f"       DDoS(fake): {n_fake_use:,}")
    return X_new, y_new


# ── Retrain Detector ──────────────────────────────────────────────────────────

def retrain_detector(X_train_adv, y_train_adv, round_id: int = 1):
    """Train Detector mới trên adversarial train set."""

    # Load val/test set thật (không thay đổi)
    X_val  = np.load(f"{DATA_DIR}/X_val.npy")
    y_val  = np.load(f"{DATA_DIR}/y_val.npy")
    X_test = np.load(f"{DATA_DIR}/X_test.npy")
    y_test = np.load(f"{DATA_DIR}/y_test.npy")

    # Load model config
    with open(f"{DETECTOR_DIR}/model_config.pkl", "rb") as f:
        cfg = pickle.load(f)

    input_dim = cfg["input_dim"]
    model = MLPDetector(cfg["input_dim"], cfg["hidden_dims"], cfg["dropout"]).to(DEVICE)

    # DataLoaders
    train_loader = DataLoader(
        TensorDataset(
            torch.tensor(X_train_adv, dtype=torch.float32),
            torch.tensor(y_train_adv, dtype=torch.float32),
        ),
        batch_size=BATCH_SIZE, shuffle=True
    )
    val_loader = DataLoader(
        TensorDataset(
            torch.tensor(X_val,  dtype=torch.float32),
            torch.tensor(y_val,  dtype=torch.float32),
        ),
        batch_size=BATCH_SIZE
    )
    test_loader = DataLoader(
        TensorDataset(
            torch.tensor(X_test, dtype=torch.float32),
            torch.tensor(y_test, dtype=torch.float32),
        ),
        batch_size=BATCH_SIZE
    )

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=3, factor=0.5
    )

    save_path = f"{DETECTOR_DIR}/detector_adv_r{round_id}.pt"
    best_f1   = 0.0

    print(f"\n[INFO] Retraining Detector (Phase 4) on adversarial data ...")
    print(f"       Device: {DEVICE} | Epochs: {EPOCHS} | Batch: {BATCH_SIZE}")
    print("-" * 65)
    print(f"{'Epoch':>5} | {'Loss':>8} | {'Val Acc':>8} | {'Val F1':>8} | {'Val AUC':>8} | {'FNR':>6}")
    print("-" * 65)

    for epoch in range(1, EPOCHS + 1):
        # Train
        model.train()
        total_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(X_batch), y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(y_batch)
        avg_loss = total_loss / len(train_loader.dataset)

        # Validate
        val_metrics = evaluate(model, val_loader)
        scheduler.step(1 - val_metrics["f1"])

        print(
            f"{epoch:>5} | {avg_loss:>8.4f} | "
            f"{val_metrics['accuracy']:>8.4f} | "
            f"{val_metrics['f1']:>8.4f} | "
            f"{val_metrics['auc']:>8.4f} | "
            f"{val_metrics['fnr']:>6.4f}"
        )

        if val_metrics["f1"] > best_f1:
            best_f1 = val_metrics["f1"]
            torch.save(model.state_dict(), save_path)

    print("-" * 65)
    print(f"\n✅ Best Val F1: {best_f1:.4f} → saved to {save_path}")

    # Final eval trên real test set
    model.load_state_dict(
        torch.load(save_path, map_location=DEVICE, weights_only=True)
    )
    print(f"\n[Phase 4] Detector v2 on REAL test set:")
    metrics_real = final_eval(model, test_loader,
                               phase=f"Phase 4 — Detector v2 (Real Test Set)")

    # Eval trên adversarial test set (real BENIGN + fake DDoS)
    fake_ddos = np.load(f"{GAN_DIR}/fake_ddos_r{round_id}.npy")
    X_benign  = X_test[y_test == 0]
    n_adv     = min(len(X_benign), len(fake_ddos))
    X_adv     = np.concatenate([X_benign[:n_adv], fake_ddos[:n_adv]])
    y_adv     = np.concatenate([
        np.zeros(n_adv, dtype=np.int64),
        np.ones(n_adv,  dtype=np.int64),
    ])
    adv_loader = DataLoader(
        TensorDataset(
            torch.tensor(X_adv, dtype=torch.float32),
            torch.tensor(y_adv.astype(np.float32)),
        ),
        batch_size=BATCH_SIZE
    )
    print(f"\n[Phase 4] Detector v2 on ADVERSARIAL test set (fake DDoS):")
    metrics_adv = final_eval(model, adv_loader,
                              phase=f"Phase 4 — Detector v2 (Adversarial Test)")

    # ── Arms Race Summary so far ──────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  ARMS RACE — Progress so far")
    print(f"{'='*60}")
    print(f"  {'Phase':<35} {'F1':>8} {'FNR':>8}")
    print(f"  {'-'*52}")

    # Load Phase 3 metrics
    p3 = np.load(f"{EVAL_DIR}/phase3_metrics_r1.npy", allow_pickle=True).item()
    print(f"  {'Phase 1 — Baseline (real data)':<35} {p3['baseline']['f1']:>8.4f} {p3['baseline']['fnr']:>8.4f}")
    print(f"  {'Phase 3 — After GAN Round 1':<35} {p3['phase3']['f1']:>8.4f} {p3['phase3']['fnr']:>8.4f}")
    print(f"  {'Phase 4 — Adv Training (real test)':<35} {metrics_real['f1']:>8.4f} {metrics_real['fnr']:>8.4f}")
    print(f"  {'Phase 4 — Adv Training (adv test)':<35} {metrics_adv['f1']:>8.4f} {metrics_adv['fnr']:>8.4f}")
    print(f"{'='*60}")
    print(f"\n  ➡️  Next: python main.py gan 2  (GAN Round 2 — Phase 5)")

    # Save
    result = {
        "metrics_real": metrics_real,
        "metrics_adv" : metrics_adv,
    }
    np.save(f"{EVAL_DIR}/phase4_metrics_r{round_id}.npy", result)
    print(f"  Saved → {EVAL_DIR}/phase4_metrics_r{round_id}.npy")
    return model, result


# ── Main ──────────────────────────────────────────────────────────────────────

def main(round_id: int = 1):
    print(f"\n{'='*60}")
    print(f"  Phase 4 — Adversarial Training (Round {round_id})")
    print(f"{'='*60}")
    X_train_adv, y_train_adv = build_adv_trainset(round_id=round_id, mix_ratio=1.0)
    retrain_detector(X_train_adv, y_train_adv, round_id=round_id)


if __name__ == "__main__":
    import sys
    round_id = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    main(round_id)