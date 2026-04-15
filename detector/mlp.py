"""
Day 1 - MLP Detector (Phase 1 Baseline)
Train một MLP đơn giản để detect DDoS.
Target: F1 > 95%, AUC > 97%
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.metrics import (
    f1_score, accuracy_score, roc_auc_score,
    confusion_matrix, classification_report
)
import pickle
import os

# ── Config ────────────────────────────────────────────────────────────────────
DATA_DIR    = "data/processed"
SAVE_DIR    = "detector"
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RANDOM_SEED = 42

BATCH_SIZE  = 512
EPOCHS      = 30
LR          = 1e-3
HIDDEN_DIMS = [256, 128, 64]   # MLP layers
DROPOUT     = 0.3

torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


# ── Model ─────────────────────────────────────────────────────────────────────

class MLPDetector(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: list, dropout: float = 0.3):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for h in hidden_dims:
            layers += [
                nn.Linear(prev_dim, h),
                nn.BatchNorm1d(h),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
            prev_dim = h
        layers.append(nn.Linear(prev_dim, 1))   # Binary output
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(1)   # shape: (batch,)


# ── Data Loader ───────────────────────────────────────────────────────────────

def load_tensors():
    print(f"[INFO] Loading processed data from {DATA_DIR}/ ...")
    X_train = torch.tensor(np.load(f"{DATA_DIR}/X_train.npy"), dtype=torch.float32)
    X_val   = torch.tensor(np.load(f"{DATA_DIR}/X_val.npy"),   dtype=torch.float32)
    X_test  = torch.tensor(np.load(f"{DATA_DIR}/X_test.npy"),  dtype=torch.float32)
    y_train = torch.tensor(np.load(f"{DATA_DIR}/y_train.npy"), dtype=torch.float32)
    y_val   = torch.tensor(np.load(f"{DATA_DIR}/y_val.npy"),   dtype=torch.float32)
    y_test  = torch.tensor(np.load(f"{DATA_DIR}/y_test.npy"),  dtype=torch.float32)
    print(f"      Input dim: {X_train.shape[1]} | Train: {len(X_train):,}")
    return X_train, X_val, X_test, y_train, y_val, y_test


# ── Training ──────────────────────────────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
        optimizer.zero_grad()
        logits = model(X_batch)
        loss = criterion(logits, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(y_batch)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    all_probs, all_labels = [], []
    for X_batch, y_batch in loader:
        X_batch = X_batch.to(DEVICE)
        logits = model(X_batch)
        probs = torch.sigmoid(logits).cpu().numpy()
        all_probs.extend(probs)
        all_labels.extend(y_batch.numpy())

    all_probs  = np.array(all_probs)
    all_labels = np.array(all_labels)
    preds = (all_probs >= 0.5).astype(int)

    return {
        "accuracy": accuracy_score(all_labels, preds),
        "f1"      : f1_score(all_labels, preds),
        "auc"     : roc_auc_score(all_labels, all_probs),
        "fnr"     : 1 - (preds[all_labels == 1].sum() / (all_labels == 1).sum()),
        "probs"   : all_probs,
        "labels"  : all_labels,
    }


def train(model, train_loader, val_loader):
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=3, factor=0.5, verbose=True
    )

    best_f1   = 0.0
    best_path = f"{SAVE_DIR}/detector_best.pt"
    os.makedirs(SAVE_DIR, exist_ok=True)

    print(f"\n[INFO] Training on {DEVICE} | Epochs: {EPOCHS} | Batch: {BATCH_SIZE}")
    print("-" * 65)
    print(f"{'Epoch':>5} | {'Loss':>8} | {'Val Acc':>8} | {'Val F1':>8} | {'Val AUC':>8} | {'FNR':>6}")
    print("-" * 65)

    for epoch in range(1, EPOCHS + 1):
        loss = train_one_epoch(model, train_loader, optimizer, criterion)
        val_metrics = evaluate(model, val_loader)

        scheduler.step(1 - val_metrics["f1"])

        print(
            f"{epoch:>5} | {loss:>8.4f} | "
            f"{val_metrics['accuracy']:>8.4f} | "
            f"{val_metrics['f1']:>8.4f} | "
            f"{val_metrics['auc']:>8.4f} | "
            f"{val_metrics['fnr']:>6.4f}"
        )

        # Save best model
        if val_metrics["f1"] > best_f1:
            best_f1 = val_metrics["f1"]
            torch.save(model.state_dict(), best_path)

    print("-" * 65)
    print(f"\n✅ Best Val F1: {best_f1:.4f} → saved to {best_path}")
    return best_path


# ── Final Evaluation ──────────────────────────────────────────────────────────

def final_eval(model, test_loader, phase: str = "Phase 1 (Baseline)"):
    metrics = evaluate(model, test_loader)
    preds = (metrics["probs"] >= 0.5).astype(int)

    print(f"\n{'='*55}")
    print(f"  {phase} — Test Set Results")
    print(f"{'='*55}")
    print(f"  Accuracy : {metrics['accuracy']:.4f}")
    print(f"  F1 Score : {metrics['f1']:.4f}")
    print(f"  AUC-ROC  : {metrics['auc']:.4f}")
    print(f"  FNR      : {metrics['fnr']:.4f}  ← DDoS bị bỏ sót")
    print(f"\n  Confusion Matrix:")
    cm = confusion_matrix(metrics["labels"], preds)
    print(f"    TN={cm[0,0]:,}  FP={cm[0,1]:,}")
    print(f"    FN={cm[1,0]:,}  TP={cm[1,1]:,}")
    print(f"\n  Classification Report:")
    print(classification_report(metrics["labels"], preds,
                                target_names=["BENIGN", "DDoS"]))
    print(f"{'='*55}")

    # Save metrics để dùng lại ở các phase sau
    np.save(f"{SAVE_DIR}/phase1_metrics.npy", metrics["probs"])
    np.save(f"{SAVE_DIR}/phase1_labels.npy",  metrics["labels"])
    print(f"  Saved probs + labels → {SAVE_DIR}/phase1_*.npy")
    return metrics


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    X_train, X_val, X_test, y_train, y_val, y_test = load_tensors()

    input_dim = X_train.shape[1]
    model = MLPDetector(input_dim, HIDDEN_DIMS, DROPOUT).to(DEVICE)
    print(f"\n[INFO] Model architecture:")
    print(model)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"       Total params: {total_params:,}")

    train_loader = DataLoader(
        TensorDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True
    )
    val_loader = DataLoader(
        TensorDataset(X_val, y_val), batch_size=BATCH_SIZE
    )
    test_loader = DataLoader(
        TensorDataset(X_test, y_test), batch_size=BATCH_SIZE
    )

    best_path = train(model, train_loader, val_loader)

    # Load best và evaluate trên test set
    model.load_state_dict(torch.load(best_path, map_location=DEVICE, weights_only=True))
    final_eval(model, test_loader, phase="Phase 1 (Baseline)")

    # Save model config để load lại ở các phase sau
    config = {"input_dim": input_dim, "hidden_dims": HIDDEN_DIMS, "dropout": DROPOUT}
    with open(f"{SAVE_DIR}/model_config.pkl", "wb") as f:
        pickle.dump(config, f)
    print(f"\n  Saved model config → {SAVE_DIR}/model_config.pkl")


if __name__ == "__main__":
    main()