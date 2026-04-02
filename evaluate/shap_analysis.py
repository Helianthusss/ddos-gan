"""
SHAP Analysis
So sánh feature importance của Detector v1 vs v2.
Finding kỳ vọng: Detector v2 shift attention sang functional features
mà GAN khó fake — đây là insight mới cho paper.
Dùng GradientExplainer thay vì DeepExplainer để tránh shape issues.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pickle, os, torch
import shap

from detector.mlp import MLPDetector
from gan.feature_config import FeatureConfig

DATA_DIR     = "data/processed"
DETECTOR_DIR = "detector"
EVAL_DIR     = "evaluate"
DEVICE       = torch.device("cpu")   # SHAP cần CPU
os.makedirs(EVAL_DIR, exist_ok=True)


# ── Wrapper để SHAP nhận output 2D ───────────────────────────────────────────

class MLPWrapper(torch.nn.Module):
    """Wrap MLPDetector để output shape (batch, 1) thay vì (batch,)"""
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        out = self.model(x)          # shape: (batch,)
        return out.unsqueeze(1)      # shape: (batch, 1)


def load_mlp(path: str) -> MLPDetector:
    with open(f"{DETECTOR_DIR}/model_config.pkl", "rb") as f:
        cfg = pickle.load(f)
    model = MLPDetector(cfg["input_dim"], cfg["hidden_dims"], cfg["dropout"])
    model.load_state_dict(torch.load(path, map_location=DEVICE, weights_only=True))
    model.eval()
    return model


def get_shap_values(model: MLPDetector, X_bg: np.ndarray,
                    X_test: np.ndarray, n_bg: int = 100,
                    n_test: int = 300) -> np.ndarray:
    """
    Dùng GradientExplainer — ổn định hơn DeepExplainer với MLP output 1D.
    Returns: mean absolute SHAP per feature, shape (n_features,)
    """
    wrapped = MLPWrapper(model)
    wrapped.eval()

    bg   = torch.tensor(X_bg[:n_bg],   dtype=torch.float32)
    test = torch.tensor(X_test[:n_test], dtype=torch.float32)

    explainer   = shap.GradientExplainer(wrapped, bg)
    shap_values = explainer.shap_values(test)   # list of arrays

    # shap_values là list[array] — lấy phần tử đầu
    if isinstance(shap_values, list):
        sv = np.array(shap_values[0])
    else:
        sv = np.array(shap_values)

    # Loại bỏ mọi extra dimension (e.g. (n_test, n_features, 1) → (n_test, n_features))
    sv = sv.squeeze()

    # Nếu sau squeeze vẫn còn 2D → axis-0 là samples, axis-1 là features
    if sv.ndim == 2:
        result = np.abs(sv).mean(axis=0)
    elif sv.ndim == 1:
        # Đã là (n_features,) — trường hợp n_test=1
        result = np.abs(sv)
    else:
        # Fallback: flatten hết rồi không mean được, raise rõ lỗi
        raise ValueError(f"Unexpected sv shape after squeeze: {sv.shape}")

    return result.flatten()   # Đảm bảo 1D tuyệt đối


def run_shap_analysis():
    print("\n[SHAP] Loading data & models ...")
    X_train = np.load(f"{DATA_DIR}/X_train.npy")
    X_test  = np.load(f"{DATA_DIR}/X_test.npy")
    y_train = np.load(f"{DATA_DIR}/y_train.npy")

    with open(f"{DATA_DIR}/feature_names.pkl", "rb") as f:
        feature_names = pickle.load(f)
    # Đảm bảo feature_names là flat list of str (tránh nested list / ndarray)
    if isinstance(feature_names, np.ndarray):
        feature_names = feature_names.flatten().tolist()
    elif isinstance(feature_names, list) and len(feature_names) > 0 and isinstance(feature_names[0], (list, np.ndarray)):
        feature_names = [n for sub in feature_names for n in sub]
    else:
        feature_names = list(feature_names)  # copy

    cfg = FeatureConfig()
    # Ưūu tiên dùng cfg.feature_names nếu match về số lượng (guaranteed flat)
    if len(cfg.feature_names) == len(feature_names):
        feature_names = list(cfg.feature_names)

    model_v1 = load_mlp(f"{DETECTOR_DIR}/detector_best.pt")
    model_v2 = load_mlp(f"{DETECTOR_DIR}/detector_adv_r1.pt")

    # Dùng DDoS samples làm background để SHAP focus vào DDoS detection
    X_ddos = X_train[y_train == 1]

    print("[SHAP] Computing SHAP values for Detector v1 ...")
    shap_v1 = get_shap_values(model_v1, X_ddos, X_test, n_bg=100, n_test=300)
    print(f"       shap_v1 shape={shap_v1.shape}, dtype={shap_v1.dtype}")

    print("[SHAP] Computing SHAP values for Detector v2 ...")
    shap_v2 = get_shap_values(model_v2, X_ddos, X_test, n_bg=100, n_test=300)
    print(f"       shap_v2 shape={shap_v2.shape}, dtype={shap_v2.dtype}")

    # ── Compare functional vs non-functional ─────────────────────────────────
    func_idx    = cfg.functional_idx
    nonfunc_idx = cfg.non_functional_idx

    func_imp_v1    = shap_v1[func_idx].mean()    if len(func_idx) > 0 else 0
    func_imp_v2    = shap_v2[func_idx].mean()    if len(func_idx) > 0 else 0
    nonfunc_imp_v1 = shap_v1[nonfunc_idx].mean() if len(nonfunc_idx) > 0 else 0
    nonfunc_imp_v2 = shap_v2[nonfunc_idx].mean() if len(nonfunc_idx) > 0 else 0

    print(f"\n{'='*58}")
    print(f"  SHAP — Functional vs Non-functional importance")
    print(f"{'='*58}")
    print(f"  {'Group':<22} {'Detector v1':>12} {'Detector v2':>12} {'Change':>10}")
    print(f"  {'-'*58}")
    print(f"  {'Functional':<22} {func_imp_v1:>12.6f} {func_imp_v2:>12.6f} "
          f"{(func_imp_v2-func_imp_v1):>+10.6f}")
    print(f"  {'Non-functional':<22} {nonfunc_imp_v1:>12.6f} {nonfunc_imp_v2:>12.6f} "
          f"{(nonfunc_imp_v2-nonfunc_imp_v1):>+10.6f}")
    print(f"{'='*58}")

    shift = func_imp_v2 - func_imp_v1
    if shift > 0:
        print(f"\n  ✅ Finding: Detector v2 +{shift:.6f} attention trên functional features")
        print(f"     → Harder for GAN to bypass in Round 2")
    else:
        print(f"\n  ⚠️  No shift detected (diff={shift:.6f})")
        print(f"     → GAN Round 1 bypass bằng non-functional features — consistent với Validity=0")

    # ── Plot top 20 features ──────────────────────────────────────────────────
    top_idx     = [int(i) for i in np.argsort(shap_v1 + shap_v2)[-20:][::-1]]
    names       = [feature_names[i] for i in top_idx]
    colors      = ["#F44336" if i in func_idx else "#2196F3" for i in top_idx]
    top_idx_arr = np.array(top_idx, dtype=np.int64)

    fig, axes = plt.subplots(1, 2, figsize=(16, 7), sharey=True)
    for ax, sv, title in zip(
        axes,
        [shap_v1[top_idx_arr], shap_v2[top_idx_arr]],
        ["Detector v1 (Baseline)", "Detector v2 (After Adv. Training)"]
    ):
        ax.barh(names[::-1], sv[::-1], color=colors[::-1])
        ax.set_xlabel("Mean |SHAP value|")
        ax.set_title(title, fontsize=12)

    from matplotlib.patches import Patch
    legend = [Patch(color="#F44336", label="Functional"),
              Patch(color="#2196F3", label="Non-functional")]
    axes[0].legend(handles=legend, loc="lower right", fontsize=9)

    plt.suptitle("SHAP Feature Importance: Detector v1 vs v2",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig_path = f"{EVAL_DIR}/shap_v1_vs_v2.png"
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  Saved → {fig_path}")

    # Save raw
    np.save(f"{EVAL_DIR}/shap_v1.npy", shap_v1)
    np.save(f"{EVAL_DIR}/shap_v2.npy", shap_v2)
    print(f"  Saved → {EVAL_DIR}/shap_v1.npy, shap_v2.npy")
    return shap_v1, shap_v2


if __name__ == "__main__":
    run_shap_analysis()