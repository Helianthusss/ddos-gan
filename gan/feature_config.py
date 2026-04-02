"""
Feature Classification — Functional vs Non-functional
Dựa trên network traffic semantics của CIC-DDoS2019.

Functional features: định nghĩa bản chất của DDoS attack
→ KHÔNG được modify (constraint loss penalize nếu bị thay đổi)

Non-functional features: statistical/timing features
→ CÓ THỂ modify để bypass detector mà không làm mất bản chất DDoS
"""

import numpy as np
import pickle

FUNCTIONAL_FEATURES = [
    "Protocol", "Flow Duration",
    "FIN Flag Count", "SYN Flag Count", "RST Flag Count",
    "PSH Flag Count", "ACK Flag Count", "URG Flag Count",
    "CWE Flag Count", "ECE Flag Count",
    "Fwd PSH Flags", "Bwd PSH Flags",
    "Fwd URG Flags", "Bwd URG Flags",
    "Fwd Header Length", "Bwd Header Length",
]

NON_FUNCTIONAL_FEATURES = [
    "Total Fwd Packets", "Total Backward Packets",
    "Total Length of Fwd Packets", "Total Length of Bwd Packets",
    "Fwd Packet Length Max", "Fwd Packet Length Min",
    "Fwd Packet Length Mean", "Fwd Packet Length Std",
    "Bwd Packet Length Max", "Bwd Packet Length Min",
    "Bwd Packet Length Mean", "Bwd Packet Length Std",
    "Flow Bytes/s", "Flow Packets/s",
    "Flow IAT Mean", "Flow IAT Std", "Flow IAT Max", "Flow IAT Min",
    "Fwd IAT Total", "Fwd IAT Mean", "Fwd IAT Std", "Fwd IAT Max", "Fwd IAT Min",
    "Bwd IAT Total", "Bwd IAT Mean", "Bwd IAT Std", "Bwd IAT Max", "Bwd IAT Min",
    "Fwd Packets/s", "Bwd Packets/s",
    "Min Packet Length", "Max Packet Length",
    "Packet Length Mean", "Packet Length Std", "Packet Length Variance",
    "Down/Up Ratio", "Average Packet Size",
    "Avg Fwd Segment Size", "Avg Bwd Segment Size",
    "Fwd Avg Bytes/Bulk", "Fwd Avg Packets/Bulk", "Fwd Avg Bulk Rate",
    "Bwd Avg Bytes/Bulk", "Bwd Avg Packets/Bulk", "Bwd Avg Bulk Rate",
    "Subflow Fwd Packets", "Subflow Fwd Bytes",
    "Subflow Bwd Packets", "Subflow Bwd Bytes",
    "Init_Win_bytes_forward", "Init_Win_bytes_backward",
    "act_data_pkt_fwd", "min_seg_size_forward",
    "Active Mean", "Active Std", "Active Max", "Active Min",
    "Idle Mean", "Idle Std", "Idle Max", "Idle Min",
]


class FeatureConfig:
    def __init__(self, feature_names_path: str = "data/processed/feature_names.pkl"):
        with open(feature_names_path, "rb") as f:
            self.feature_names = pickle.load(f)
        self.n_features = len(self.feature_names)

        name_map = {n.strip().lower(): i for i, n in enumerate(self.feature_names)}

        self.functional_idx, self.functional_names = [], []
        self.non_functional_idx, self.non_functional_names = [], []

        for feat in FUNCTIONAL_FEATURES:
            idx = name_map.get(feat.strip().lower())
            if idx is not None:
                self.functional_idx.append(idx)
                self.functional_names.append(self.feature_names[idx])

        for feat in NON_FUNCTIONAL_FEATURES:
            idx = name_map.get(feat.strip().lower())
            if idx is not None:
                self.non_functional_idx.append(idx)
                self.non_functional_names.append(self.feature_names[idx])

        # Unlabeled → non-functional by default
        labeled = set(self.functional_idx) | set(self.non_functional_idx)
        for i in range(self.n_features):
            if i not in labeled:
                self.non_functional_idx.append(i)
                self.non_functional_names.append(self.feature_names[i])

        self.functional_idx     = sorted(self.functional_idx)
        self.non_functional_idx = sorted(self.non_functional_idx)

        print(f"[FeatureConfig] Total: {self.n_features} | "
              f"Functional: {len(self.functional_idx)} | "
              f"Non-functional: {len(self.non_functional_idx)}")

    def summary(self):
        print("\n  Functional features (constraint protected):")
        for n in self.functional_names:
            print(f"    - {n}")

    def get_valid_ranges(self, X_real_ddos: np.ndarray) -> dict:
        """[min_p1, max_p99] của mỗi functional feature trên real DDoS."""
        ranges = {}
        for i, name in zip(self.functional_idx, self.functional_names):
            col = X_real_ddos[:, i]
            ranges[name] = {
                "min":  float(np.percentile(col, 1)),
                "max":  float(np.percentile(col, 99)),
                "mean": float(np.mean(col)),
                "std":  float(np.std(col)),
            }
        return ranges


if __name__ == "__main__":
    cfg = FeatureConfig()
    cfg.summary()
    X_train = np.load("data/processed/X_train.npy")
    y_train = np.load("data/processed/y_train.npy")
    ranges  = cfg.get_valid_ranges(X_train[y_train == 1])
    print(f"\n  Valid ranges (top 5):")
    for name, r in list(ranges.items())[:5]:
        print(f"    {name:<30} [{r['min']:.3f}, {r['max']:.3f}]")