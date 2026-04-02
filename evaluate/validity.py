"""
Validity Score
Đo % fake samples có functional features nằm trong valid range của real DDoS.
"""

import numpy as np
import json, os
from gan.feature_config import FeatureConfig

DATA_DIR = "data/processed"
GAN_DIR  = "gan"
EVAL_DIR = "evaluate"
os.makedirs(EVAL_DIR, exist_ok=True)


def compute_validity_score(fake: np.ndarray, valid_ranges: dict,
                            cfg: FeatureConfig) -> dict:
    n         = len(fake)
    per_feat  = {}
    all_valid = np.ones(n, dtype=bool)

    for i, name in zip(cfg.functional_idx, cfg.functional_names):
        col   = fake[:, i]
        lo    = valid_ranges[name]["min"]
        hi    = valid_ranges[name]["max"]
        valid = (col >= lo) & (col <= hi)
        per_feat[name] = float(valid.mean())
        all_valid &= valid

    return {"overall": float(all_valid.mean()), "per_feature": per_feat}


def run_validity(round_id: int = 1):
    print(f"\n{'='*55}")
    print(f"  Validity Score — GAN Round {round_id}")
    print(f"{'='*55}")

    cfg     = FeatureConfig()
    X_train = np.load(f"{DATA_DIR}/X_train.npy")
    y_train = np.load(f"{DATA_DIR}/y_train.npy")
    fake    = np.load(f"{GAN_DIR}/fake_ddos_r{round_id}.npy")

    valid_ranges = cfg.get_valid_ranges(X_train[y_train == 1])
    result       = compute_validity_score(fake, valid_ranges, cfg)

    print(f"\n  Overall Validity Score : {result['overall']:.4f} "
          f"({result['overall']*100:.1f}% samples fully valid)")
    print(f"\n  Per-feature validity:")

    # Sort theo score tăng dần — feature nào GAN vi phạm nhiều nhất ở trên
    for name, score in sorted(result["per_feature"].items(), key=lambda x: x[1]):
        bar = "█" * int(score * 20)
        print(f"    {name:<30} {score:.3f}  {bar}")

    out  = {"round_id": round_id, "overall_validity": result["overall"],
            "per_feature": result["per_feature"]}
    path = f"{EVAL_DIR}/validity_r{round_id}.json"
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n  Saved → {path}")
    return result


if __name__ == "__main__":
    import sys
    round_id = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    run_validity(round_id)