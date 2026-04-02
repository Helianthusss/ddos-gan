"""
Arms Race Summary
Tổng hợp tất cả metrics từ Phase 1 → 5 thành 1 table.
Output: console + JSON + CSV — dùng trực tiếp cho paper Table 2.
"""

import numpy as np
import json, os, csv
from pathlib import Path

EVAL_DIR     = "evaluate"
DETECTOR_DIR = "detector"
GAN_DIR      = "gan"
os.makedirs(EVAL_DIR, exist_ok=True)


def load_json(path: str) -> dict:
    if not Path(path).exists():
        return {}
    with open(path) as f:
        return json.load(f)


def build_summary() -> list:
    rows = []

    # Phase 1 — Baseline
    p3 = np.load(f"{EVAL_DIR}/phase3_metrics_r1.npy", allow_pickle=True).item()
    b  = p3.get("baseline", {})
    rows.append({
        "Phase": "Phase 1 — Baseline (MLP v1)",
        "Detector": "MLP v1",
        "Test Set": "Real",
        "Accuracy": b.get("accuracy", "-"),
        "F1": b.get("f1", "-"),
        "AUC": b.get("auc", "-"),
        "FNR": b.get("fnr", "-"),
        "FoolRate": "-",
        "Validity": "-",
    })

    # Phase 3 — GAN Round 1 attack
    p3adv = p3.get("phase3", {})
    qual  = p3.get("quality", {})
    v1    = load_json(f"{EVAL_DIR}/validity_r1.json")
    rows.append({
        "Phase": "Phase 3 — GAN Round 1 attack",
        "Detector": "MLP v1",
        "Test Set": "Adversarial",
        "Accuracy": p3adv.get("accuracy", "-"),
        "F1": p3adv.get("f1", "-"),
        "AUC": p3adv.get("auc", "-"),
        "FNR": p3adv.get("fnr", "-"),
        "FoolRate": 1 - p3adv.get("f1", 0),
        "Validity": v1.get("overall_validity", "-"),
    })

    # Phase 4 — Adversarial Training
    p4 = np.load(f"{EVAL_DIR}/phase4_metrics_r1.npy", allow_pickle=True).item()
    for key, label, tset in [
        ("metrics_real", "Phase 4 — Adv Training (MLP v2)", "Real"),
        ("metrics_adv",  "Phase 4 — Adv Training (MLP v2)", "Adversarial"),
    ]:
        m = p4.get(key, {})
        rows.append({
            "Phase": label,
            "Detector": "MLP v2",
            "Test Set": tset,
            "Accuracy": m.get("accuracy", "-"),
            "F1": m.get("f1", "-"),
            "AUC": m.get("auc", "-"),
            "FNR": m.get("fnr", "-"),
            "FoolRate": "-",
            "Validity": "-",
        })

    # Phase 5 — GAN Round 2 (nếu có)
    p3r2_path = f"{EVAL_DIR}/phase3_metrics_r2.npy"
    if Path(p3r2_path).exists():
        p3r2 = np.load(p3r2_path, allow_pickle=True).item()
        p5   = p3r2.get("phase3", {})
        v2   = load_json(f"{EVAL_DIR}/validity_r2.json")
        rows.append({
            "Phase": "Phase 5 — GAN Round 2 attack",
            "Detector": "MLP v2",
            "Test Set": "Adversarial",
            "Accuracy": p5.get("accuracy", "-"),
            "F1": p5.get("f1", "-"),
            "AUC": p5.get("auc", "-"),
            "FNR": p5.get("fnr", "-"),
            "FoolRate": 1 - p5.get("f1", 0),
            "Validity": v2.get("overall_validity", "-"),
        })

    # Transferability
    t1_path = f"{EVAL_DIR}/transferability_r1.json"
    if Path(t1_path).exists():
        t1 = load_json(t1_path)
        for det_result in t1:
            if det_result.get("detector") == "rf":
                rows.append({
                    "Phase": "Phase 3 — GAN Round 1 (RF)",
                    "Detector": "Random Forest",
                    "Test Set": "Adversarial",
                    "Accuracy": det_result.get("accuracy", "-"),
                    "F1": det_result.get("f1", "-"),
                    "AUC": det_result.get("auc", "-"),
                    "FNR": det_result.get("fnr", "-"),
                    "FoolRate": det_result.get("fool_rate", "-"),
                    "Validity": "-",
                })

    return rows


def fmt(v, decimals=4):
    if isinstance(v, float):
        return f"{v:.{decimals}f}"
    return str(v)


def print_table(rows: list):
    cols = ["Phase", "Detector", "Test Set", "F1", "AUC", "FNR", "FoolRate", "Validity"]
    widths = {c: max(len(c), max(len(fmt(r.get(c, "-"))) for r in rows)) for c in cols}

    header = "  " + " | ".join(c.ljust(widths[c]) for c in cols)
    sep    = "  " + "-+-".join("-" * widths[c] for c in cols)

    print(f"\n{'='*len(header)}")
    print("  ARMS RACE — Full Summary (Paper Table 2)")
    print(f"{'='*len(header)}")
    print(header)
    print(sep)
    for r in rows:
        print("  " + " | ".join(fmt(r.get(c, "-")).ljust(widths[c]) for c in cols))
    print(f"{'='*len(header)}")


def save_outputs(rows: list):
    # JSON
    json_path = f"{EVAL_DIR}/arms_race_summary.json"
    with open(json_path, "w") as f:
        json.dump(rows, f, indent=2, default=str)

    # CSV — paste thẳng vào paper
    csv_path = f"{EVAL_DIR}/arms_race_summary.csv"
    cols = ["Phase", "Detector", "Test Set", "Accuracy", "F1", "AUC", "FNR", "FoolRate", "Validity"]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: fmt(r.get(c, "-")) for c in cols})

    print(f"\n  Saved → {json_path}")
    print(f"  Saved → {csv_path}  ← paste vào paper")


def main():
    rows = build_summary()
    print_table(rows)
    save_outputs(rows)


if __name__ == "__main__":
    main()