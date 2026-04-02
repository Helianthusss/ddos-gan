"""
Main entry point
Usage:
    python main.py preprocess               ← preprocess data
    python main.py train                    ← train MLP baseline (Phase 1)
    python main.py train_rf                 ← train Random Forest detector
    python main.py gan [round]              ← train WGAN-GP (round 1 hoặc 2)
    python main.py eval [round]             ← Phase 3 evaluation
    python main.py adv_train [round]        ← Phase 4 adversarial training
    python main.py validity [round]         ← validity score
    python main.py transferability [round]  ← transferability test (MLP + RF)
    python main.py shap                     ← SHAP v1 vs v2
    python main.py summary                  ← arms race full summary table
    python main.py feature_config           ← check feature split
"""

import sys


def run_preprocess():
    from data.preprocess import main as fn; fn()

def run_train():
    from detector.mlp import main as fn; fn()

def run_train_rf():
    from detector.random_forest import train_rf; train_rf()

def run_gan(round_id=1):
    from gan.train_wgan import train_wgan, generate_adversarial_samples
    train_wgan(round_id=round_id)
    generate_adversarial_samples(round_id=round_id, n_samples=20000)

def run_eval(round_id=1):
    from evaluate.phase3 import phase3_evaluation
    phase3_evaluation(round_id=round_id)

def run_adv_train(round_id=1):
    from evaluate.phase4 import main as fn; fn(round_id=round_id)

def run_validity(round_id=1):
    from evaluate.validity import run_validity
    run_validity(round_id=round_id)

def run_transferability(round_id=1):
    from evaluate.transferability import run_transferability
    run_transferability(round_id=round_id)

def run_shap():
    from evaluate.shap_analysis import run_shap_analysis
    run_shap_analysis()

def run_summary():
    from evaluate.arms_race_summary import main as fn; fn()

def run_feature_config():
    from gan.feature_config import FeatureConfig
    import numpy as np
    cfg = FeatureConfig()
    cfg.summary()
    X_train = np.load("data/processed/X_train.npy")
    y_train = np.load("data/processed/y_train.npy")
    ranges  = cfg.get_valid_ranges(X_train[y_train == 1])
    print(f"\n  Valid ranges (top 5):")
    for name, r in list(ranges.items())[:5]:
        print(f"    {name:<30} [{r['min']:.3f}, {r['max']:.3f}]")


if __name__ == "__main__":
    cmd      = sys.argv[1] if len(sys.argv) > 1 else "help"
    round_id = int(sys.argv[2]) if len(sys.argv) > 2 else 1

    dispatch = {
        "preprocess":      run_preprocess,
        "train":           run_train,
        "train_rf":        run_train_rf,
        "shap":            run_shap,
        "summary":         run_summary,
        "feature_config":  run_feature_config,
    }
    dispatch_round = {
        "gan":             run_gan,
        "eval":            run_eval,
        "adv_train":       run_adv_train,
        "validity":        run_validity,
        "transferability": run_transferability,
    }

    if cmd in dispatch:
        dispatch[cmd]()
    elif cmd in dispatch_round:
        dispatch_round[cmd](round_id)
    else:
        print(__doc__)