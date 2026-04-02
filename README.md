# DDoS-GAN: Adversarial DDoS Traffic Generation for IDS Robustness Testing

Adversarial Machine Learning framework sử dụng WGAN-GP để sinh DDoS traffic giả nhằm đánh giá và tăng cường độ bền của ML-based Intrusion Detection Systems (IDS).

## Overview

Pipeline gồm 5 phases theo framework Arms Race:

```
Phase 1: Train baseline MLP detector trên CIC-DDoS2019
    ↓
Phase 2: Train WGAN-GP để sinh adversarial DDoS samples (bypass detector)
    ↓
Phase 3: Evaluate detector với adversarial samples → F1 drop
    ↓
Phase 4: Adversarial Training → retrain detector với fake samples
    ↓
Phase 5: Train GAN Round 2 → attack detector đã được hardened
```

## Key Results

| Phase | Detector | Test Set | F1 | FoolRate | Validity |
|---|---|---|---|---|---|
| Phase 1 — Baseline | MLP v1 | Real | 0.9996 | — | — |
| Phase 3 — GAN Round 1 | MLP v1 | Adversarial | 0.0000 | 1.0000 | 0.0000 |
| Phase 4 — Adv Training | MLP v2 | Real | 0.9996 | — | — |
| Phase 4 — Adv Training | MLP v2 | Adversarial | 0.9998 | 0.0000 | — |
| Phase 5 — GAN Round 2 | MLP v2 | Adversarial | 0.0000 | 1.0000 | 0.0000 |
| Transferability | Random Forest | Adversarial | 0.5325 | 0.6371 | — |

## Project Structure

```
ddos-gan/
├── main.py                     # Entry point
├── requirements.txt
│
├── data/
│   ├── preprocess.py           # Load, clean, balance, split, scale
│   └── processed/              # Preprocessed numpy arrays (git-ignored)
│
├── detector/
│   ├── mlp.py                  # MLP detector (PyTorch)
│   └── random_forest.py        # RF detector (sklearn, transferability)
│
├── gan/
│   ├── generator.py            # WGAN-GP Generator
│   ├── discriminator.py        # WGAN-GP Critic + gradient penalty
│   ├── feature_config.py       # Functional/non-functional feature split
│   └── train_wgan.py           # Training loop với auxiliary + FM loss
│
├── evaluate/
│   ├── phase3.py               # Detector vs GAN adversarial samples
│   ├── phase4.py               # Adversarial Training (mixed val set)
│   ├── validity.py             # Validity Score metric
│   ├── transferability.py      # Cross-architecture transferability
│   ├── shap_analysis.py        # SHAP feature importance v1 vs v2
│   └── arms_race_summary.py    # Full summary table cho paper
│
└── notebooks/
    └── eda.ipynb               # Exploratory Data Analysis
```

## Setup

### Requirements

- Python 3.11
- CUDA 11.8 (hoặc 12.1)
- GPU recommended (tested trên NVIDIA GPU với CUDA 11.8)

### Installation

```bash
# Clone repo
git clone https://github.com/<your-username>/ddos-gan.git
cd ddos-gan

# Tạo virtual environment
python -m venv .venv

# Activate
# Windows:
.venv\Scripts\activate
# Linux/Mac:
source .venv/bin/activate

# Install PyTorch với CUDA 11.8
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu118

# Install các thư viện còn lại
pip install -r requirements.txt
```

### Dataset

Download **CIC-DDoS2019** từ Kaggle:
- Link: https://www.kaggle.com/datasets/aymenabb/ddos-evaluation-dataset-cic-ddos2019
- File cần dùng: `Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv`
- Đặt vào: `data/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv`

## Usage

```bash
# Step 1: Preprocess data
python main.py preprocess

# Step 2: Train baseline MLP detector (Phase 1)
python main.py train

# Step 3: Train Random Forest detector
python main.py train_rf

# Step 4: Train WGAN-GP Round 1 (Phase 2)
python main.py gan 1

# Step 5: Evaluate Phase 3 (Detector vs GAN)
python main.py eval 1

# Step 6: Validity Score
python main.py validity 1

# Step 7: Transferability test (MLP + RF)
python main.py transferability 1

# Step 8: Adversarial Training (Phase 4)
python main.py adv_train 1

# Step 9: GAN Round 2 (Phase 5)
python main.py gan 2
python main.py eval 2
python main.py validity 2

# Step 10: SHAP analysis
python main.py shap

# Step 11: Full summary table
python main.py summary
```

## Methodology

### WGAN-GP Loss

```
L_Critic   = E[C(fake)] - E[C(real)] + λ_gp * GP
L_Generator = -E[C(fake)]
            + λ_adv * BCE(Detector(fake), BENIGN)   ← fool detector
            + λ_fm  * MSE(mean/std fake, mean/std real)  ← feature matching
```

### Feature Classification

80 features được phân thành 2 nhóm:
- **Functional (16 features):** Protocol, Flag counts, Flow Duration — định nghĩa bản chất DDoS, không được modify
- **Non-functional (64 features):** Byte/packet stats, IAT, window sizes — có thể modify để bypass detector

### Validity Score

Metric mới đo % fake samples có functional features nằm trong valid range `[p1, p99]` của real DDoS. Validity = 0 với unconstrained GAN chứng minh cần có constraint loss.

## TODO

- [ ] Thêm constraint loss vào GAN (penalize functional feature violations)
- [ ] Fix GAN Round 2 load `detector_adv_r1.pt` (arms race thực sự)
- [ ] Ablation study: GAN với vs không có constraint loss
- [ ] Cross-dataset evaluation (thêm Monday file từ CIC-DDoS2019)

## Requirements

Xem `requirements.txt`

## Citation

```bibtex
@misc{ddos-gan-2025,
  title  = {Adversarial DDoS Traffic Generation for IDS Robustness Testing using WGAN-GP},
  year   = {2025},
  note   = {Course project}
}
```

## License

MIT

## Presentation

Toàn bộ slide báo cáo quá trình nghiên cứu và kết quả được thiết kế chuyên nghiệp bằng LaTeX nằm trong thư mục `slides/`.
- `main.tex`: Source code LaTeX (20 trang) với biểu đồ TiKZ minh họa pipeline.
- `gen_figures.py`: Script tự động render đồ thị heatmap, timeline và phân phối.

Để biên dịch slide ra PDF:
```bash
cd slides
pdflatex main.tex
```