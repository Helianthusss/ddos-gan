"""
Day 2 — WGAN-GP Training
Mục tiêu: Generator học sinh fake DDoS traffic đủ tinh vi để
          bypass Detector (white-box attack via auxiliary loss).

Loss tổng quát của Generator:
    L_G = -E[Critic(fake)]                           ← WGAN generator loss
        + lambda_adv  * BCE(Detector(fake), 0)       ← fool detector (bypass)
        + lambda_con  * MSE(fake[func], mean[func])  ← constraint: giữ functional features
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pickle
import os
from tqdm import tqdm

from gan.generator import Generator
from gan.discriminator import Critic, gradient_penalty
from gan.feature_config import FeatureConfig
from detector.mlp import MLPDetector

# ── Config ────────────────────────────────────────────────────────────────────
DATA_DIR     = "data/processed"
DETECTOR_DIR = "detector"
SAVE_DIR     = "gan"

DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RANDOM_SEED  = 42

# GAN hyperparams
LATENT_DIM   = 64
BATCH_SIZE   = 512
EPOCHS       = 200
LR_G         = 1e-4
LR_C         = 1e-4
N_CRITIC     = 5          # Train critic N_CRITIC lần mỗi bước Generator
LAMBDA_GP    = 10.0       # Gradient penalty weight
LAMBDA_ADV   = 1.0        # Auxiliary detector loss weight
LAMBDA_CON   = 5.0        # Constraint loss weight (0 = ablation without constraint)
LOG_EVERY    = 20         # Print log mỗi N epoch

GEN_HIDDEN   = [128, 256, 256, 128]
CRIT_HIDDEN  = [256, 128, 64]

torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


# ── Load Data ─────────────────────────────────────────────────────────────────

def load_ddos_only() -> torch.Tensor:
    """Chỉ lấy DDoS samples để train GAN (Generator học phân phối DDoS)"""
    X_train = np.load(f"{DATA_DIR}/X_train.npy")
    y_train = np.load(f"{DATA_DIR}/y_train.npy")
    X_ddos  = X_train[y_train == 1]
    print(f"[INFO] DDoS training samples: {len(X_ddos):,} | Features: {X_ddos.shape[1]}")
    return torch.tensor(X_ddos, dtype=torch.float32)


def load_detector(round_id: int = 1) -> MLPDetector:
    """
    Load detector đúng với round:
      round_id=1 → detector_best.pt    (Detector v1 — baseline)
      round_id=2 → detector_adv_r1.pt  (Detector v2 — sau adversarial training)
    """
    with open(f"{DETECTOR_DIR}/model_config.pkl", "rb") as f:
        cfg = pickle.load(f)
    model = MLPDetector(cfg["input_dim"], cfg["hidden_dims"], cfg["dropout"])

    det_path = (
        f"{DETECTOR_DIR}/detector_best.pt"    if round_id == 1
        else f"{DETECTOR_DIR}/detector_adv_r1.pt"
    )
    model.load_state_dict(
        torch.load(det_path, map_location=DEVICE, weights_only=True)
    )
    model.to(DEVICE)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False   # Freeze — chỉ dùng để tính loss
    print(f"[INFO] Detector loaded & frozen → {det_path}")
    return model


# ── Training ──────────────────────────────────────────────────────────────────

def train_wgan(round_id: int = 1, prev_generator_path: str = None):
    """
    Train WGAN-GP.
    round_id=0 → Phase A: Pure generation (chỉ WGAN loss, không bypass detector)
    round_id=1 → Phase B Round 1 (attack Detector v1)
    round_id=2 → Phase B Round 2 (attack Detector v2 sau adversarial training)
    prev_generator_path: nếu có, init Generator từ checkpoint trước
    """
    print(f"\n{'='*60}")
    print(f"  WGAN-GP Training — Round {round_id}")
    print(f"  Device: {DEVICE} | Epochs: {EPOCHS} | Latent: {LATENT_DIM}")
    mode = "pure" if round_id == 0 else "adversarial"
    print(f"  Mode     : {mode.upper()} | Round: {round_id}")
    print(f"{'='*60}")

    # Load data
    X_ddos    = load_ddos_only()
    input_dim = X_ddos.shape[1]

    # Detector chỉ cần ở adversarial mode
    detector = None
    if round_id > 0:
        detector = load_detector(round_id=round_id)

    # Feature config cho constraint loss
    feat_cfg = FeatureConfig()
    func_idx = torch.tensor(feat_cfg.functional_idx, dtype=torch.long, device=DEVICE)

    # Tính mean của functional features trên real DDoS — dùng làm anchor cho constraint
    X_real_np   = X_ddos.numpy()
    func_mean   = torch.tensor(
        X_real_np[:, feat_cfg.functional_idx].mean(axis=0),
        dtype=torch.float32, device=DEVICE
    )  # shape: (n_func_features,)

    loader = DataLoader(
        TensorDataset(X_ddos),
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=True,
    )

    # Init models
    G = Generator(LATENT_DIM, input_dim, GEN_HIDDEN).to(DEVICE)
    C = Critic(input_dim, CRIT_HIDDEN).to(DEVICE)

    if prev_generator_path and os.path.exists(prev_generator_path):
        G.load_state_dict(torch.load(prev_generator_path,
                                     map_location=DEVICE, weights_only=True))
        print(f"[INFO] Generator warm-started from {prev_generator_path}")
    elif round_id > 0:
        # Adversarial mode: warm-start từ pure generator nếu có
        pure_path = f"{SAVE_DIR}/generator_r0.pt"
        if os.path.exists(pure_path):
            G.load_state_dict(torch.load(pure_path, map_location=DEVICE, weights_only=True))
            print(f"[INFO] Warm-started from pure generator: {pure_path}")

    opt_G = torch.optim.Adam(G.parameters(), lr=LR_G, betas=(0.0, 0.9))
    opt_C = torch.optim.Adam(C.parameters(), lr=LR_C, betas=(0.0, 0.9))
    bce   = nn.BCEWithLogitsLoss()

    os.makedirs(SAVE_DIR, exist_ok=True)

    # Tracking
    history = {
        "critic_loss": [], "gen_loss": [], "adv_loss": [],
        "con_loss": [], "detector_fool_rate": []
    }

    print(f"\n{'Epoch':>6} | {'C_loss':>8} | {'G_loss':>8} | {'Adv':>8} | {'Con':>8} | {'FoolRate':>8}")
    print("-" * 65)

    for epoch in range(1, EPOCHS + 1):
        G.train(); C.train()
        epoch_c_loss = []
        epoch_g_loss = []
        epoch_adv    = []
        epoch_con    = []

        for (real_batch,) in loader:
            real = real_batch.to(DEVICE)
            bsz  = real.size(0)

            # ── Train Critic N_CRITIC steps ──────────────────────────────────
            for _ in range(N_CRITIC):
                z    = torch.randn(bsz, LATENT_DIM, device=DEVICE)
                fake = G(z).detach()

                c_real = C(real)
                c_fake = C(fake)
                gp     = gradient_penalty(C, real, fake, DEVICE, LAMBDA_GP)

                # Wasserstein loss + gradient penalty
                c_loss = c_fake.mean() - c_real.mean() + gp

                opt_C.zero_grad()
                c_loss.backward()
                opt_C.step()

            epoch_c_loss.append(c_loss.item())

            # ── Train Generator ───────────────────────────────────────────────
            z    = torch.randn(bsz, LATENT_DIM, device=DEVICE)
            fake = G(z)

            # WGAN Generator loss: fool Critic
            g_loss_wgan = -C(fake).mean()

            if round_id == 0:
                # Pure mode: chỉ WGAN loss + constraint loss (giữ functional features)
                adv_loss = torch.tensor(0.0, device=DEVICE)
                fake_func = fake[:, func_idx]
                con_loss  = nn.functional.mse_loss(
                    fake_func, func_mean.unsqueeze(0).expand(bsz, -1)
                )
                g_loss = g_loss_wgan + LAMBDA_CON * con_loss
            else:
                # Adversarial mode: WGAN + fool detector + constraint
                det_logits  = detector(fake)
                adv_targets = torch.zeros(bsz, device=DEVICE)
                adv_loss    = bce(det_logits, adv_targets)
                fake_func = fake[:, func_idx]
                con_loss  = nn.functional.mse_loss(
                    fake_func, func_mean.unsqueeze(0).expand(bsz, -1)
                )
                g_loss = g_loss_wgan + LAMBDA_ADV * adv_loss + LAMBDA_CON * con_loss

            opt_G.zero_grad()
            g_loss.backward()
            opt_G.step()

            epoch_g_loss.append(g_loss_wgan.item())
            epoch_adv.append(adv_loss.item())
            epoch_con.append(con_loss.item())

        # ── Evaluate fool rate (chỉ adversarial mode) ──────────────────────────
        G.eval()
        with torch.no_grad():
            z_eval    = torch.randn(2000, LATENT_DIM, device=DEVICE)
            fake_eval = G(z_eval)
            if round_id > 0 and detector is not None:
                det_logits = detector(fake_eval)
                det_preds  = (torch.sigmoid(det_logits) >= 0.5).float()
                fool_rate  = (det_preds == 0).float().mean().item()
            else:
                # Pure mode: tính validity thay thế (% samples trong valid range)
                fool_rate = 0.0   # không có ý nghĩa trong pure mode

        avg_c    = np.mean(epoch_c_loss)
        avg_g    = np.mean(epoch_g_loss)
        avg_adv  = np.mean(epoch_adv)
        avg_con  = np.mean(epoch_con)

        history["critic_loss"].append(avg_c)
        history["gen_loss"].append(avg_g)
        history["adv_loss"].append(avg_adv)
        history["con_loss"].append(avg_con)
        history["detector_fool_rate"].append(fool_rate)

        if epoch % LOG_EVERY == 0 or epoch == 1:
            print(f"{epoch:>6} | {avg_c:>8.4f} | {avg_g:>8.4f} | "
                  f"{avg_adv:>8.4f} | {avg_con:>8.4f} | {fool_rate:>8.4f}")

    print("-" * 65)
    print(f"\n✅ Final Fool Rate: {history['detector_fool_rate'][-1]:.4f}")
    print(f"   (tỉ lệ fake DDoS bị Detector nhầm là BENIGN)")

    # Save Generator & history
    gen_path = f"{SAVE_DIR}/generator_r{round_id}.pt"
    torch.save(G.state_dict(), gen_path)
    np.save(f"{SAVE_DIR}/history_r{round_id}.npy", history)
    print(f"\n   Saved Generator → {gen_path}")

    return G, history


# ── Generate Adversarial Samples (dùng cho Phase 3 & 5) ──────────────────────

def generate_adversarial_samples(round_id: int, n_samples: int = 20000) -> np.ndarray:
    """
    Load Generator đã train, sinh n_samples fake DDoS records.
    Dùng để test Detector (Phase 3) hoặc Adversarial Training (Phase 4).
    """
    with open(f"{DETECTOR_DIR}/model_config.pkl", "rb") as f:
        cfg = pickle.load(f)
    input_dim = cfg["input_dim"]

    G = Generator(LATENT_DIM, input_dim, GEN_HIDDEN).to(DEVICE)
    G.load_state_dict(
        torch.load(f"{SAVE_DIR}/generator_r{round_id}.pt",
                   map_location=DEVICE, weights_only=True)
    )
    G.eval()

    all_samples = []
    with torch.no_grad():
        for _ in range(0, n_samples, BATCH_SIZE):
            bsz = min(BATCH_SIZE, n_samples - len(all_samples) * BATCH_SIZE)
            if bsz <= 0:
                break
            z = torch.randn(BATCH_SIZE, LATENT_DIM, device=DEVICE)
            all_samples.append(G(z).cpu().numpy())

    samples = np.concatenate(all_samples, axis=0)[:n_samples]
    out_path = f"{SAVE_DIR}/fake_ddos_r{round_id}.npy"
    np.save(out_path, samples)
    print(f"[INFO] Generated {len(samples):,} fake DDoS samples → {out_path}")
    return samples


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Round 1: Train GAN để attack Detector Phase 1
    G, history = train_wgan(round_id=1)

    print("\n[INFO] Generating adversarial samples for Phase 3 evaluation ...")
    generate_adversarial_samples(round_id=1, n_samples=20000)