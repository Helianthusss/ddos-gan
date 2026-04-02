"""
Sinh biểu đồ cho slide từ data thực tế của experiment.
Chạy từ thư mục gốc: python slides/gen_figures.py
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pickle, json, os

os.makedirs("slides/figures", exist_ok=True)

# ── Màu sắc nhất quán ─────────────────────────────────────────────────────
C_REAL  = "#1565C0"   # blue
C_FAKE0 = "#6A1B9A"   # purple (pure GAN)
C_FAKE1 = "#C62828"   # red    (adversarial R1)
C_FAKE2 = "#E65100"   # orange (adversarial R2)
ALPHA   = 0.55

plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 11,
    'figure.dpi': 150,
})

# ── Load data ──────────────────────────────────────────────────────────────
X_test  = np.load("data/processed/X_test.npy")
y_test  = np.load("data/processed/y_test.npy")
X_train = np.load("data/processed/X_train.npy")
y_train = np.load("data/processed/y_train.npy")

real_ddos = X_test[y_test == 1]
fake_r0   = np.load("gan/fake_ddos_r0.npy")
fake_r1   = np.load("gan/fake_ddos_r1.npy")
fake_r2   = np.load("gan/fake_ddos_r2.npy")

with open("data/processed/feature_names.pkl", "rb") as f:
    feature_names = pickle.load(f)

# feature_names có thể là danh sách phẳng hoặc lồng
if isinstance(feature_names[0], (list, tuple)):
    feature_names = [item for sublist in feature_names for item in sublist]


# ══════════════════════════════════════════════════════════════════════════
# Fig 1: Phân phối so sánh Real vs Fake (4 features)
# ══════════════════════════════════════════════════════════════════════════
# Chọn 4 features thú vị: 2 có KL thấp (giống real) + 2 có validity cao
sel_names = ["Flow IAT Std", "Bwd Packets/s", "ACK Flag Count", "Flow Duration"]
sel_idx   = [feature_names.index(n) for n in sel_names if n in feature_names]
if len(sel_idx) < 4:
    sel_idx = [0, 5, 10, 20]   # fallback
    sel_names = [feature_names[i] for i in sel_idx]

fig, axes = plt.subplots(2, 2, figsize=(10, 6))
fig.suptitle("So sánh phân phối: Real DDoS vs GAN-generated", fontweight='bold', fontsize=14)

for ax, idx, name in zip(axes.flatten(), sel_idx, sel_names):
    bins = 40
    rng  = (
        min(real_ddos[:,idx].min(), fake_r0[:,idx].min(), fake_r1[:,idx].min()),
        max(real_ddos[:,idx].max(), fake_r0[:,idx].max(), fake_r1[:,idx].max()) + 1e-8
    )
    ax.hist(real_ddos[:5000, idx], bins=bins, range=rng, alpha=ALPHA, color=C_REAL,
            label="Real DDoS", density=True)
    ax.hist(fake_r0[:5000, idx],   bins=bins, range=rng, alpha=ALPHA, color=C_FAKE0,
            label="GAN Round 0 (Pure)", density=True)
    ax.hist(fake_r1[:5000, idx],   bins=bins, range=rng, alpha=ALPHA, color=C_FAKE1,
            label="GAN Round 1 (Adv)", density=True)
    ax.set_title(name, fontsize=11)
    ax.set_xlabel("Normalized value")
    ax.set_ylabel("Density")
    ax.tick_params(labelsize=9)

handles = [
    mpatches.Patch(color=C_REAL,  alpha=0.8, label="Real DDoS"),
    mpatches.Patch(color=C_FAKE0, alpha=0.8, label="GAN Round 0 (Pure)"),
    mpatches.Patch(color=C_FAKE1, alpha=0.8, label="GAN Round 1 (Adv)"),
]
fig.legend(handles=handles, loc='lower center', ncol=3, fontsize=10,
           bbox_to_anchor=(0.5, -0.02))
plt.tight_layout(rect=[0, 0.05, 1, 1])
plt.savefig("slides/figures/fig_distribution.pdf", bbox_inches='tight')
plt.savefig("slides/figures/fig_distribution.png", bbox_inches='tight')
plt.close()
print("Saved: fig_distribution")


# ══════════════════════════════════════════════════════════════════════════
# Fig 2: Per-feature Validity so sánh Round 0 vs Round 1 vs Round 2
# ══════════════════════════════════════════════════════════════════════════
with open("evaluate/validity_r0.json") as f: v0 = json.load(f)
with open("evaluate/validity_r1.json") as f: v1 = json.load(f)
with open("evaluate/validity_r2.json") as f: v2 = json.load(f)

feat_v  = list(v0["per_feature"].keys())
vals_r0 = [v0["per_feature"][k] for k in feat_v]
vals_r1 = [v1["per_feature"][k] for k in feat_v]
vals_r2 = [v2["per_feature"][k] for k in feat_v]

x     = np.arange(len(feat_v))
width = 0.28

fig, ax = plt.subplots(figsize=(12, 5))
bars0 = ax.bar(x - width, vals_r0, width, label="Round 0 (Pure)",  color=C_FAKE0, alpha=0.85)
bars1 = ax.bar(x,          vals_r1, width, label="Round 1 (Adv R1)", color=C_FAKE1, alpha=0.85)
bars2 = ax.bar(x + width,  vals_r2, width, label="Round 2 (Adv R2)", color=C_FAKE2, alpha=0.85)

ax.set_xticks(x)
ax.set_xticklabels([n.replace(" ", "\n") for n in feat_v], fontsize=8, ha='center')
ax.set_ylabel("Validity (0–1)")
ax.set_title("Per-feature Validity Score — Functional Features", fontweight='bold')
ax.set_ylim(0, 1.05)
ax.legend(fontsize=10)
ax.axhline(1.0, color='gray', linestyle='--', linewidth=0.8)
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig("slides/figures/fig_validity.pdf", bbox_inches='tight')
plt.savefig("slides/figures/fig_validity.png", bbox_inches='tight')
plt.close()
print("Saved: fig_validity")


# ══════════════════════════════════════════════════════════════════════════
# Fig 3: Arms Race — FoolRate qua các round
# ══════════════════════════════════════════════════════════════════════════
rounds      = ["Round 1\n(vs MLP v1)", "Round 2\n(vs MLP v2)"]
foolrate_mlp = [1.0000, 1.0000]
foolrate_rf  = [0.6371, 0.8716]

fig, ax = plt.subplots(figsize=(7, 4.5))
x = np.arange(len(rounds))
w = 0.35
ax.bar(x - w/2, foolrate_mlp, w, label="MLP (target)",   color=C_REAL,  alpha=0.85)
ax.bar(x + w/2, foolrate_rf,  w, label="RF (unseen)",     color="#2E7D32", alpha=0.85)

ax.set_xticks(x)
ax.set_xticklabels(rounds, fontsize=11)
ax.set_ylabel("FoolRate")
ax.set_ylim(0, 1.15)
ax.set_title("Arms Race: FoolRate theo Round", fontweight='bold')
ax.axhline(1.0, color='gray', linestyle='--', linewidth=0.8)
ax.legend(fontsize=10)
for bar in ax.patches:
    h = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, h + 0.02,
            f"{h:.0%}", ha='center', va='bottom', fontsize=10, fontweight='bold')
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig("slides/figures/fig_armsrace.pdf", bbox_inches='tight')
plt.savefig("slides/figures/fig_armsrace.png", bbox_inches='tight')
plt.close()
print("Saved: fig_armsrace")


# ══════════════════════════════════════════════════════════════════════════
# Fig 4: SHAP — Functional vs Non-functional importance
# ══════════════════════════════════════════════════════════════════════════
groups   = ["Functional\nFeatures", "Non-functional\nFeatures"]
shap_v1  = [0.479418, 0.465992]
shap_v2  = [0.445783, 0.399818]

fig, ax = plt.subplots(figsize=(6.5, 4.5))
x = np.arange(len(groups))
w = 0.35
b1 = ax.bar(x - w/2, shap_v1, w, label="Detector v1 (Baseline)", color=C_REAL,  alpha=0.85)
b2 = ax.bar(x + w/2, shap_v2, w, label="Detector v2 (Adv. Trained)", color=C_FAKE1, alpha=0.85)

ax.set_xticks(x)
ax.set_xticklabels(groups, fontsize=12)
ax.set_ylabel("Mean |SHAP value|")
ax.set_title("SHAP Feature Importance: Detector v1 vs v2", fontweight='bold')
ax.set_ylim(0, 0.58)
ax.legend(fontsize=10)

for bar in list(b1) + list(b2):
    h = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, h + 0.006,
            f"{h:.3f}", ha='center', va='bottom', fontsize=10)

# Annotate change arrows
for i, (v_old, v_new) in enumerate(zip(shap_v1, shap_v2)):
    diff = v_new - v_old
    ax.annotate(f"Δ={diff:+.3f}",
                xy=(i, max(v_old, v_new) + 0.035),
                ha='center', fontsize=10, color='gray', style='italic')

ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig("slides/figures/fig_shap.pdf", bbox_inches='tight')
plt.savefig("slides/figures/fig_shap.png", bbox_inches='tight')
plt.close()
print("Saved: fig_shap")


# ══════════════════════════════════════════════════════════════════════════
# Fig 5: KL Divergence per round (scatter: real vs fake similarity)
# ══════════════════════════════════════════════════════════════════════════
import pandas as pd
q0 = pd.read_csv("evaluate/gan_quality_r0.csv")
q1 = pd.read_csv("evaluate/gan_quality_r1.csv")
q2 = pd.read_csv("evaluate/gan_quality_r2.csv")

fig, axes = plt.subplots(1, 3, figsize=(13, 4.5), sharey=True)
titles = ["Round 0 (Pure GAN)", "Round 1 (Adversarial)", "Round 2 (Adversarial)"]
dfs    = [q0, q1, q2]
colors = [C_FAKE0, C_FAKE1, C_FAKE2]

for ax, df, title, color in zip(axes, dfs, titles, colors):
    df_sorted = df.sort_values("kl_div")
    top10 = df_sorted.head(10)
    ax.barh(range(len(top10)), top10["kl_div"], color=color, alpha=0.8)
    ax.set_yticks(range(len(top10)))
    ax.set_yticklabels(
        [n[:20] for n in top10["feature"].tolist()],
        fontsize=8
    )
    ax.set_xlabel("KL Divergence (thấp = giống real)")
    ax.set_title(f"{title}\nMean KL={df['kl_div'].mean():.3f}", fontweight='bold', fontsize=11)
    ax.grid(axis='x', alpha=0.3)

fig.suptitle("Top 10 Features Giống Real DDoS Nhất (KL thấp nhất)", fontweight='bold', fontsize=13)
plt.tight_layout()
plt.savefig("slides/figures/fig_kl.pdf", bbox_inches='tight')
plt.savefig("slides/figures/fig_kl.png", bbox_inches='tight')
plt.close()
print("Saved: fig_kl")


# ══════════════════════════════════════════════════════════════════════════
# Fig 6: F1 & FNR/FoolRate timeline — Arms Race evolution (line chart)
# ══════════════════════════════════════════════════════════════════════════
phases_short = ["P1\nBaseline\n(MLP v1)", "P3\nGAN R1\nattack", "P4\nAdv.Train\n(MLP v2)", "P5\nGAN R2\nattack"]
f1_vals      = [0.9996, 0.0000, 0.9998, 0.0000]
fnr_vals     = [0.0003, 1.0000, 0.0000, 1.0000]

fig, ax1 = plt.subplots(figsize=(8, 4.5))
x_pos = np.arange(len(phases_short))

color_f1  = "#1565C0"
color_fnr = "#C62828"

ax1.plot(x_pos, f1_vals, "o-", color=color_f1, linewidth=2.5, markersize=9,
         label="F1 Score (Detector)", zorder=3)
ax1.plot(x_pos, fnr_vals, "s--", color=color_fnr, linewidth=2.5, markersize=9,
         label="FNR / FoolRate (GAN)", zorder=3)

for xi, (f, r) in enumerate(zip(f1_vals, fnr_vals)):
    ax1.annotate(f"{f:.4f}", (xi, f), textcoords="offset points",
                 xytext=(0, 10), ha='center', fontsize=9, color=color_f1, fontweight='bold')
    ax1.annotate(f"{r:.4f}", (xi, r), textcoords="offset points",
                 xytext=(0, -16), ha='center', fontsize=9, color=color_fnr, fontweight='bold')

ax1.set_xticks(x_pos)
ax1.set_xticklabels(phases_short, fontsize=10)
ax1.set_ylabel("Score")
ax1.set_ylim(-0.15, 1.25)
ax1.set_title("Arms Race Evolution: F1 vs FoolRate theo Phase", fontweight='bold', fontsize=13)
ax1.legend(fontsize=10, loc='center')
ax1.axhspan(0, 0.5, alpha=0.04, color=color_fnr)
ax1.axhspan(0.5, 1.25, alpha=0.04, color=color_f1)
ax1.grid(axis='y', alpha=0.3)

ax1.annotate("", xy=(1, 0.05), xytext=(0, 0.95),
             arrowprops=dict(arrowstyle="->", color=color_fnr, lw=1.5))
ax1.text(0.5, 0.50, "GAN bypasses", ha='center', fontsize=8, color=color_fnr, style='italic')
ax1.annotate("", xy=(2, 0.95), xytext=(1, 0.05),
             arrowprops=dict(arrowstyle="->", color=color_f1, lw=1.5))
ax1.text(1.5, 0.54, "Detector adapts", ha='center', fontsize=8, color=color_f1, style='italic')
ax1.annotate("", xy=(3, 0.05), xytext=(2, 0.95),
             arrowprops=dict(arrowstyle="->", color=color_fnr, lw=1.5))
ax1.text(2.5, 0.50, "GAN bypasses again", ha='center', fontsize=8, color=color_fnr, style='italic')

plt.tight_layout()
plt.savefig("slides/figures/fig_timeline.pdf", bbox_inches='tight')
plt.savefig("slides/figures/fig_timeline.png", bbox_inches='tight')
plt.close()
print("Saved: fig_timeline")


# ══════════════════════════════════════════════════════════════════════════
# Fig 7: Validity Heatmap — per feature per round
# ══════════════════════════════════════════════════════════════════════════
feat_labels = list(v0["per_feature"].keys())
data_matrix = np.array([
    [v0["per_feature"][k] for k in feat_labels],
    [v1["per_feature"][k] for k in feat_labels],
    [v2["per_feature"][k] for k in feat_labels],
])

import matplotlib.colors as mcolors
fig, ax = plt.subplots(figsize=(11, 3.2))
cmap = mcolors.LinearSegmentedColormap.from_list(
    "validity", ["#C62828", "#FFF9C4", "#2E7D32"]
)
im = ax.imshow(data_matrix, aspect='auto', cmap=cmap, vmin=0, vmax=1)

ax.set_yticks([0, 1, 2])
ax.set_yticklabels(["Round 0\n(Pure)", "Round 1\n(Adv R1)", "Round 2\n(Adv R2)"], fontsize=10)
ax.set_xticks(range(len(feat_labels)))
ax.set_xticklabels([f.replace(" ", "\n") for f in feat_labels], fontsize=8, ha='center')
ax.set_title("Validity Heatmap: Per-feature x Round  (xanh=valid, do=invalid)", fontweight='bold')

for i in range(3):
    for j in range(len(feat_labels)):
        val = data_matrix[i, j]
        color = "white" if val < 0.3 or val > 0.85 else "black"
        ax.text(j, i, f"{val:.2f}", ha='center', va='center', fontsize=8,
                color=color, fontweight='bold')

cbar = fig.colorbar(im, ax=ax, orientation='vertical', fraction=0.02, pad=0.01)
cbar.set_label("Validity", fontsize=9)
plt.tight_layout()
plt.savefig("slides/figures/fig_validity_heatmap.pdf", bbox_inches='tight')
plt.savefig("slides/figures/fig_validity_heatmap.png", bbox_inches='tight')
plt.close()
print("Saved: fig_validity_heatmap")


print("\nAll figures saved to slides/figures/")
