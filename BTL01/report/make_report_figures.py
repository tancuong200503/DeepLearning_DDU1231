# -*- coding: utf-8 -*-
"""Tạo lại các hình cho báo cáo Word với cỡ chữ lớn, dễ đọc khi in.

Chạy: python make_report_figures.py
Kết quả: thư mục figures/ cạnh file script.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from matplotlib import rcParams

BASE = Path(__file__).resolve().parent.parent
OUT = Path(__file__).resolve().parent / "figures"
OUT.mkdir(parents=True, exist_ok=True)
LOG = BASE / "outputs" / "logs"
MODELS = BASE / "outputs" / "models" / "runs"

rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 12,
        "axes.titlesize": 13,
        "axes.labelsize": 12,
        "xtick.labelsize": 10.5,
        "ytick.labelsize": 10.5,
        "legend.fontsize": 10.5,
        "figure.dpi": 110,
        "savefig.dpi": 220,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)

# ---------------- dữ liệu ----------------
MU = np.array([0.0, 0.0])
COV = np.array([[0.3, -0.5], [-0.5, 2.0]])
COV_INV = np.linalg.inv(COV)
NORM = 2 * np.pi * np.sqrt(np.linalg.det(COV))


def gaussian2d(X):
    d = np.asarray(X, dtype=np.float64) - MU
    q = np.einsum("ni,ij,nj->n", d, COV_INV, d)
    return np.exp(-0.5 * q) / NORM


train = pd.read_csv(BASE / "outputs" / "data" / "train_pool.csv")
val = pd.read_csv(BASE / "outputs" / "data" / "validation.csv")
test = pd.read_csv(BASE / "outputs" / "data" / "test.csv")

Xtr = train[["x1", "x2"]].to_numpy()
ytr = train["y"].to_numpy()
Xva = val[["x1", "x2"]].to_numpy()
yva = val["y"].to_numpy()
Xte = test[["x1", "x2"]].to_numpy()
yte = test["y"].to_numpy()

grid_x = np.linspace(-2, 2, 100)
grid_y = np.linspace(-4, 4, 100)
GX, GY = np.meshgrid(grid_x, grid_y)
GRID = np.column_stack([GX.ravel(), GY.ravel()])
Z_TRUE = gaussian2d(GRID).reshape(GX.shape)

df_A = pd.read_csv(LOG / "results_A.csv")
df_B = pd.read_csv(LOG / "results_B.csv")
df_C = pd.read_csv(LOG / "results_C.csv")
hist_B = pd.read_csv(LOG / "history_B.csv")
hist_C = pd.read_csv(LOG / "history_C.csv")
base_metrics = pd.read_csv(LOG / "results_baseline.csv", index_col=0)
final_df = pd.read_csv(LOG / "final_test_results.csv")
final_best = final_df.loc[final_df["val_MSE"].idxmin()]


# ---------------- mô hình ----------------
class MLP(nn.Module):
    def __init__(self, hidden_sizes, activation):
        super().__init__()
        acts = {"tanh": nn.Tanh, "relu": nn.ReLU, "sigmoid": nn.Sigmoid}
        layers, prev = [], 2
        for h in hidden_sizes:
            layers += [nn.Linear(prev, h), acts[activation]()]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def load_model(path):
    ckpt = torch.load(path, weights_only=False)
    model = MLP(ckpt["hidden_sizes"], ckpt["activation"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model


def predict(model, X):
    with torch.no_grad():
        return model(torch.as_tensor(X, dtype=torch.float32)).numpy().ravel()


baseline_model = load_model(MODELS / "baseline.pt")
final_model = load_model(BASE / "outputs" / "models" / "best_model.pt")


def save(fig, name):
    fig.tight_layout()
    fig.savefig(OUT / name, bbox_inches="tight")
    plt.close(fig)


# ---------------- 1. dữ liệu ----------------
fig = plt.figure(figsize=(10, 3.6))
ax = fig.add_subplot(1, 3, 1, projection="3d")
rng = np.random.default_rng(0)
idx = rng.choice(len(Xtr), 600, replace=False)
ax.scatter(Xtr[idx, 0], Xtr[idx, 1], ytr[idx], c=ytr[idx], cmap="viridis", s=9, alpha=0.85)
ax.set_xlabel("x1", labelpad=2)
ax.set_ylabel("x2", labelpad=2)
ax.set_zlabel("y", labelpad=2)
ax.set_title("(a) 700 mẫu train", pad=2)
ax.view_init(elev=26, azim=-60)

ax = fig.add_subplot(1, 3, 2, projection="3d")
ax.plot_surface(GX, GY, Z_TRUE, cmap="viridis", linewidth=0, antialiased=True)
ax.set_xlabel("x1", labelpad=2)
ax.set_ylabel("x2", labelpad=2)
ax.set_zlabel("y", labelpad=2)
ax.set_title("(b) Mặt hàm thật", pad=2)
ax.view_init(elev=26, azim=-60)

ax = fig.add_subplot(1, 3, 3)
cf = ax.contourf(GX, GY, Z_TRUE, levels=30, cmap="viridis")
ax.contour(GX, GY, Z_TRUE, levels=8, colors="white", linewidths=0.5, alpha=0.65)
ax.set_xlabel("x1")
ax.set_ylabel("x2")
ax.set_title("(c) Đường đồng mức", pad=2)
fig.colorbar(cf, ax=ax, shrink=0.85, pad=0.03)
save(fig, "r01_data_overview.png")

fig, axes = plt.subplots(1, 3, figsize=(9.2, 3.0))
for ax, values, title, color in [
    (axes[0], Xtr[:, 0], "x1 ~ U[-2, 2]", "#1f77b4"),
    (axes[1], Xtr[:, 1], "x2 ~ U[-4, 4]", "#ff7f0e"),
    (axes[2], ytr, "y = f(x1, x2)", "#2ca02c"),
]:
    ax.hist(values, bins=50, color=color, alpha=0.85, edgecolor="white")
    ax.set_title(title)
    ax.set_ylabel("Số mẫu")
save(fig, "r02_data_distributions.png")

# ---------------- 2. baseline ----------------
hist_base = pd.read_csv(LOG / "history_baseline.csv")
fig, ax = plt.subplots(figsize=(6.0, 3.6))
ax.plot(hist_base["epoch"], hist_base["train_loss"], label="train loss", lw=1.8)
ax.plot(hist_base["epoch"], hist_base["val_loss"], label="validation loss", lw=1.8)
ax.set_yscale("log")
ax.set_xlabel("Epoch")
ax.set_ylabel("MSE loss (thang log)")
ax.set_title("Baseline: loss theo epoch")
ax.legend()
save(fig, "r03_baseline_loss.png")


def plot_yhat(y_true, y_pred, metrics, name, title):
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    ax.scatter(y_true, y_pred, s=22, alpha=0.6, edgecolors="none")
    hi = float(max(np.max(y_true), np.max(y_pred))) * 1.05
    ax.plot([0, hi], [0, hi], "r--", lw=1.5, label="y = ŷ")
    ax.set_xlim(0, hi)
    ax.set_ylim(0, hi)
    ax.set_xlabel("y thật")
    ax.set_ylabel("ŷ dự đoán")
    ax.set_title(title, fontsize=12.5)
    ax.legend(loc="lower right")
    ax.text(
        0.04,
        0.96,
        f"MSE  = {metrics['MSE']:.3e}\nMAE  = {metrics['MAE']:.3e}\nRMSE = {metrics['RMSE']:.3e}",
        transform=ax.transAxes,
        va="top",
        fontsize=9.5,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.85, edgecolor="0.7"),
    )
    save(fig, name)


plot_yhat(
    yte,
    predict(baseline_model, Xte),
    base_metrics.loc["test"].to_dict(),
    "r04_baseline_yhat.png",
    "Baseline: y và ŷ trên test (100 mẫu)",
)


def plot_surfaces(model, name, title):
    zhat = predict(model, GRID).reshape(GX.shape)
    zmax = max(float(Z_TRUE.max()), float(zhat.max())) * 1.05
    fig = plt.figure(figsize=(10, 3.8))
    for i, (Z, name_panel) in enumerate([(Z_TRUE, "Hàm thật f(x)"), (zhat, "MLP xấp xỉ")]):
        ax = fig.add_subplot(1, 2, i + 1, projection="3d")
        ax.plot_surface(GX, GY, Z, cmap="viridis", linewidth=0, antialiased=True)
        ax.set_xlabel("x1", labelpad=2)
        ax.set_ylabel("x2", labelpad=2)
        ax.set_zlabel("y", labelpad=2)
        ax.set_zlim(0, zmax)
        ax.set_title(name_panel, pad=2)
        ax.view_init(elev=26, azim=-60)
    fig.suptitle(title, fontsize=12.5)
    save(fig, name)


plot_surfaces(baseline_model, "r05_baseline_surface.png", "Baseline: bề mặt thật và bề mặt xấp xỉ")

# ---------------- 3. thực nghiệm A ----------------
A_vals = sorted(df_A["A"].unique())
gA = df_A.groupby("A")

fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.4))
for ax, metric, label in zip(
    axes,
    ["test_MSE", "test_MAE", "val_loss_best"],
    ["Test MSE", "Test MAE", "Validation MSE"],
):
    m = gA[metric].mean()
    s = gA[metric].std().fillna(0).values
    ax.errorbar(m.index, m.values, yerr=s, marker="o", ms=5, capsize=3, lw=1.6)
    ax.set_yscale("log")
    ax.set_xlabel("Số mẫu A")
    ax.set_ylabel(label)
    ax.set_title(f"{label} theo A")
    ax.set_xticks(A_vals)
    ax.tick_params(axis="x", rotation=45)
save(fig, "r06_expA.png")

fig, ax = plt.subplots(figsize=(7.0, 3.5))
for metric, label in [
    ("train_MSE", "Train MSE"),
    ("val_MSE", "Validation MSE"),
    ("test_MSE", "Test MSE"),
]:
    m = gA[metric].mean()
    ax.plot(m.index, m.values, marker="o", ms=5, lw=1.6, label=label)
ax.set_yscale("log")
ax.set_xlabel("Số mẫu A")
ax.set_ylabel("MSE (thang log)")
ax.set_title("Train / validation / test MSE theo A")
ax.set_xticks(A_vals)
ax.tick_params(axis="x", rotation=45)
ax.legend()
save(fig, "r07_expA_gap.png")

# ---------------- 4. thực nghiệm B ----------------
B_vals = sorted(df_B["B"].unique())
gB = df_B.groupby("B")

fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.4))
ax = axes[0]
for metric, label in [("test_MSE", "Test MSE"), ("test_MAE", "Test MAE")]:
    m = gB[metric].mean()
    s = gB[metric].std().fillna(0).values
    ax.errorbar(m.index, m.values, yerr=s, marker="o", ms=5, capsize=3, lw=1.6, label=label)
ax.set_yscale("log")
ax.set_xlabel("Số nơ-ron B")
ax.set_ylabel("Sai số (thang log)")
ax.set_title("(a) Test error theo B")
ax.set_xticks(B_vals)
ax.legend()

ax = axes[1]
for metric, label in [("train_MSE", "Train MSE"), ("val_MSE", "Validation MSE")]:
    ax.plot(gB[metric].mean().index, gB[metric].mean().values, marker="o", ms=5, lw=1.6, label=label)
ax.set_yscale("log")
ax.set_xlabel("Số nơ-ron B")
ax.set_ylabel("MSE (thang log)")
ax.set_title("(b) Train vs validation")
ax.set_xticks(B_vals)
ax.legend()

ax = axes[2]
gap = gB["val_MSE"].mean() - gB["train_MSE"].mean()
ax.plot(gap.index, gap.values, marker="o", ms=5, lw=1.6, color="#d62728")
ax.axhline(0, color="0.4", lw=0.8)
ax.set_xlabel("Số nơ-ron B")
ax.set_ylabel("Val MSE − Train MSE")
ax.set_title("(c) Khoảng cách tổng quát hóa")
ax.set_xticks(B_vals)
ax.ticklabel_format(axis="y", style="sci", scilimits=(-5, -5))
save(fig, "r08_expB.png")

fig, ax = plt.subplots(figsize=(7.0, 3.5))
for b in [1, 3, 5, 10]:
    h = hist_B[(hist_B["B"] == b) & (hist_B["seed"] == 42)]
    ax.plot(h["epoch"], h["val_loss"], lw=1.6, label=f"B = {b}")
ax.set_yscale("log")
ax.set_xlabel("Epoch")
ax.set_ylabel("Validation MSE (thang log)")
ax.set_title("Hội tụ validation theo B (seed 42)")
ax.legend()
save(fig, "r09_expB_conv.png")

# ---------------- 5. thực nghiệm C ----------------
C_vals = sorted(df_C["C"].unique())
gC = df_C.groupby("C")

fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.4))
ax = axes[0]
for metric, label in [("test_MSE", "Test MSE"), ("test_MAE", "Test MAE")]:
    m = gC[metric].mean()
    s = gC[metric].std().fillna(0).values
    ax.errorbar(m.index, m.values, yerr=s, marker="o", ms=5, capsize=3, lw=1.6, label=label)
ax.set_yscale("log")
ax.set_xlabel("Số tầng ẩn C")
ax.set_ylabel("Sai số (thang log)")
ax.set_title("(a) Test error theo C")
ax.set_xticks(C_vals)
ax.legend()

ax = axes[1]
for c in C_vals:
    h = hist_C[(hist_C["C"] == c) & (hist_C["seed"] == 42)]
    ax.plot(h["epoch"], h["val_loss"], lw=1.5, label=f"C = {c}")
ax.set_yscale("log")
ax.set_xlabel("Epoch")
ax.set_ylabel("Validation MSE (thang log)")
ax.set_title("(b) Hội tụ validation")
ax.legend()

ax = axes[2]
m = gC["train_time_s"].mean()
s = gC["train_time_s"].std().fillna(0).values
ax.bar(m.index, m.values, yerr=s, capsize=4, color="#4c72b0", alpha=0.85)
ax.set_xlabel("Số tầng ẩn C")
ax.set_ylabel("Thời gian (giây)")
ax.set_title("(c) Thời gian huấn luyện")
ax.set_xticks(C_vals)
save(fig, "r10_expC.png")

fig, ax = plt.subplots(figsize=(7.0, 3.5))
for metric, label in [("train_MSE", "Train MSE"), ("val_MSE", "Validation MSE")]:
    m = gC[metric].mean()
    ax.plot(m.index, m.values, marker="o", ms=5, lw=1.6, label=label)
ax.set_yscale("log")
ax.set_xlabel("Số tầng ẩn C")
ax.set_ylabel("MSE (thang log)")
ax.set_title("Train / validation MSE theo C")
ax.set_xticks(C_vals)
ax.legend()
save(fig, "r11_expC_gap.png")

# ---------------- 6. mô hình cuối ----------------
final_metrics = {
    "MSE": float(final_best["test_MSE"]),
    "MAE": float(final_best["test_MAE"]),
    "RMSE": float(final_best["test_RMSE"]),
}
plot_yhat(
    yte,
    predict(final_model, Xte),
    final_metrics,
    "r12_final_yhat.png",
    "Mô hình cuối: y và ŷ trên test",
)
plot_surfaces(final_model, "r13_final_surface.png", "Mô hình cuối: bề mặt thật và xấp xỉ")

residual = predict(final_model, Xte) - yte
fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.4))
axes[0].hist(residual, bins=30, color="#4c72b0", edgecolor="white")
axes[0].axvline(0, color="r", ls="--", lw=1.2)
axes[0].set_xlabel("Sai số ŷ − y")
axes[0].set_ylabel("Số mẫu")
axes[0].set_title("Phân bố phần dư")

order = np.argsort(yte)
axes[1].plot(yte[order], predict(final_model, Xte)[order], ".", ms=5, label="ŷ")
axes[1].plot(yte[order], yte[order], "r--", lw=1.2, label="y thật")
axes[1].set_xlabel("y thật (sắp tăng dần)")
axes[1].set_ylabel("y")
axes[1].set_title("Dự đoán theo thứ tự y")
axes[1].legend()
save(fig, "r14_final_residuals.png")

print("Created 14 figures in", OUT)
