"""Generate the dataset and results figures used in the README (English labels).

Inputs
  * data/organamnist_64.npz            (only for the dataset figures)
  * results/model_comparison.csv       (final accuracies, taken from the notebook)
  * results/training_histories.json    (per-epoch logs, parsed from the notebook outputs)

Usage:
    python scripts/make_summary_figures.py --data data/organamnist_64.npz
"""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

ORGANS = ["bladder", "femur-left", "femur-right", "heart", "kidney-left",
          "kidney-right", "liver", "lung-left", "lung-right", "pancreas", "spleen"]
FAMILY_COLORS = {
    "Classical (PCA)": "#8c8c8c",
    "Flattened pixels": "#e69f00",
    "CNN from scratch": "#009e73",
    "Pretrained ResNet-50": "#0072b2",
}
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})


def dataset_figures(data_path, out):
    d = np.load(data_path)
    ytr = d["train_labels"].ravel(); yva = d["val_labels"].ravel(); yte = d["test_labels"].ravel()

    fig, axs = plt.subplots(1, 11, figsize=(14, 1.9))
    for c, ax in enumerate(axs):
        ax.imshow(d["train_images"][np.where(ytr == c)[0][0]], cmap="gray")
        ax.set_title(f"{c}\n{ORGANS[c]}", fontsize=8); ax.axis("off")
    fig.suptitle("OrganAMNIST (64×64): one training example per class", y=1.12, fontsize=11)
    fig.savefig(out / "class_samples.png", dpi=150, bbox_inches="tight"); plt.close(fig)

    x = np.arange(11); w = 0.27
    fig, ax = plt.subplots(figsize=(9, 3.8))
    for k, (y, name, col) in enumerate([(ytr, "train", "#0072b2"), (yva, "validation", "#e69f00"), (yte, "test", "#009e73")]):
        ax.bar(x + (k - 1) * w, 100 * np.bincount(y, minlength=11) / len(y), w, label=f"{name} (n={len(y):,})", color=col)
    ax.set_xticks(x); ax.set_xticklabels(ORGANS, rotation=40, ha="right")
    ax.set_ylabel("% of the split"); ax.set_title("Class distribution per split"); ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(out / "class_distribution.png", dpi=150); plt.close(fig)


def comparison_figure(csv_path, out):
    rows = list(csv.DictReader(open(csv_path)))
    models = [r for r in rows if r["family"] != "Literature"]
    refs = [r for r in rows if r["family"] == "Literature"]
    models.sort(key=lambda r: float(r["test_accuracy_pct"]))
    fig, ax = plt.subplots(figsize=(9, 7))
    y = np.arange(len(models))
    ax.barh(y, [float(r["test_accuracy_pct"]) for r in models],
            color=[FAMILY_COLORS[r["family"]] for r in models], zorder=2)
    for i, r in enumerate(models):
        v = float(r["test_accuracy_pct"])
        ax.text(v + 0.8, i, f"{v:.2f}", va="center", fontsize=9, zorder=4,
                fontweight="bold" if v > 94.9 else None,
                bbox=dict(facecolor="white", edgecolor="none", pad=0.6, alpha=0.9))
    ax.set_yticks(y); ax.set_yticklabels([r["model"] for r in models], fontsize=9)
    ref_handles = []
    for r, ls in zip(refs, ["--", ":"]):
        v = float(r["test_accuracy_pct"]); res = r["model"].split("(")[1][:-1]
        ax.axvline(v, color="black", ls=ls, lw=1, zorder=1)
        ref_handles.append(Line2D([0], [0], color="black", ls=ls, lw=1, label=f"MedMNIST v2 ref., {res} ({v}%)"))
    ax.set_xlim(0, 104); ax.set_ylim(-0.6, len(models) - 0.4)
    ax.set_xlabel("Test accuracy (%)"); ax.set_title("OrganAMNIST 64×64 — test accuracy of every model")
    handles = [Patch(color=c, label=f) for f, c in FAMILY_COLORS.items()] + ref_handles
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.4, -0.09), ncol=3, fontsize=8.5, frameon=False)
    fig.tight_layout(); fig.savefig(out / "model_comparison.png", dpi=150, bbox_inches="tight"); plt.close(fig)


def curves_figures(hist_path, out):
    H = json.load(open(hist_path))

    fig, axs = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True)
    for name, col in [("MLP", "#e69f00"), ("LeNet (adapted)", "#56b4e9"), ("CNN2", "#009e73")]:
        ep = [h["epoch"] for h in H[name]]
        axs[0].plot(ep, [h["train_acc"] for h in H[name]], color=col, label=name)
        axs[1].plot(ep, [h["val_acc"] for h in H[name]], color=col, label=name)
    axs[0].set_title("Training accuracy"); axs[1].set_title("Validation accuracy")
    for a in axs: a.set_xlabel("Epoch"); a.grid(alpha=.3)
    axs[0].set_ylabel("Accuracy (%)"); axs[1].legend(frameon=False)
    fig.tight_layout(); fig.savefig(out / "nn_training_curves.png", dpi=150); plt.close(fig)

    cos, pla = H["ResNet-50 + CosineAnnealingLR"], H["ResNet-50 + ReduceLROnPlateau"]
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.6))
    ep = [h["epoch"] + 1 for h in cos]
    axs[0].plot(ep, [h["val_acc"] for h in cos], "o-", color="#0072b2", label="CosineAnnealingLR")
    axs[0].plot(ep, [h["val_acc"] for h in pla], "s--", color="#d55e00", label="ReduceLROnPlateau")
    axs[0].set_title("Validation accuracy"); axs[0].set_ylabel("Accuracy (%)"); axs[0].set_ylim(74, 100)
    axs[1].plot(ep, [h["lr"] for h in cos], "o-", color="#0072b2", label="CosineAnnealingLR")
    axs[1].plot(ep, [h["lr"] for h in pla], "s--", color="#d55e00", label="ReduceLROnPlateau")
    axs[1].set_yscale("log"); axs[1].set_title("Learning rate (end of each epoch)")
    for a in axs: a.set_xlabel("Epoch"); a.grid(alpha=.3)
    axs[0].legend(frameon=False)
    fig.tight_layout(); fig.savefig(out / "scheduler_comparison.png", dpi=150); plt.close(fig)

    pl, ag = H["ResNet-50 + Cosine, no augmentation"], H["ResNet-50 + Cosine + augmentation"]
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.6))
    for h, lab, col, ls in [(pl, "no augmentation", "#0072b2", "-"), (ag, "with augmentation", "#d55e00", "--")]:
        ep = [x["epoch"] + 1 for x in h]
        axs[0].plot(ep, [x["train_acc"] for x in h], ls, color=col, label=lab)
        axs[1].plot(ep, [x["val_acc"] for x in h], ls, color=col, label=lab)
    axs[0].set_title("Training accuracy"); axs[1].set_title("Validation accuracy")
    axs[1].set_ylim(74, 100)
    for a in axs: a.set_xlabel("Epoch"); a.grid(alpha=.3)
    axs[0].set_ylabel("Accuracy (%)"); axs[0].legend(frameon=False)
    fig.tight_layout(); fig.savefig(out / "augmentation_effect.png", dpi=150); plt.close(fig)

    steps = [("Head only", 83.34), ("Full\nfine-tuning", 90.72), ("+ Cosine LR\nscheduler", 93.22), ("+ Data\naugmentation", 94.97)]
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    bars = ax.bar([s[0] for s in steps], [s[1] for s in steps], color="#0072b2", width=0.6)
    for b, s in zip(bars, steps): ax.text(b.get_x() + b.get_width() / 2, s[1] + 0.4, f"{s[1]:.2f}", ha="center", fontweight="bold")
    ax.axhline(88.83, color="#009e73", ls="--", lw=1.2, label="Best CNN trained from scratch (88.83)")
    ax.legend(loc="upper left", frameon=False, fontsize=9)
    ax.set_ylim(75, 100); ax.set_ylabel("Test accuracy (%)"); ax.set_title("Pretrained ResNet-50: from head-only to the final recipe")
    fig.tight_layout(); fig.savefig(out / "resnet_progression.png", dpi=150); plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/organamnist_64.npz")
    ap.add_argument("--out", default="assets")
    ap.add_argument("--results", default="results")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(exist_ok=True, parents=True)
    if Path(a.data).exists():
        dataset_figures(a.data, out)
    else:
        print(f"[skip] {a.data} not found: dataset figures not generated")
    comparison_figure(Path(a.results) / "model_comparison.csv", out)
    curves_figures(Path(a.results) / "training_histories.json", out)
    print("figures written to", out)
