"""Re-create the PCA-related figures of the project (English labels).

Reproduces, with the same settings as the notebook (pixels scaled to [0, 1],
logistic regression with C=0.1, max_iter=1000):
  * PCA reconstruction error vs. number of components K
  * Leading eigen-images and sample reconstructions (K=100)
  * Logistic-regression accuracy vs. K
  * Confusion matrix of the PCA(K=100) + logistic-regression baseline

PCA is computed through the eigen-decomposition of the 4096x4096 covariance
matrix (equivalent to sklearn's PCA, but light enough for a laptop).

Usage:
    python scripts/make_pca_figures.py --data data/organamnist_64.npz --out assets
"""
import argparse
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

ORGANS = ["bladder", "femur-left", "femur-right", "heart", "kidney-left",
          "kidney-right", "liver", "lung-left", "lung-right", "pancreas", "spleen"]
LR_KS = [1, 10, 25, 50, 75, 100, 150, 200, 300, 500, 800, 1000]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/organamnist_64.npz")
    ap.add_argument("--out", default="assets")
    ap.add_argument("--results", default="results")
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(exist_ok=True, parents=True)
    Path(args.results).mkdir(exist_ok=True, parents=True)

    d = np.load(args.data)
    Xtr = d["train_images"].reshape(len(d["train_images"]), -1).astype(np.float32) / 255
    Xva = d["val_images"].reshape(len(d["val_images"]), -1).astype(np.float32) / 255
    Xte = d["test_images"].reshape(len(d["test_images"]), -1).astype(np.float32) / 255
    ytr = d["train_labels"].ravel(); yte = d["test_labels"].ravel()
    n = len(Xtr)

    # ---- PCA through the eigen-decomposition of the 4096x4096 covariance ----
    t0 = time.time()
    mean = Xtr.mean(0)
    Xtr -= mean; Xva -= mean; Xte -= mean          # centre in place (saves RAM)
    cov = np.zeros((4096, 4096))                   # accumulate in blocks to keep RAM low
    for i in range(0, n, 2000):
        blk = Xtr[i:i + 2000].astype(np.float64)
        cov += blk.T @ blk
    cov /= (n - 1)
    evals, V = np.linalg.eigh(cov)
    order = np.argsort(evals)[::-1]
    V = V[:, order].astype(np.float32); del cov, evals
    Ztr, Zva, Zte = Xtr @ V, Xva @ V, Xte @ V
    print(f"PCA done in {time.time() - t0:.0f}s", flush=True)

    # ---- reconstruction error vs K ----
    def err_curve(Xc, Z):
        total = np.square(Xc).sum(1).mean()
        return total - np.cumsum(np.square(Z).sum(0) / len(Z)), total
    Ks = np.arange(0, 4096 + 50, 50); Ks[-1] = 4096
    ctr, ttr = err_curve(Xtr, Ztr); cva, tva = err_curve(Xva, Zva)
    e_tr = np.array([ttr if k == 0 else ctr[k - 1] for k in Ks])
    e_va = np.array([tva if k == 0 else cva[k - 1] for k in Ks])
    fig, ax = plt.subplots(figsize=(8, 3.6))
    ax.plot(Ks, e_tr, "-", color="#1f77b4", label="train")
    ax.plot(Ks, e_va, "--", color="#d62728", label="validation")
    ax.set_xlabel("Number of principal components K"); ax.set_ylabel("Mean squared reconstruction error")
    ax.set_title("PCA reconstruction error (64×64 images flattened to 4096-d)")
    ax.set_ylim(bottom=0); ax.grid(alpha=.3); ax.legend()
    ax.axvline(100, color="gray", ls=":"); ax.text(110, ax.get_ylim()[1] * .55, "K = 100", color="gray")
    fig.tight_layout(); fig.savefig(out / "pca_reconstruction_error.png", dpi=150); plt.close(fig)

    # ---- eigen-images ----
    fig, axs = plt.subplots(1, 12, figsize=(14, 1.9))
    for i, ax in enumerate(axs):
        ax.imshow(V[:, i].reshape(64, 64), cmap="gray"); ax.set_title(f"PC {i + 1}", fontsize=9); ax.axis("off")
    fig.suptitle("First 12 principal components (eigen-images)", y=1.08, fontsize=11)
    fig.savefig(out / "pca_eigenimages.png", dpi=150, bbox_inches="tight"); plt.close(fig)

    # ---- reconstructions with K=100 ----
    K = 100; m = 10
    rec = Zte[:m, :K] @ V[:, :K].T
    fig, axs = plt.subplots(2, m, figsize=(14, 3.2))
    for i in range(m):
        axs[0, i].imshow((Xte[i] + mean).reshape(64, 64), cmap="gray", vmin=0, vmax=1)
        axs[1, i].imshow((rec[i] + mean).reshape(64, 64), cmap="gray", vmin=0, vmax=1)
        axs[0, i].set_title(ORGANS[yte[i]], fontsize=8)
        axs[0, i].axis("off"); axs[1, i].axis("off")
    axs[0, 0].text(-0.1, .5, "original", transform=axs[0, 0].transAxes, rotation=90, va="center", ha="right", fontsize=9)
    axs[1, 0].text(-0.1, .5, f"K={K}", transform=axs[1, 0].transAxes, rotation=90, va="center", ha="right", fontsize=9)
    fig.suptitle("Test images: original vs. PCA reconstruction with K = 100", fontsize=11)
    fig.tight_layout(); fig.savefig(out / "pca_reconstruction_samples.png", dpi=150); plt.close(fig)
    del Xtr, Xva, Xte

    # ---- confusion matrix of the baseline (PCA K=100 + LR, C=0.1) ----
    t0 = time.time()
    clf = LogisticRegression(C=0.1, max_iter=5000).fit(Ztr[:, :100], ytr)
    pred = clf.predict(Zte[:, :100]); cm = confusion_matrix(yte, pred)
    acc = cm.trace() / cm.sum()
    fig, ax = plt.subplots(figsize=(8.5, 8.5))
    ConfusionMatrixDisplay(cm, display_labels=ORGANS).plot(ax=ax, xticks_rotation=90, cmap="Blues", colorbar=False)
    ax.set_title(f"Baseline: PCA (K=100) + logistic regression — test accuracy {100 * acc:.1f}%")
    fig.tight_layout(); fig.savefig(out / "confusion_matrix_baseline.png", dpi=150); plt.close(fig)
    per_class = {o: round(100 * cm[i, i] / cm[i].sum(), 1) for i, o in enumerate(ORGANS)}
    print(f"baseline acc {100 * acc:.2f}%  ({time.time() - t0:.0f}s)", per_class, flush=True)

    # ---- LR accuracy vs K ----
    accs = {"K": [], "train": [], "test": []}
    for k in LR_KS:
        t0 = time.time()
        c = LogisticRegression(C=0.1, max_iter=1000).fit(Ztr[:, :k], ytr)
        accs["K"].append(k)
        accs["train"].append(100 * c.score(Ztr[:, :k], ytr)); accs["test"].append(100 * c.score(Zte[:, :k], yte))
        print(f"K={k:5d} train {accs['train'][-1]:.2f} test {accs['test'][-1]:.2f} ({time.time() - t0:.0f}s)", flush=True)
        json.dump({"baseline_per_class_acc": per_class, "baseline_acc": 100 * acc, "lr_vs_k": accs},
                  open(Path(args.results) / "pca_lr_study.json", "w"), indent=1)
    fig, ax = plt.subplots(figsize=(8, 3.6))
    ax.plot(accs["K"], accs["train"], "+-", color="#1f77b4", label="train")
    ax.plot(accs["K"], accs["test"], "x--", color="#d62728", label="test")
    ax.set_xscale("log"); ax.set_xlabel("Number of principal components K (log scale)"); ax.set_ylabel("Accuracy (%)")
    ax.set_title("Logistic regression (C=0.1) on PCA projections: accuracy vs. K")
    ax.grid(alpha=.3, which="both"); ax.legend()
    fig.tight_layout(); fig.savefig(out / "pca_logreg_vs_k.png", dpi=150); plt.close(fig)
    print("ALL DONE", flush=True)


if __name__ == "__main__":
    main()
