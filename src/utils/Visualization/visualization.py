import os
import numpy as np
import matplotlib
matplotlib.use("Agg")   
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
from typing import Dict, List, Optional

from src.config import (
    BLENDSHAPE_NAMES, N_BLENDSHAPES,
    MOUTH_INDICES, EYE_INDICES,
    N_MFCC, TARGET_FPS,
)


def _maybe_display(path: str, display_inline: bool) -> None:
    if not display_inline:
        return
    try:
        from IPython.display import display, Image
        display(Image(filename=path))
    except ImportError:
        pass  


def _savefig(fig: plt.Figure, path: str, dpi: int = 150) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def _frame_axis(ax: plt.Axes, n_frames: int, fps: int = TARGET_FPS) -> None:
    ticks = np.arange(0, n_frames, fps)
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{t/fps:.1f}s" for t in ticks], fontsize=7)



def plot_loss_curves(
    history: Dict[str, List[float]],
    save_path: str,
    title: str = "Loss krive",
    display_inline: bool = False,
) -> None:
    components = [k.replace("train_", "") for k in history if k.startswith("train_")]
    n = len(components)
    if n == 0:
        return

    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4), squeeze=False)
    fig.suptitle(title, fontsize=13, fontweight="bold")
    colors = {"train": "#2563eb", "val": "#dc2626"}

    for ax, comp in zip(axes[0], components):
        for split in ("train", "val"):
            key = f"{split}_{comp}"
            if key in history and len(history[key]) > 0:
                ax.plot(history[key], label=split, color=colors[split], linewidth=1.8)
        ax.set_title(comp.upper(), fontsize=10)
        ax.set_xlabel("Epoha")
        ax.set_ylabel("Loss")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    _savefig(fig, save_path)
    _maybe_display(save_path, display_inline)


def plot_per_blendshape_mse(
    mse_per_bs: np.ndarray,
    save_path: str,
    title: str = "MSE po blendshape-u",
    display_inline: bool = False,
) -> None:
    assert len(mse_per_bs) == N_BLENDSHAPES, (
        f"Ocekivano {N_BLENDSHAPES} vrijednosti, dobijeno {len(mse_per_bs)}"
    )
    colors = []
    for i in range(N_BLENDSHAPES):
        if i in MOUTH_INDICES:
            colors.append("#ef4444")
        elif i in EYE_INDICES:
            colors.append("#3b82f6")
        else:
            colors.append("#6b7280")

    fig, ax = plt.subplots(figsize=(8, 14))
    y = np.arange(N_BLENDSHAPES)
    ax.barh(y, mse_per_bs, color=colors, edgecolor="white", height=0.7)
    ax.set_yticks(y)
    ax.set_yticklabels(BLENDSHAPE_NAMES, fontsize=7)
    ax.set_xlabel("MSE")
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.grid(True, axis="x", alpha=0.3)
    ax.invert_yaxis()
    legend_elements = [
        Patch(facecolor="#ef4444", label="Usta / Vilica / Jezik"),
        Patch(facecolor="#3b82f6", label="Oci / Obrve"),
        Patch(facecolor="#6b7280", label="Ostalo"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=8)
    fig.tight_layout()
    _savefig(fig, save_path)
    _maybe_display(save_path, display_inline)


def plot_prediction_vs_target(
    pred: np.ndarray,
    target: np.ndarray,
    save_path: str,
    indices: Optional[List[int]] = None,
    sample_name: str = "",
    display_inline: bool = False,
) -> None:
    if indices is None:
        indices = MOUTH_INDICES[:12]

    n = len(indices)
    ncols = 3
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 5, nrows * 2.5))
    fig.suptitle(
        f"Predikcija vs. Target  —  {sample_name}" if sample_name else "Predikcija vs. Target",
        fontsize=12, fontweight="bold",
    )
    axes_flat = np.array(axes).flatten()
    T = pred.shape[0]

    for ax_i, bs_i in enumerate(indices):
        ax = axes_flat[ax_i]
        ax.plot(target[:, bs_i], label="Target", color="#2563eb", linewidth=1.4, alpha=0.85)
        ax.plot(pred[:, bs_i],   label="Pred",   color="#ef4444", linewidth=1.2,
                alpha=0.85, linestyle="--")
        ax.set_title(BLENDSHAPE_NAMES[bs_i], fontsize=8)
        ax.set_ylim(-0.05, 1.05)
        _frame_axis(ax, T)
        ax.grid(True, alpha=0.25)
        if ax_i == 0:
            ax.legend(fontsize=7)

    for ax in axes_flat[n:]:
        ax.set_visible(False)

    fig.tight_layout()
    _savefig(fig, save_path)
    _maybe_display(save_path, display_inline)


def plot_blendshape_heatmap(
    pred: np.ndarray,
    target: np.ndarray,
    save_path: str,
    sample_name: str = "",
    display_inline: bool = False,
) -> None:
    error  = np.abs(pred - target)
    data   = [target.T, pred.T, error.T]
    titles = ["Target", "Predikcija", "Apsolutna greska"]
    cmaps  = ["viridis", "viridis", "hot"]

    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    fig.suptitle(
        f"Blendshape heatmap  —  {sample_name}" if sample_name else "Blendshape heatmap",
        fontsize=12, fontweight="bold",
    )
    for ax, d, title, cmap in zip(axes, data, titles, cmaps):
        vmin, vmax = (0, 1) if title != "Apsolutna greska" else (0, 0.3)
        im = ax.imshow(d, aspect="auto", origin="lower",
                       cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
        ax.set_title(title, fontsize=9)
        ax.set_ylabel("Blendshape idx")
        plt.colorbar(im, ax=ax, pad=0.01)

    T = pred.shape[0]
    axes[-1].set_xlabel("Vreme")
    _frame_axis(axes[-1], T)
    fig.tight_layout()
    _savefig(fig, save_path)
    _maybe_display(save_path, display_inline)


def plot_mfcc(
    mfcc_feats: np.ndarray,
    save_path: str,
    sample_name: str = "",
    display_inline: bool = False,
) -> None:
    feats = mfcc_feats[:, :N_MFCC].T

    fig, ax = plt.subplots(figsize=(12, 4))
    im = ax.imshow(feats, aspect="auto", origin="lower",
                   cmap="magma", interpolation="nearest")
    ax.set_title(
        f"MFCC  —  {sample_name}" if sample_name else "MFCC spektrogram",
        fontsize=11, fontweight="bold",
    )
    ax.set_ylabel("MFCC koeficijent")
    ax.set_xlabel("Vreme")
    _frame_axis(ax, mfcc_feats.shape[0])
    plt.colorbar(im, ax=ax, pad=0.01)
    fig.tight_layout()
    _savefig(fig, save_path)
    _maybe_display(save_path, display_inline)


def plot_phoneme_alignment(
    phoneme_ids: np.ndarray,
    phoneme_trel: np.ndarray,
    save_path: str,
    idx_to_phoneme: Optional[Dict[int, str]] = None,
    sample_name: str = "",
    display_inline: bool = False,
) -> None:
    from src.config import PHONEME_VOCAB
    if idx_to_phoneme is None:
        idx_to_phoneme = {i: p for i, p in enumerate(PHONEME_VOCAB)}

    T = len(phoneme_ids)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 4), sharex=True)
    fig.suptitle(
        f"Fonemsko poravnanje  —  {sample_name}" if sample_name else "Fonemsko poravnanje",
        fontsize=11, fontweight="bold",
    )
    ax1.step(range(T), phoneme_ids, where="mid", color="#7c3aed", linewidth=1.2)
    ax1.set_ylabel("Fonem ID")
    ax1.grid(True, alpha=0.25)
    ax2.plot(phoneme_trel, color="#059669", linewidth=1.0)
    ax2.set_ylabel("t_rel [0,1]")
    ax2.set_ylim(-0.05, 1.05)
    ax2.grid(True, alpha=0.25)
    _frame_axis(ax2, T)
    fig.tight_layout()
    _savefig(fig, save_path)
    _maybe_display(save_path, display_inline)


def plot_scatter_pred_vs_target(
    pred: np.ndarray,
    target: np.ndarray,
    save_path: str,
    indices: Optional[List[int]] = None,
    sample_name: str = "",
    display_inline: bool = False,
) -> None:
    if indices is None:
        indices = MOUTH_INDICES[:9]

    n = len(indices)
    ncols = 3
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3.5, nrows * 3.5))
    fig.suptitle(
        f"Scatter Pred vs. Target  —  {sample_name}" if sample_name else "Scatter Pred vs. Target",
        fontsize=11, fontweight="bold",
    )
    axes_flat = np.array(axes).flatten()

    for ax_i, bs_i in enumerate(indices):
        ax = axes_flat[ax_i]
        ax.scatter(target[:, bs_i], pred[:, bs_i],
                   s=4, alpha=0.3, color="#2563eb", rasterized=True)
        ax.plot([0, 1], [0, 1], "r--", linewidth=1, label="y=x")
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        ax.set_title(BLENDSHAPE_NAMES[bs_i], fontsize=8)
        ax.set_xlabel("Target", fontsize=7)
        ax.set_ylabel("Pred",   fontsize=7)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.25)

    for ax in axes_flat[n:]:
        ax.set_visible(False)

    fig.tight_layout()
    _savefig(fig, save_path)
    _maybe_display(save_path, display_inline)


def plot_velocity_profile(
    pred: np.ndarray,
    target: np.ndarray,
    save_path: str,
    indices: Optional[List[int]] = None,
    sample_name: str = "",
    display_inline: bool = False,
) -> None:

    if indices is None:
        indices = MOUTH_INDICES[:6]

    pred_vel   = np.diff(pred,   axis=0)
    target_vel = np.diff(target, axis=0)
    pred_acc   = np.diff(pred_vel,   axis=0)
    target_acc = np.diff(target_vel, axis=0)

    n = len(indices)
    fig, axes = plt.subplots(n, 2, figsize=(12, n * 2.5))
    fig.suptitle(
        f"Velocity & Acceleration  —  {sample_name}" if sample_name else "Velocity & Acceleration",
        fontsize=11, fontweight="bold",
    )
    if n == 1:
        axes = [axes]

    for row, bs_i in enumerate(indices):
        ax_v, ax_a = axes[row]
        ax_v.plot(target_vel[:, bs_i], label="Target", color="#2563eb", linewidth=1.2)
        ax_v.plot(pred_vel[:, bs_i],   label="Pred",   color="#ef4444",
                  linewidth=1.0, linestyle="--")
        ax_v.set_title(f"{BLENDSHAPE_NAMES[bs_i]} — Velocity", fontsize=8)
        ax_v.grid(True, alpha=0.25)
        if row == 0:
            ax_v.legend(fontsize=7)

        ax_a.plot(target_acc[:, bs_i], label="Target", color="#2563eb", linewidth=1.2)
        ax_a.plot(pred_acc[:, bs_i],   label="Pred",   color="#ef4444",
                  linewidth=1.0, linestyle="--")
        ax_a.set_title(f"{BLENDSHAPE_NAMES[bs_i]} — Acceleration", fontsize=8)
        ax_a.grid(True, alpha=0.25)

    fig.tight_layout()
    _savefig(fig, save_path)
    _maybe_display(save_path, display_inline)


def plot_error_correlation(
    pred: np.ndarray,
    target: np.ndarray,
    save_path: str,
    title: str = "Korelaciona matrica greske",
    display_inline: bool = False,
) -> None:
    error = pred - target
    corr  = np.corrcoef(error.T)

    fig, ax = plt.subplots(figsize=(13, 11))
    im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1, aspect="auto")
    plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02)

    tick_labels = [n[:12] for n in BLENDSHAPE_NAMES]
    ax.set_xticks(range(N_BLENDSHAPES))
    ax.set_xticklabels(tick_labels, rotation=90, fontsize=5)
    ax.set_yticks(range(N_BLENDSHAPES))
    ax.set_yticklabels(tick_labels, fontsize=5)
    ax.set_title(title, fontsize=11, fontweight="bold")
    fig.tight_layout()
    _savefig(fig, save_path)
    _maybe_display(save_path, display_inline)


def plot_sample_report(
    pred: np.ndarray,
    target: np.ndarray,
    mfcc_feats: np.ndarray,
    phoneme_ids: np.ndarray,
    phoneme_trel: np.ndarray,
    save_path: str,
    sample_name: str = "",
    display_inline: bool = False,
) -> None:
    T   = pred.shape[0]
    fig = plt.figure(figsize=(16, 14))
    gs  = gridspec.GridSpec(4, 1, hspace=0.45)
    fig.suptitle(
        f"Uzorak: {sample_name}" if sample_name else "Izvjestaj uzorka",
        fontsize=13, fontweight="bold",
    )

    # MFCC
    ax0 = fig.add_subplot(gs[0])
    im0 = ax0.imshow(mfcc_feats[:, :N_MFCC].T, aspect="auto",
                     origin="lower", cmap="magma")
    ax0.set_title("MFCC", fontsize=9)
    ax0.set_ylabel("Koef.")
    plt.colorbar(im0, ax=ax0, pad=0.01)
    _frame_axis(ax0, T)

    # Fonemsko poravnanje
    ax1 = fig.add_subplot(gs[1])
    ax1.step(range(T), phoneme_ids, where="mid", color="#7c3aed", linewidth=1.0)
    ax1.set_title("Fonemsko poravnanje", fontsize=9)
    ax1.set_ylabel("Fonem ID")
    ax1.grid(True, alpha=0.25)
    _frame_axis(ax1, T)

    # Predikcija vs. target (kljucni blendshape-ovi)
    ax2 = fig.add_subplot(gs[2])
    key_bs   = [BLENDSHAPE_NAMES.index("jawOpen")] + MOUTH_INDICES[:3]
    clr_list = ["#ef4444", "#3b82f6", "#10b981", "#f59e0b"]
    for i, bs_i in enumerate(key_bs):
        c   = clr_list[i % len(clr_list)]
        lbl = BLENDSHAPE_NAMES[bs_i][:10]
        ax2.plot(target[:, bs_i], color=c, linewidth=1.4, alpha=0.8,  label=f"T:{lbl}")
        ax2.plot(pred[:, bs_i],   color=c, linewidth=1.0, alpha=0.7,
                 linestyle="--", label=f"P:{lbl}")
    ax2.set_ylim(-0.05, 1.05)
    ax2.set_title("Predikcija vs. Target (usta/vilica)", fontsize=9)
    ax2.legend(fontsize=6, ncol=4, loc="upper right")
    ax2.grid(True, alpha=0.25)
    _frame_axis(ax2, T)

    # Heatmap greske
    ax3 = fig.add_subplot(gs[3])
    im3 = ax3.imshow(np.abs(pred - target).T, aspect="auto",
                     origin="lower", cmap="hot", vmin=0, vmax=0.3)
    ax3.set_title("Apsolutna greska (svi blendshape-ovi)", fontsize=9)
    ax3.set_ylabel("BS idx")
    plt.colorbar(im3, ax=ax3, pad=0.01)
    _frame_axis(ax3, T)

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    fig.savefig(save_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    _maybe_display(save_path, display_inline)