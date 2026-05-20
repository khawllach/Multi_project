"""
Part 5 — Evaluation & Visualisation
5a — Quality Metrics: PSNR, compression ratio, frame-type breakdown
5b — Pipeline Visualisation: full matplotlib figure showing every stage
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")   # headless backend — safe for all environments
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from scipy.fftpack import dct
import cv2
import os


# ══════════════════════════════════════════════════════════════════════════════
# 5a — Quality Metrics
# ══════════════════════════════════════════════════════════════════════════════

def compute_psnr(original: np.ndarray, reconstructed: np.ndarray,
                 max_val: float = 255.0) -> float:
    """Compute Peak Signal-to-Noise Ratio (dB) between two uint8/float images."""
    mse = np.mean((original.astype(np.float64) - reconstructed.astype(np.float64)) ** 2)
    if mse == 0:
        return float("inf")
    return 10 * np.log10(max_val ** 2 / mse)


def compute_metrics(original_frames_bgr: list[np.ndarray],
                    reconstructed_frames_bgr: list[np.ndarray],
                    compressed_size_bytes: int,
                    frame_types: list[str]) -> dict:
    """
    Compute per-frame and global quality metrics.

    Parameters
    ----------
    original_frames_bgr      : list of original BGR frames
    reconstructed_frames_bgr : list of reconstructed BGR frames
    compressed_size_bytes    : size of the .bin file in bytes
    frame_types              : list of 'I' or 'P' per frame

    Returns
    -------
    dict with:
      - per_frame_psnr        : list[float]
      - mean_psnr             : float
      - original_size_bytes   : int
      - compressed_size_bytes : int
      - compression_ratio     : float
      - num_iframes, num_pframes
    """
    # Raw (uncompressed) size = H * W * 3 bytes per frame
    original_size_bytes = sum(f.shape[0] * f.shape[1] * 3 for f in original_frames_bgr)

    per_frame_psnr = [
        compute_psnr(orig, rec)
        for orig, rec in zip(original_frames_bgr, reconstructed_frames_bgr)
    ]

    compression_ratio = original_size_bytes / compressed_size_bytes \
        if compressed_size_bytes > 0 else float("inf")

    num_iframes = frame_types.count("I")
    num_pframes = frame_types.count("P")

    return {
        "per_frame_psnr": per_frame_psnr,
        "mean_psnr": float(np.mean(per_frame_psnr)),
        "original_size_bytes": original_size_bytes,
        "compressed_size_bytes": compressed_size_bytes,
        "compression_ratio": compression_ratio,
        "num_iframes": num_iframes,
        "num_pframes": num_pframes,
        "frame_types": frame_types,
    }


def print_metrics(metrics: dict) -> None:
    """Pretty-print quality metrics to stdout."""
    print("=" * 55)
    print("  EVALUATION METRICS")
    print("=" * 55)
    print(f"  Frames          : {len(metrics['frame_types'])}  "
          f"(I={metrics['num_iframes']}, P={metrics['num_pframes']})")
    print(f"  Original size   : {metrics['original_size_bytes'] / 1024:.1f} KB")
    print(f"  Compressed size : {metrics['compressed_size_bytes'] / 1024:.1f} KB")
    print(f"  Compression ratio: {metrics['compression_ratio']:.2f}×")
    print(f"  Mean PSNR       : {metrics['mean_psnr']:.2f} dB")
    print("-" * 55)
    for i, (psnr, ftype) in enumerate(zip(metrics["per_frame_psnr"],
                                          metrics["frame_types"])):
        psnr_str = f"{psnr:.2f} dB" if psnr != float("inf") else "∞  (perfect)"
        print(f"  Frame {i+1:3d} [{ftype}] : {psnr_str}")
    print("=" * 55)


# ══════════════════════════════════════════════════════════════════════════════
# 5b — Pipeline Visualisation helpers
# ══════════════════════════════════════════════════════════════════════════════

def _dct2(block: np.ndarray) -> np.ndarray:
    return dct(dct(block.T, norm="ortho").T, norm="ortho")


def _show_frame(ax, img_bgr: np.ndarray, title: str) -> None:
    """Display a BGR image on an axes (converted to RGB)."""
    ax.imshow(cv2.cvtColor(img_bgr.astype(np.uint8), cv2.COLOR_BGR2RGB))
    ax.set_title(title, fontsize=8, fontweight="bold")
    ax.axis("off")


def _show_channel(ax, channel: np.ndarray, title: str, cmap: str = "gray") -> None:
    ax.imshow(channel, cmap=cmap, vmin=0, vmax=255)
    ax.set_title(title, fontsize=8, fontweight="bold")
    ax.axis("off")


def _show_block(ax, data: np.ndarray, title: str, cmap: str = "viridis",
                annotate: bool = True) -> None:
    """Display an 8×8 block with optional value annotations."""
    ax.imshow(data, cmap=cmap, aspect="auto")
    ax.set_title(title, fontsize=7, fontweight="bold")
    ax.set_xticks([]); ax.set_yticks([])
    if annotate:
        for r in range(data.shape[0]):
            for c in range(data.shape[1]):
                val = data[r, c]
                ax.text(c, r, f"{val:.0f}", ha="center", va="center",
                        fontsize=4, color="white" if abs(val) > data.max() * 0.5 else "black")


def _overlay_motion_vectors(ax, frame_bgr: np.ndarray,
                              motion_vectors: np.ndarray,
                              mb_size: int = 16) -> None:
    """Overlay motion vector arrows on a frame image."""
    ax.imshow(cv2.cvtColor(frame_bgr.astype(np.uint8), cv2.COLOR_BGR2RGB))
    ax.set_title("Motion Vectors (P-frame)", fontsize=8, fontweight="bold")
    ax.axis("off")

    mbh, mbw = motion_vectors.shape[:2]
    for i in range(mbh):
        for j in range(mbw):
            dy, dx = motion_vectors[i, j]
            # Arrow origin = centre of macroblock
            cx = j * mb_size + mb_size // 2
            cy = i * mb_size + mb_size // 2
            if dx != 0 or dy != 0:
                ax.annotate("", xy=(cx + dx, cy + dy), xytext=(cx, cy),
                            arrowprops=dict(arrowstyle="->", color="lime",
                                            lw=0.8, mutation_scale=6))
            else:
                ax.plot(cx, cy, ".", color="yellow", markersize=1.5)


# ══════════════════════════════════════════════════════════════════════════════
# Main visualisation function
# ══════════════════════════════════════════════════════════════════════════════

def visualise_pipeline(
        original_frames_bgr: list[np.ndarray],
        reconstructed_frames_bgr: list[np.ndarray],
        frame_types: list[str],
        preprocessed_sample: dict,          # output of preprocess_frame for frame 0
        encoded_frames: list[dict],         # raw encoded frame dicts
        metrics: dict,
        output_path: str = "pipeline_visualisation.png",
) -> None:
    """
    Produce the full pipeline visualisation figure (Part 5b).

    Layout (rows):
      Row 0  — Original frames (up to 5)
      Row 1  — Y / Cb / Cr channels of frame 0
      Row 2  — DCT / quantisation walkthrough (4 panels)
      Row 3  — Motion vectors + residual + reconstructed P-frame
      Row 4  — PSNR per frame bar chart + frame-type pie
    """

    MAX_FRAMES_SHOWN = min(5, len(original_frames_bgr))

    # ── Figure layout ─────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(20, 22), facecolor="#0d0d0d")
    fig.suptitle("Simplified MPEG-4 Pipeline Visualisation",
                 fontsize=18, fontweight="bold", color="white", y=0.98)

    outer = gridspec.GridSpec(5, 1, figure=fig,
                              hspace=0.45,
                              top=0.95, bottom=0.04,
                              left=0.04, right=0.97)

    # ── Helper: section label ─────────────────────────────────────────────────
    def section_label(ax_title: str, gs_item):
        dummy = fig.add_subplot(gs_item)
        dummy.set_visible(False)
        fig.text(
            dummy.get_position().x0,
            dummy.get_position().y1 + 0.005,
            ax_title,
            fontsize=10, fontweight="bold", color="#00e5ff",
            transform=fig.transFigure
        )

    # ═══════════════════════════════════════════════════════════════════════
    # ROW 0 — Original frames
    # ═══════════════════════════════════════════════════════════════════════
    section_label("① Original Frames", outer[0])
    gs0 = gridspec.GridSpecFromSubplotSpec(1, MAX_FRAMES_SHOWN,
                                           subplot_spec=outer[0], wspace=0.06)
    for k in range(MAX_FRAMES_SHOWN):
        ax = fig.add_subplot(gs0[k])
        ftype = frame_types[k] if k < len(frame_types) else "?"
        _show_frame(ax, original_frames_bgr[k],
                    f"Frame {k+1}  [{ftype}]")

    # ═══════════════════════════════════════════════════════════════════════
    # ROW 1 — Color-space channels
    # ═══════════════════════════════════════════════════════════════════════
    section_label("② Colour Space  (Frame 1)", outer[1])
    gs1 = gridspec.GridSpecFromSubplotSpec(1, 4, subplot_spec=outer[1], wspace=0.06)

    ax_orig = fig.add_subplot(gs1[0])
    _show_frame(ax_orig, original_frames_bgr[0], "Original BGR")

    ax_Y  = fig.add_subplot(gs1[1])
    ax_Cb = fig.add_subplot(gs1[2])
    ax_Cr = fig.add_subplot(gs1[3])
    _show_channel(ax_Y,  preprocessed_sample["Y"],  "Y  (Luma)")
    _show_channel(ax_Cb, preprocessed_sample["Cb"], "Cb (Chroma-Blue)", cmap="Blues")
    _show_channel(ax_Cr, preprocessed_sample["Cr"], "Cr (Chroma-Red)",  cmap="Reds")

    # ═══════════════════════════════════════════════════════════════════════
    # ROW 2 — DCT & Quantisation walkthrough (one 8×8 block)
    # ═══════════════════════════════════════════════════════════════════════
    section_label("③ DCT & Quantisation  (8×8 block from Frame 1 Y-channel)", outer[2])
    gs2 = gridspec.GridSpecFromSubplotSpec(1, 4, subplot_spec=outer[2], wspace=0.10)

    Y0 = preprocessed_sample["Y"]
    # Pick a visually interesting block (avoid uniform sky/white patches)
    bh_max = (Y0.shape[0] // 8)
    bw_max = (Y0.shape[1] // 8)
    best_var, best_bi, best_bj = 0, bh_max // 2, bw_max // 2
    for bi in range(bh_max):
        for bj in range(bw_max):
            blk = Y0[bi*8:(bi+1)*8, bj*8:(bj+1)*8]
            v = float(np.var(blk))
            if v > best_var:
                best_var = v
                best_bi, best_bj = bi, bj
    raw_block = Y0[best_bi*8:(best_bi+1)*8, best_bj*8:(best_bj+1)*8]

    from intra_coding import scale_quantisation_matrix, LUMA_Q
    Qy = scale_quantisation_matrix(LUMA_Q, qf=50)

    shifted   = raw_block.astype(np.float32) - 128.0
    dct_coeffs = _dct2(shifted)
    quant_coeffs = np.round(dct_coeffs / Qy).astype(np.int16).astype(np.float32)
    from scipy.fftpack import idct
    def idct2(b): return idct(idct(b.T, norm="ortho").T, norm="ortho")
    dequant   = quant_coeffs * Qy
    recon_block = np.clip(idct2(dequant) + 128.0, 0, 255)

    ax_raw   = fig.add_subplot(gs2[0])
    ax_dct   = fig.add_subplot(gs2[1])
    ax_quant = fig.add_subplot(gs2[2])
    ax_recon = fig.add_subplot(gs2[3])

    _show_block(ax_raw,   raw_block,    "Raw Pixels",          cmap="gray",    annotate=True)
    _show_block(ax_dct,   dct_coeffs,   "DCT Coefficients",    cmap="RdBu_r",  annotate=False)
    _show_block(ax_quant, quant_coeffs, "Quantised Coeffs",    cmap="plasma",  annotate=True)
    _show_block(ax_recon, recon_block,  "Reconstructed Block", cmap="gray",    annotate=True)

    # ═══════════════════════════════════════════════════════════════════════
    # ROW 3 — Motion vectors + residual + reconstructed P-frame
    # ═══════════════════════════════════════════════════════════════════════
    section_label("④ Motion Vectors  &  ⑤ Residuals / Reconstruction", outer[3])
    gs3 = gridspec.GridSpecFromSubplotSpec(1, 4, subplot_spec=outer[3], wspace=0.06)

    # Find first P-frame
    p_idx = next((i for i, t in enumerate(frame_types) if t == "P"), None)

    if p_idx is not None and p_idx < len(encoded_frames):
        pf = encoded_frames[p_idx]
        orig_p = original_frames_bgr[p_idx]
        recon_p = reconstructed_frames_bgr[p_idx]

        # Motion vectors panel
        ax_mv = fig.add_subplot(gs3[0])
        _overlay_motion_vectors(ax_mv, orig_p, pf["motion_vectors"])

        # Residual Y (decoded from quant blocks)
        from inter_coding import decode_residual_channel
        Qy50 = scale_quantisation_matrix(LUMA_Q, pf["qf"])
        res_Y = decode_residual_channel(
            pf["residual_Y_blocks"], Qy50, pf["residual_Y_shape"]
        )
        # Crop to original Y shape
        oh, ow = pf["Y_original_shape"]
        res_Y = res_Y[:oh, :ow]

        ax_res = fig.add_subplot(gs3[1])
        res_display = np.clip(res_Y + 128, 0, 255)
        ax_res.imshow(res_display, cmap="seismic", vmin=0, vmax=255)
        ax_res.set_title("Residual Y  (P-frame)", fontsize=8, fontweight="bold")
        ax_res.axis("off")

        ax_recon_p = fig.add_subplot(gs3[2])
        _show_frame(ax_recon_p, recon_p, f"Reconstructed  Frame {p_idx+1}")

        ax_diff = fig.add_subplot(gs3[3])
        diff = cv2.absdiff(orig_p.astype(np.uint8), recon_p.astype(np.uint8))
        diff_vis = np.clip(diff * 5, 0, 255).astype(np.uint8)
        ax_diff.imshow(cv2.cvtColor(diff_vis, cv2.COLOR_BGR2RGB))
        ax_diff.set_title("Difference  ×5", fontsize=8, fontweight="bold")
        ax_diff.axis("off")
    else:
        # No P-frame available — show placeholder
        for k in range(4):
            ax = fig.add_subplot(gs3[k])
            ax.text(0.5, 0.5, "No P-frame\navailable",
                    ha="center", va="center", color="gray",
                    transform=ax.transAxes, fontsize=9)
            ax.axis("off")

    # ═══════════════════════════════════════════════════════════════════════
    # ROW 4 — Metrics: PSNR bar chart + frame-type pie
    # ═══════════════════════════════════════════════════════════════════════
    section_label("⑥ Quality Metrics", outer[4])
    gs4 = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=outer[4], wspace=0.25)

    # PSNR bar chart
    ax_psnr = fig.add_subplot(gs4[0])
    ax_psnr.set_facecolor("#1a1a2e")

    psnr_vals = [p if p != float("inf") else 60.0
                 for p in metrics["per_frame_psnr"]]
    bar_colors = ["#00bfff" if t == "I" else "#ff6b6b"
                  for t in metrics["frame_types"]]
    x_pos = np.arange(len(psnr_vals))

    bars = ax_psnr.bar(x_pos, psnr_vals, color=bar_colors, edgecolor="#333", linewidth=0.5)
    ax_psnr.set_xlabel("Frame index", color="white", fontsize=8)
    ax_psnr.set_ylabel("PSNR (dB)", color="white", fontsize=8)
    ax_psnr.set_title("PSNR per Frame", color="white", fontsize=9, fontweight="bold")
    ax_psnr.tick_params(colors="white", labelsize=7)
    ax_psnr.spines[:].set_color("#444")
    ax_psnr.yaxis.label.set_color("white")
    ax_psnr.xaxis.label.set_color("white")

    # Horizontal mean-PSNR line
    mean_psnr = metrics["mean_psnr"]
    ax_psnr.axhline(mean_psnr, color="#ffd700", linewidth=1.2, linestyle="--",
                    label=f"Mean {mean_psnr:.1f} dB")
    ax_psnr.legend(fontsize=7, facecolor="#222", labelcolor="white", edgecolor="#555")

    # Legend patches for I/P
    patch_i = mpatches.Patch(color="#00bfff", label="I-frame")
    patch_p = mpatches.Patch(color="#ff6b6b", label="P-frame")
    ax_psnr.legend(handles=[patch_i, patch_p,
                             mpatches.Patch(color="#ffd700",
                                            label=f"Mean {mean_psnr:.1f} dB")],
                   fontsize=7, facecolor="#222", labelcolor="white", edgecolor="#555")

    # Frame-type pie chart
    ax_pie = fig.add_subplot(gs4[1])
    ax_pie.set_facecolor("#1a1a2e")

    ni, np_ = metrics["num_iframes"], metrics["num_pframes"]
    wedge_colors = ["#00bfff", "#ff6b6b"]
    labels = [f"I-frames ({ni})", f"P-frames ({np_})"]
    sizes  = [ni, np_] if (ni + np_) > 0 else [1, 0]

    wedges, texts, autotexts = ax_pie.pie(
        sizes, labels=labels, colors=wedge_colors,
        autopct="%1.0f%%", startangle=90,
        textprops={"color": "white", "fontsize": 8},
        wedgeprops={"edgecolor": "#333", "linewidth": 0.8}
    )
    for at in autotexts:
        at.set_fontsize(8)
        at.set_color("white")

    cr = metrics["compression_ratio"]
    ax_pie.set_title(
        f"Frame-type breakdown\nCompression ratio: {cr:.2f}×",
        color="white", fontsize=9, fontweight="bold"
    )

    # ── Apply dark theme to all axes ─────────────────────────────────────────
    for ax in fig.get_axes():
        ax.set_facecolor("#1a1a2e") if ax.get_visible() else None
        for spine in ax.spines.values():
            spine.set_edgecolor("#333")
        ax.title.set_color("white")

    plt.savefig(output_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Pipeline visualisation saved → {output_path}")


# ══════════════════════════════════════════════════════════════════════════════
# Experimental analysis plots (for the report)
# ══════════════════════════════════════════════════════════════════════════════

def plot_compression_vs_qf(frames_bgr: list[np.ndarray],
                            qf_values: list[float] = None,
                            gop_size: int = 4,
                            output_path: str = "compression_vs_qf.png") -> None:
    """
    Plot compression ratio vs Quality Factor (QF).
    Encodes the video at each QF and records the compressed size.
    """
    if qf_values is None:
        qf_values = [5, 10, 20, 30, 50, 70, 85, 95]

    from entropy_coding import encode_video
    import tempfile, os

    ratios = []
    psnrs  = []
    tmp = tempfile.NamedTemporaryFile(suffix=".bin", delete=False)
    tmp.close()

    print("Running compression_vs_qf sweep …")
    for qf in qf_values:
        stats = encode_video(frames_bgr, tmp.name,
                             gop_size=gop_size, qf=qf, search_range=4)
        # Quick decode to get PSNR
        from entropy_coding import decode_video
        recon, _ = decode_video(tmp.name)
        psnr_vals = [_psnr_safe(o, r) for o, r in zip(frames_bgr, recon)]
        ratios.append(stats["compression_ratio"])
        psnrs.append(float(np.mean(psnr_vals)))
        print(f"  QF={qf:3d} → ratio={ratios[-1]:.2f}×  PSNR={psnrs[-1]:.1f} dB")

    os.unlink(tmp.name)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), facecolor="#0d0d0d")
    for ax in (ax1, ax2):
        ax.set_facecolor("#1a1a2e")
        ax.tick_params(colors="white")
        for spine in ax.spines.values():
            spine.set_edgecolor("#444")

    ax1.plot(qf_values, ratios, "o-", color="#00e5ff", linewidth=2, markersize=6)
    ax1.set_xlabel("Quality Factor (QF)", color="white")
    ax1.set_ylabel("Compression Ratio (×)", color="white")
    ax1.set_title("Compression Ratio vs QF", color="white", fontweight="bold")
    ax1.grid(alpha=0.2, color="white")

    ax2.plot(qf_values, psnrs, "s-", color="#ff6b6b", linewidth=2, markersize=6)
    ax2.set_xlabel("Quality Factor (QF)", color="white")
    ax2.set_ylabel("Mean PSNR (dB)", color="white")
    ax2.set_title("PSNR vs QF", color="white", fontweight="bold")
    ax2.grid(alpha=0.2, color="white")

    fig.suptitle("Rate–Distortion Analysis", color="white", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Saved → {output_path}")


def plot_compression_vs_gop(frames_bgr: list[np.ndarray],
                             gop_values: list[int] = None,
                             qf: float = 50,
                             output_path: str = "compression_vs_gop.png") -> None:
    """
    Plot compression ratio vs GOP size.
    """
    if gop_values is None:
        n = len(frames_bgr)
        gop_values = sorted(set([1, 2, 4, 8, max(1, n//2), n]))
        gop_values = [g for g in gop_values if g <= n]

    from entropy_coding import encode_video
    import tempfile, os

    ratios = []
    psnrs  = []
    tmp = tempfile.NamedTemporaryFile(suffix=".bin", delete=False)
    tmp.close()

    print("Running compression_vs_gop sweep …")
    for gop in gop_values:
        stats = encode_video(frames_bgr, tmp.name,
                             gop_size=gop, qf=qf, search_range=4)
        from entropy_coding import decode_video
        recon, _ = decode_video(tmp.name)
        psnr_vals = [_psnr_safe(o, r) for o, r in zip(frames_bgr, recon)]
        ratios.append(stats["compression_ratio"])
        psnrs.append(float(np.mean(psnr_vals)))
        print(f"  GOP={gop:3d} → ratio={ratios[-1]:.2f}×  PSNR={psnrs[-1]:.1f} dB")

    os.unlink(tmp.name)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), facecolor="#0d0d0d")
    for ax in (ax1, ax2):
        ax.set_facecolor("#1a1a2e")
        ax.tick_params(colors="white")
        for spine in ax.spines.values():
            spine.set_edgecolor("#444")

    ax1.plot(gop_values, ratios, "o-", color="#ffd700", linewidth=2, markersize=6)
    ax1.set_xlabel("GOP Size", color="white")
    ax1.set_ylabel("Compression Ratio (×)", color="white")
    ax1.set_title("Compression Ratio vs GOP Size", color="white", fontweight="bold")
    ax1.grid(alpha=0.2, color="white")

    ax2.plot(gop_values, psnrs, "s-", color="#a78bfa", linewidth=2, markersize=6)
    ax2.set_xlabel("GOP Size", color="white")
    ax2.set_ylabel("Mean PSNR (dB)", color="white")
    ax2.set_title("PSNR vs GOP Size", color="white", fontweight="bold")
    ax2.grid(alpha=0.2, color="white")

    fig.suptitle("GOP Size Analysis", color="white", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Saved → {output_path}")


def _psnr_safe(orig: np.ndarray, rec: np.ndarray, max_val: float = 255.0) -> float:
    mse = np.mean((orig.astype(np.float64) - rec.astype(np.float64)) ** 2)
    return 10 * np.log10(max_val**2 / mse) if mse > 0 else 60.0

