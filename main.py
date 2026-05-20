"""
main.py — Full MPEG-4-like pipeline (Parts 1–5)

Usage:
    python main.py --frames_dir frames/ --output video.bin --gop 8 --qf 50
    python main.py --decode --input video.bin --output_dir decoded_frames/
"""

import argparse
import os
import glob
import cv2
import numpy as np


def load_frames(frames_dir: str) -> list[np.ndarray]:
    """Load all .png / .jpg frames from a directory, sorted by name."""
    patterns = ["*.png", "*.jpg", "*.jpeg", "*.PNG", "*.JPG"]
    paths = []
    for pat in patterns:
        paths.extend(glob.glob(os.path.join(frames_dir, pat)))
    paths = sorted(set(paths))
    if not paths:
        raise FileNotFoundError(f"No image files found in {frames_dir}")
    frames = []
    for p in paths:
        img = cv2.imread(p)
        if img is None:
            print(f"  Warning: could not read {p}, skipping.")
            continue
        frames.append(img)
    print(f"Loaded {len(frames)} frames from '{frames_dir}'")
    return frames


def encode_pipeline(frames_dir: str, output_bin: str,
                    gop: int, qf: float, search_range: int) -> None:
    """End-to-end encode: frames → .bin + visualisation."""
    from preprocessing import preprocess_frame
    from intra_coding import encode_iframe, decode_iframe
    from inter_coding import encode_pframe, decode_pframe
    from entropy_coding import entropy_encode
    from evaluation import (compute_metrics, print_metrics,
                             visualise_pipeline,
                             plot_compression_vs_qf,
                             plot_compression_vs_gop)
    from intra_coding import scale_quantisation_matrix, LUMA_Q

    frames_bgr = load_frames(frames_dir)
    if not frames_bgr:
        raise RuntimeError("No frames to encode.")

    # ── Parts 1–3: encode all frames ──────────────────────────────────────────
    encoded_frames = []
    reconstructed_frames = []
    frame_types = []
    ref_Y = ref_Cb = ref_Cr = None

    # Keep preprocessed data for frame 0 (used in visualisation)
    preprocessed_sample = preprocess_frame(frames_bgr[0])

    print("\n[Encoding]")
    for idx, frame_bgr in enumerate(frames_bgr):
        proc = preprocess_frame(frame_bgr)
        Y, Cb_sub, Cr_sub = proc["Y"], proc["Cb_sub"], proc["Cr_sub"]

        if idx % gop == 0:
            fd = encode_iframe(Y, Cb_sub, Cr_sub, qf=qf)
            ref_Y, ref_Cb, ref_Cr = decode_iframe(fd)
            frame_types.append("I")
        else:
            fd = encode_pframe(Y, Cb_sub, Cr_sub,
                               ref_Y, ref_Cb, ref_Cr,
                               qf=qf, search_range=search_range)
            ref_Y, ref_Cb, ref_Cr = decode_pframe(fd, ref_Y, ref_Cb, ref_Cr)
            frame_types.append("P")

        from preprocessing import ycbcr_to_bgr
        rec_bgr = ycbcr_to_bgr(ref_Y, ref_Cb, ref_Cr)
        reconstructed_frames.append(rec_bgr)
        encoded_frames.append(fd)
        print(f"  Frame {idx+1:3d}/{len(frames_bgr)}  [{frame_types[-1]}]")

    # ── Part 4: entropy encode ────────────────────────────────────────────────
    print(f"\n[Part 4] Writing compressed file → {output_bin}")
    stats = entropy_encode(encoded_frames, output_bin)
    print(f"  Original size   : {stats['original_size'] / 1024:.1f} KB  (serialised)")
    print(f"  Compressed size : {stats['compressed_size'] / 1024:.1f} KB")
    print(f"  Compression ratio (bitstream): {stats['compression_ratio']:.2f}×")

    # ── Part 5a: metrics ──────────────────────────────────────────────────────
    print("\n[Part 5a] Quality Metrics")
    bin_size = os.path.getsize(output_bin)
    metrics = compute_metrics(frames_bgr, reconstructed_frames, bin_size, frame_types)
    print_metrics(metrics)

    # ── Part 5b: visualisation ────────────────────────────────────────────────
    print("\n[Part 5b] Generating pipeline visualisation …")
    os.makedirs("output_part5", exist_ok=True)

    visualise_pipeline(
        original_frames_bgr=frames_bgr,
        reconstructed_frames_bgr=reconstructed_frames,
        frame_types=frame_types,
        preprocessed_sample=preprocessed_sample,
        encoded_frames=encoded_frames,
        metrics=metrics,
        output_path="output_part5/pipeline_visualisation.png",
    )

    # ── Experimental analysis (report plots) ──────────────────────────────────
    # Only run when there are enough frames to make the sweep meaningful
    if len(frames_bgr) >= 4:
        print("\n[Report] Compression ratio vs QF …")
        plot_compression_vs_qf(
            frames_bgr,
            qf_values=[5, 10, 20, 30, 50, 70, 85, 95],
            gop_size=gop,
            output_path="output_part5/compression_vs_qf.png"
        )

        print("\n[Report] Compression ratio vs GOP size …")
        plot_compression_vs_gop(
            frames_bgr,
            gop_values=None,  # auto-chosen based on frame count
            qf=qf,
            output_path="output_part5/compression_vs_gop.png"
        )

    print("\nDone.  All outputs in output_part5/")


def decode_pipeline(input_bin: str, output_dir: str) -> None:
    """End-to-end decode: .bin → reconstructed BGR frames saved as PNGs."""
    from entropy_coding import decode_video
    import os

    print(f"\n[Decoding] {input_bin} → {output_dir}")
    reconstructed, header = decode_video(input_bin)
    os.makedirs(output_dir, exist_ok=True)
    for idx, frame_bgr in enumerate(reconstructed):
        path = os.path.join(output_dir, f"frame{idx+1:04d}.png")
        cv2.imwrite(path, frame_bgr)
    print(f"Saved {len(reconstructed)} frames to '{output_dir}'")
    print(f"Header info: {header}")


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simplified MPEG-4 Pipeline")
    parser.add_argument("--decode", action="store_true",
                        help="Run decode mode instead of encode")
    parser.add_argument("--frames_dir", default="frames/",
                        help="Directory with input frames (encode mode)")
    parser.add_argument("--input",  default="video.bin",
                        help=".bin file to decode (decode mode)")
    parser.add_argument("--output", default="video.bin",
                        help="Output .bin file (encode) or decoded frames dir (decode)")
    parser.add_argument("--output_dir", default="decoded_frames/",
                        help="Directory to save decoded frames")
    parser.add_argument("--gop",  type=int,   default=8,
                        help="GOP size (default 8)")
    parser.add_argument("--qf",   type=float, default=50,
                        help="Quality factor 1–100 (default 50)")
    parser.add_argument("--search", type=int, default=8,
                        help="Motion estimation search range ±S (default 8)")

    args = parser.parse_args()

    if args.decode:
        decode_pipeline(args.input, args.output_dir)
    else:
        encode_pipeline(args.frames_dir, args.output, args.gop, args.qf, args.search)
        
