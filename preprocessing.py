"""
Part 1 — Pre-processing
- BGR -> YCbCr color space conversion
- 4:2:0 chroma subsampling
"""

import numpy as np
import cv2


def bgr_to_ycbcr(frame_bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert a BGR frame (uint8, H×W×3) to YCbCr.
    Returns Y, Cb, Cr as float32 arrays.
    """
    # OpenCV converts BGR -> YCrCb (note: Cr before Cb in OpenCV)
    ycrcb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2YCrCb).astype(np.float32)
    Y  = ycrcb[:, :, 0]
    Cb = ycrcb[:, :, 2]   # OpenCV channel 2 = Cb
    Cr = ycrcb[:, :, 1]   # OpenCV channel 1 = Cr
    return Y, Cb, Cr


def ycbcr_to_bgr(Y: np.ndarray, Cb: np.ndarray, Cr: np.ndarray) -> np.ndarray:
    """
    Inverse: YCbCr -> BGR uint8.
    Upsamples Cb/Cr if they were subsampled.
    """
    h, w = Y.shape
    Cb_up = cv2.resize(Cb, (w, h), interpolation=cv2.INTER_LINEAR)
    Cr_up = cv2.resize(Cr, (w, h), interpolation=cv2.INTER_LINEAR)

    ycrcb = np.stack([Y, Cr_up, Cb_up], axis=2)          # OpenCV YCrCb order
    ycrcb = np.clip(ycrcb, 0, 255).astype(np.uint8)
    bgr = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)
    return bgr


def subsample_420(Cb: np.ndarray, Cr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    4:2:0 chroma subsampling: downsample Cb and Cr by 2 in both dimensions.
    """
    Cb_sub = Cb[::2, ::2]
    Cr_sub = Cr[::2, ::2]
    return Cb_sub, Cr_sub


def preprocess_frame(frame_bgr: np.ndarray) -> dict:
    """
    Full preprocessing of one BGR frame.
    Returns dict with Y (full-res), Cb_sub, Cr_sub (half-res), plus originals for viz.
    """
    Y, Cb, Cr = bgr_to_ycbcr(frame_bgr)
    Cb_sub, Cr_sub = subsample_420(Cb, Cr)
    return {
        "shape": frame_bgr.shape,
        "Y": Y,
        "Cb": Cb,         # full-res (for visualisation)
        "Cr": Cr,
        "Cb_sub": Cb_sub, # stored / encoded
        "Cr_sub": Cr_sub,
    }
