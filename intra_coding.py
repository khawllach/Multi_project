"""
Part 2 — Intra-frame Coding (I-frames)
- Split channel into 8×8 blocks
- Apply 2-D DCT
- Quantise with the standard JPEG luminance / chrominance matrices
- Dequantise + IDCT to reconstruct
"""

import numpy as np
from scipy.fftpack import dct, idct


# ── Standard JPEG quantisation matrices ──────────────────────────────────────

LUMA_Q = np.array([
    [16, 11, 10, 16, 24, 40, 51, 61],
    [12, 12, 14, 19, 26, 58, 60, 55],
    [14, 13, 16, 24, 40, 57, 69, 56],
    [14, 17, 22, 29, 51, 87, 80, 62],
    [18, 22, 37, 56, 68,109,103, 77],
    [24, 35, 55, 64, 81,104,113, 92],
    [49, 64, 78, 87,103,121,120,101],
    [72, 92, 95, 98,112,100,103, 99],
], dtype=np.float32)

CHROMA_Q = np.array([
    [17, 18, 24, 47, 99, 99, 99, 99],
    [18, 21, 26, 66, 99, 99, 99, 99],
    [24, 26, 56, 99, 99, 99, 99, 99],
    [47, 66, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
], dtype=np.float32)


def scale_quantisation_matrix(base_q: np.ndarray, qf: float) -> np.ndarray:
    """
    Scale a base quantisation matrix by quality factor qf (1–100).
    qf=50 → no scaling; qf<50 → coarser; qf>50 → finer.
    """
    if qf <= 0:
        qf = 1
    elif qf > 100:
        qf = 100

    if qf < 50:
        scale = 5000.0 / qf
    else:
        scale = 200.0 - 2.0 * qf

    q = np.floor((base_q * scale + 50) / 100).astype(np.float32)
    q = np.clip(q, 1, 255)
    return q#array


# ── 2-D DCT / IDCT helpers ───────────────────────────────────────────────────

def dct2(block: np.ndarray) -> np.ndarray:
    """2-D DCT-II (orthonormal) of an 8×8 block."""
    return dct(dct(block.T, norm="ortho").T, norm="ortho")


def idct2(block: np.ndarray) -> np.ndarray:
    """Inverse 2-D DCT-II."""
    return idct(idct(block.T, norm="ortho").T, norm="ortho")


# ── Block-level encode / decode ───────────────────────────────────────────────

def encode_block(block: np.ndarray, Q: np.ndarray) -> np.ndarray:
    """
    Encode one 8×8 block:
      1. Level-shift: subtract 128
      2. DCT
      3. Quantise (round to nearest int)
    Returns int16 quantised coefficients.
    """
    shifted = block.astype(np.float32) - 128.0 #Because DCT works better when numbers are around zero
    coeffs  = dct2(shifted)
    quant   = np.round(coeffs / Q).astype(np.int16)
    return quant


def decode_block(quant: np.ndarray, Q: np.ndarray) -> np.ndarray:
    """
    Decode one 8×8 block:
      1. Dequantise
      2. IDCT
      3. Level-unshift: add 128
    Returns float32 pixel values.
    """
    coeffs = quant.astype(np.float32) * Q
    recon  = idct2(coeffs) + 128.0
    return recon


# ── Full-channel encode / decode ─────────────────────────────────────────────

def pad_to_multiple(channel: np.ndarray, block_size: int = 8) -> np.ndarray:
    """Pad channel so height and width are multiples of block_size."""
    h, w = channel.shape
    ph = (block_size - h % block_size) % block_size
    pw = (block_size - w % block_size) % block_size
    return np.pad(channel, ((0, ph), (0, pw)), mode="edge")


def encode_channel(channel: np.ndarray, Q: np.ndarray) -> tuple[np.ndarray, tuple]:
    """
    Encode a full channel (arbitrary size) block by block.
    Returns:
      - quant_blocks: array of shape (num_blocks_h, num_blocks_w, 8, 8) int16
      - original_shape: (h, w) before padding
    """
    original_shape = channel.shape
    padded = pad_to_multiple(channel)
    h, w = padded.shape
    bh, bw = h // 8, w // 8

    quant_blocks = np.zeros((bh, bw, 8, 8), dtype=np.int16)
    for i in range(bh):
        for j in range(bw):
            block = padded[i*8:(i+1)*8, j*8:(j+1)*8]
            quant_blocks[i, j] = encode_block(block, Q)

    return quant_blocks, original_shape


def decode_channel(quant_blocks: np.ndarray, Q: np.ndarray,
                   original_shape: tuple) -> np.ndarray:
    """
    Decode a full channel from quant_blocks.
    Crops to original_shape.
    """
    bh, bw = quant_blocks.shape[:2]
    recon = np.zeros((bh * 8, bw * 8), dtype=np.float32)

    for i in range(bh):
        for j in range(bw):
            recon[i*8:(i+1)*8, j*8:(j+1)*8] = decode_block(quant_blocks[i, j], Q)

    h, w = original_shape
    return np.clip(recon[:h, :w], 0, 255)


# ── I-frame encode / decode ───────────────────────────────────────────────────

def encode_iframe(Y: np.ndarray, Cb_sub: np.ndarray, Cr_sub: np.ndarray,
                  qf: float = 50) -> dict:
    """
    Encode one I-frame.
    Returns a dict with quantised blocks for Y, Cb, Cr and their original shapes.
    """
    Qy = scale_quantisation_matrix(LUMA_Q,   qf)
    Qc = scale_quantisation_matrix(CHROMA_Q, qf)

    Y_blocks,  Y_shape  = encode_channel(Y,      Qy)
    Cb_blocks, Cb_shape = encode_channel(Cb_sub, Qc)
    Cr_blocks, Cr_shape = encode_channel(Cr_sub, Qc)

    return {
        "type": "I",
        "qf": qf,
        "Y_blocks":  Y_blocks,  "Y_shape":  Y_shape,
        "Cb_blocks": Cb_blocks, "Cb_shape": Cb_shape,
        "Cr_blocks": Cr_blocks, "Cr_shape": Cr_shape,
    }#contains the compressed coefficients.


def decode_iframe(frame_data: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Decode an I-frame dict.
    Returns reconstructed Y, Cb_sub, Cr_sub (float32).
    """
    qf = frame_data["qf"]
    Qy = scale_quantisation_matrix(LUMA_Q,   qf)
    Qc = scale_quantisation_matrix(CHROMA_Q, qf)

    Y      = decode_channel(frame_data["Y_blocks"],  Qy, frame_data["Y_shape"])
    Cb_sub = decode_channel(frame_data["Cb_blocks"], Qc, frame_data["Cb_shape"])
    Cr_sub = decode_channel(frame_data["Cr_blocks"], Qc, frame_data["Cr_shape"])

    return Y, Cb_sub, Cr_sub
