"""
Part 3 — Inter-frame Coding (P-frames)
- Group of Pictures (GOP): every G-th frame is an I-frame, rest are P-frames
- Motion estimation: exhaustive block matching on 16×16 macroblocks
- Residual = current - predicted; encoded with DCT + quantisation (8×8 sub-blocks)
"""

import numpy as np
from intra_coding import (
    encode_channel, decode_channel,
    encode_block, decode_block,
    scale_quantisation_matrix,
    LUMA_Q, CHROMA_Q,
    pad_to_multiple,
)


MB_SIZE = 16   # macroblock size


# ── Motion estimation ─────────────────────────────────────────────────────────

def full_search_block_matching(current: np.ndarray, reference: np.ndarray,
                                search_range: int = 8) -> tuple[np.ndarray, np.ndarray]:
    """
    Exhaustive (full-search) block matching for the Y channel.

    Parameters
    ----------
    current   : H×W float32 luma of current frame (padded to MB_SIZE multiples)
    reference : H×W float32 luma of previous reconstructed frame (same size)
    search_range : ±S pixel search window

    Returns
    -------
    motion_vectors : (mbh, mbw, 2) int array — (dy, dx) per macroblock
    predicted      : H×W float32 predicted frame
    """
    h, w = current.shape
    mbh, mbw = h // MB_SIZE, w // MB_SIZE

    motion_vectors = np.zeros((mbh, mbw, 2), dtype=np.int32)
    predicted = np.zeros_like(current)

    # Pad reference to allow border searches
    pad = search_range
    ref_padded = np.pad(reference, pad, mode="edge")

    for i in range(mbh):
        for j in range(mbw):
            y0, x0 = i * MB_SIZE, j * MB_SIZE
            cur_block = current[y0:y0+MB_SIZE, x0:x0+MB_SIZE]

            best_sad = float("inf")
            best_dy, best_dx = 0, 0

            # Search window
            for dy in range(-search_range, search_range + 1):
                for dx in range(-search_range, search_range + 1):
                    ry = y0 + pad + dy
                    rx = x0 + pad + dx
                    ref_block = ref_padded[ry:ry+MB_SIZE, rx:rx+MB_SIZE]
                    sad = np.sum(np.abs(cur_block - ref_block))
                    if sad < best_sad:
                        best_sad = sad
                        best_dy, best_dx = dy, dx

            motion_vectors[i, j] = [best_dy, best_dx] #motion_vectors[0, 1] = [1, 2] For this current block, the best matching block in the reference frame
                                                        #is shifted by 1 pixel down and 2 pixels right.

            # Fill prediction
            ry = y0 + pad + best_dy
            rx = x0 + pad + best_dx
            predicted[y0:y0+MB_SIZE, x0:x0+MB_SIZE] = ref_padded[ry:ry+MB_SIZE, rx:rx+MB_SIZE]

    return motion_vectors, predicted


def apply_motion_vectors(reference: np.ndarray, motion_vectors: np.ndarray,
                          mb_size: int = MB_SIZE) -> np.ndarray:
    """Build predicted frame from reference + motion vectors."""
    h, w = reference.shape
    mbh, mbw = h // mb_size, w // mb_size
    pad = motion_vectors.max() + 1 if motion_vectors.size else 8#find the padding
    pad = max(int(np.abs(motion_vectors).max()) + 1, 1)

    ref_padded = np.pad(reference, pad, mode="edge")
    predicted = np.zeros_like(reference)

    for i in range(mbh):
        for j in range(mbw):
            y0, x0 = i * mb_size, j * mb_size
            dy, dx = motion_vectors[i, j]
            ry = y0 + pad + dy
            rx = x0 + pad + dx
            predicted[y0:y0+mb_size, x0:x0+mb_size] = ref_padded[ry:ry+mb_size, rx:rx+mb_size]

    return predicted


# ── Residual encode / decode ──────────────────────────────────────────────────

def encode_residual_channel(residual: np.ndarray, Q: np.ndarray) -> tuple[np.ndarray, tuple]:
    """
    Encode a residual channel using 8×8 DCT blocks.
    Residual values are centred around 0 (no level shift needed).
    """
    original_shape = residual.shape
    padded = pad_to_multiple(residual)
    h, w = padded.shape
    bh, bw = h // 8, w // 8

    quant_blocks = np.zeros((bh, bw, 8, 8), dtype=np.int16)
    for i in range(bh):
        for j in range(bw):
            block = padded[i*8:(i+1)*8, j*8:(j+1)*8].astype(np.float32)
            from scipy.fftpack import dct, idct
            def dct2(b): return dct(dct(b.T, norm="ortho").T, norm="ortho")
            coeffs = dct2(block)           # No level shift for residuals
            quant_blocks[i, j] = np.round(coeffs / Q).astype(np.int16)

    return quant_blocks, original_shape


def decode_residual_channel(quant_blocks: np.ndarray, Q: np.ndarray,
                             original_shape: tuple) -> np.ndarray:
    """Decode a residual channel from quant blocks."""
    from scipy.fftpack import idct
    def idct2(b): return idct(idct(b.T, norm="ortho").T, norm="ortho")

    bh, bw = quant_blocks.shape[:2]
    recon = np.zeros((bh * 8, bw * 8), dtype=np.float32)

    for i in range(bh):
        for j in range(bw):
            coeffs = quant_blocks[i, j].astype(np.float32) * Q
            recon[i*8:(i+1)*8, j*8:(j+1)*8] = idct2(coeffs)

    h, w = original_shape
    return recon[:h, :w]


# ── P-frame encode / decode ───────────────────────────────────────────────────

def encode_pframe(Y: np.ndarray, Cb_sub: np.ndarray, Cr_sub: np.ndarray,
                  ref_Y: np.ndarray, ref_Cb_sub: np.ndarray, ref_Cr_sub: np.ndarray,
                  qf: float = 50, search_range: int = 8) -> dict:
    """
    Encode one P-frame given the previous reconstructed frame as reference.

    Returns a dict with motion vectors + residual blocks for Y, Cb, Cr.
    """
    Qy = scale_quantisation_matrix(LUMA_Q,   qf)
    Qc = scale_quantisation_matrix(CHROMA_Q, qf)

    # ── Luma: motion estimation on 16×16 macroblocks ─────────────────────────
    # Pad Y to macroblock boundary
    h, w = Y.shape
    ph = (MB_SIZE - h % MB_SIZE) % MB_SIZE
    pw = (MB_SIZE - w % MB_SIZE) % MB_SIZE
    Y_pad     = np.pad(Y,     ((0, ph), (0, pw)), mode="edge")
    ref_Y_pad = np.pad(ref_Y, ((0, ph), (0, pw)), mode="edge")

    motion_vectors, predicted_Y_pad = full_search_block_matching(
        Y_pad, ref_Y_pad, search_range
    )

    residual_Y_pad = Y_pad - predicted_Y_pad
    residual_Y_blocks, residual_Y_shape = encode_residual_channel(residual_Y_pad, Qy)

    # ── Chroma: simple inter prediction (half-pel motion vectors) ─────────────
    # Use half the MV (since chroma is half resolution), skip block-match
    mv_cb = motion_vectors // 2

    hc, wc = Cb_sub.shape
    phc = (MB_SIZE//2 - hc % (MB_SIZE//2)) % (MB_SIZE//2)# Calculate padding needed to make Cb_sub dimensions multiples of 8 (for DCT blocks)
    pwc = (MB_SIZE//2 - wc % (MB_SIZE//2)) % (MB_SIZE//2)

    Cb_pad     = np.pad(Cb_sub,     ((0, phc), (0, pwc)), mode="edge")
    Cr_pad     = np.pad(Cr_sub,     ((0, phc), (0, pwc)), mode="edge")
    ref_Cb_pad = np.pad(ref_Cb_sub, ((0, phc), (0, pwc)), mode="edge")
    ref_Cr_pad = np.pad(ref_Cr_sub, ((0, phc), (0, pwc)), mode="edge")

    pred_Cb = apply_motion_vectors(ref_Cb_pad, mv_cb, mb_size=MB_SIZE//2)
    pred_Cr = apply_motion_vectors(ref_Cr_pad, mv_cb, mb_size=MB_SIZE//2)

    res_Cb_pad = Cb_pad - pred_Cb
    res_Cr_pad = Cr_pad - pred_Cr

    res_Cb_blocks, res_Cb_shape = encode_residual_channel(res_Cb_pad, Qc)
    res_Cr_blocks, res_Cr_shape = encode_residual_channel(res_Cr_pad, Qc)

    return {
        "type": "P",
        "qf": qf,
        "search_range": search_range,
        "motion_vectors": motion_vectors,
        "Y_pad_shape": Y_pad.shape,
        "Y_original_shape": Y.shape,
        "residual_Y_blocks": residual_Y_blocks,
        "residual_Y_shape": residual_Y_shape,
        "Cb_pad_shape": Cb_pad.shape,
        "Cb_original_shape": Cb_sub.shape,
        "residual_Cb_blocks": res_Cb_blocks,
        "residual_Cb_shape": res_Cb_shape,
        "Cr_original_shape": Cr_sub.shape,
        "residual_Cr_blocks": res_Cr_blocks,
        "residual_Cr_shape": res_Cr_shape,
    }


def decode_pframe(frame_data: dict,
                  ref_Y: np.ndarray, ref_Cb_sub: np.ndarray,
                  ref_Cr_sub: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Decode a P-frame dict using the reference (previous reconstructed) frame.
    Returns reconstructed Y, Cb_sub, Cr_sub.
    """
    qf = frame_data["qf"]
    Qy = scale_quantisation_matrix(LUMA_Q,   qf)
    Qc = scale_quantisation_matrix(CHROMA_Q, qf)

    mv = frame_data["motion_vectors"]

    # ── Luma ──────────────────────────────────────────────────────────────────
    ph_shape = frame_data["Y_pad_shape"]
    ref_Y_pad = np.pad(ref_Y,
                       ((0, ph_shape[0] - ref_Y.shape[0]),
                        (0, ph_shape[1] - ref_Y.shape[1])),
                       mode="edge")
    predicted_Y_pad = apply_motion_vectors(ref_Y_pad, mv, MB_SIZE)
    residual_Y_pad  = decode_residual_channel(
        frame_data["residual_Y_blocks"], Qy, frame_data["residual_Y_shape"]
    )
    recon_Y_pad = predicted_Y_pad + residual_Y_pad
    h, w = frame_data["Y_original_shape"]
    Y = np.clip(recon_Y_pad[:h, :w], 0, 255)

    # ── Chroma ────────────────────────────────────────────────────────────────
    mv_cb = mv // 2
    cb_pad_shape = frame_data["Cb_pad_shape"]

    ref_Cb_pad = np.pad(ref_Cb_sub,
                        ((0, cb_pad_shape[0] - ref_Cb_sub.shape[0]),
                         (0, cb_pad_shape[1] - ref_Cb_sub.shape[1])),
                        mode="edge")
    ref_Cr_pad = np.pad(ref_Cr_sub,
                        ((0, cb_pad_shape[0] - ref_Cr_sub.shape[0]),
                         (0, cb_pad_shape[1] - ref_Cr_sub.shape[1])),
                        mode="edge")

    pred_Cb = apply_motion_vectors(ref_Cb_pad, mv_cb, MB_SIZE//2)
    pred_Cr = apply_motion_vectors(ref_Cr_pad, mv_cb, MB_SIZE//2)

    res_Cb = decode_residual_channel(frame_data["residual_Cb_blocks"], Qc,
                                     frame_data["residual_Cb_shape"])
    res_Cr = decode_residual_channel(frame_data["residual_Cr_blocks"], Qc,
                                     frame_data["residual_Cr_shape"])

    hc, wc = frame_data["Cb_original_shape"]
    Cb_sub = np.clip((pred_Cb + res_Cb)[:hc, :wc], 0, 255)
    Cr_sub = np.clip((pred_Cr + res_Cr)[:hc, :wc], 0, 255)

    return Y, Cb_sub, Cr_sub
