"""
Part 4 — Entropy Coding
- Serialise all frame data (I-frames and P-frames) into a binary stream
- Apply zlib lossless compression
- Write to a .bin output file
- Implement a corresponding decoder
"""

import zlib
import pickle
import numpy as np
import os


# ── Serialisation helpers ─────────────────────────────────────────────────────

def serialise_frames(frames: list[dict]) -> bytes:
    """
    Serialise a list of frame dicts (I-frames and P-frames) to bytes using pickle.
    Each frame dict contains numpy arrays (quantised blocks, motion vectors, etc.)
    """
    return pickle.dumps(frames, protocol=pickle.HIGHEST_PROTOCOL)


def deserialise_frames(data: bytes) -> list[dict]:
    """
    Deserialise bytes back to a list of frame dicts.
    """
    return pickle.loads(data)


# ── Entropy encode / decode ───────────────────────────────────────────────────

def entropy_encode(frames: list[dict], output_path: str) -> dict:
    """
    Encode a list of frame dicts to a compressed .bin file.

    Steps:
      1. Serialise all frames with pickle
      2. Compress with zlib (level 9 = maximum compression)
      3. Write a small header + compressed payload to output_path

    Returns a dict with stats:
      - original_size   : bytes before compression
      - compressed_size : bytes after compression
      - compression_ratio
      - num_iframes
      - num_pframes
    """
    # Step 1 — serialise
    raw_bytes = serialise_frames(frames)
    original_size = len(raw_bytes)

    # Step 2 — compress
    compressed = zlib.compress(raw_bytes, level=9)
    compressed_size = len(compressed)

    # Step 3 — build header and write file
    num_frames = len(frames)
    num_iframes = sum(1 for f in frames if f["type"] == "I")
    num_pframes = sum(1 for f in frames if f["type"] == "P")

    # Header: magic bytes + metadata (also pickled for simplicity)
    header = pickle.dumps({
        "magic": b"MMPEG4",
        "num_frames": num_frames,
        "num_iframes": num_iframes,
        "num_pframes": num_pframes,
        "original_size": original_size,
        "compressed_size": compressed_size,
    })

    # Write: 4-byte header length + header + compressed payload
    header_len = len(header).to_bytes(4, byteorder="big")

    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)

    with open(output_path, "wb") as f:
        f.write(header_len)
        f.write(header)
        f.write(compressed)

    compression_ratio = original_size / compressed_size if compressed_size > 0 else float("inf")

    return {
        "original_size": original_size,
        "compressed_size": compressed_size,
        "compression_ratio": compression_ratio,
        "num_iframes": num_iframes,
        "num_pframes": num_pframes,
        "num_frames": num_frames,
        "output_path": output_path,
    }


def entropy_decode(input_path: str) -> tuple[list[dict], dict]:
    """
    Decode a .bin file back to a list of frame dicts.

    Returns:
      - frames : list of frame dicts (ready to pass to decode_iframe / decode_pframe)
      - header : metadata dict
    """
    with open(input_path, "rb") as f:
        header_len = int.from_bytes(f.read(4), byteorder="big")
        header_bytes = f.read(header_len)
        compressed = f.read()

    header = pickle.loads(header_bytes)

    # Verify magic
    if header.get("magic") != b"MMPEG4":
        raise ValueError("Invalid .bin file: bad magic bytes")

    raw_bytes = zlib.decompress(compressed)
    frames = deserialise_frames(raw_bytes)

    return frames, header


# ── Full pipeline encode/decode ───────────────────────────────────────────────

def encode_video(frames_bgr: list, output_path: str,
                 gop_size: int = 8, qf: float = 50,
                 search_range: int = 8) -> dict:
    """
    Full pipeline: list of BGR frames → compressed .bin file.

    Parameters
    ----------
    frames_bgr   : list of H×W×3 uint8 BGR numpy arrays
    output_path  : path to write the .bin file
    gop_size     : Group of Pictures size (every gop_size-th frame is an I-frame)
    qf           : quality factor (1–100)
    search_range : motion estimation search window ±S pixels

    Returns
    -------
    stats dict with compression info + per-frame types
    """
    from preprocessing import preprocess_frame
    from intra_coding import encode_iframe, decode_iframe
    from inter_coding import encode_pframe

    encoded_frames = []
    ref_Y = ref_Cb = ref_Cr = None
    frame_types = []

    for idx, frame_bgr in enumerate(frames_bgr):
        proc = preprocess_frame(frame_bgr)
        Y, Cb_sub, Cr_sub = proc["Y"], proc["Cb_sub"], proc["Cr_sub"]

        if idx % gop_size == 0:
            # I-frame
            frame_data = encode_iframe(Y, Cb_sub, Cr_sub, qf=qf)
            # Decode immediately to get the reconstructed reference
            ref_Y, ref_Cb, ref_Cr = decode_iframe(frame_data)
            frame_types.append("I")
        else:
            # P-frame
            frame_data = encode_pframe(
                Y, Cb_sub, Cr_sub,
                ref_Y, ref_Cb, ref_Cr,
                qf=qf, search_range=search_range
            )
            # Update reference with reconstructed P-frame
            from inter_coding import decode_pframe
            ref_Y, ref_Cb, ref_Cr = decode_pframe(frame_data, ref_Y, ref_Cb, ref_Cr)
            frame_types.append("P")

        encoded_frames.append(frame_data)
        print(f"  Encoded frame {idx+1}/{len(frames_bgr)} [{frame_types[-1]}]")

    stats = entropy_encode(encoded_frames, output_path)
    stats["frame_types"] = frame_types
    stats["gop_size"] = gop_size
    stats["qf"] = qf

    return stats


def decode_video(input_path: str) -> tuple[list, dict]:
    """
    Full pipeline: compressed .bin file → list of reconstructed BGR frames.

    Returns
    -------
    frames_bgr : list of H×W×3 uint8 BGR numpy arrays
    header     : metadata dict
    """
    from preprocessing import ycbcr_to_bgr
    from intra_coding import decode_iframe
    from inter_coding import decode_pframe

    encoded_frames, header = entropy_decode(input_path)

    reconstructed_bgr = []
    ref_Y = ref_Cb = ref_Cr = None

    for idx, frame_data in enumerate(encoded_frames):
        if frame_data["type"] == "I":
            ref_Y, ref_Cb, ref_Cr = decode_iframe(frame_data)
        else:
            ref_Y, ref_Cb, ref_Cr = decode_pframe(frame_data, ref_Y, ref_Cb, ref_Cr)

        bgr = ycbcr_to_bgr(ref_Y, ref_Cb, ref_Cr)
        reconstructed_bgr.append(bgr)
        print(f"  Decoded frame {idx+1}/{len(encoded_frames)} [{frame_data['type']}]")

    return reconstructed_bgr, header

