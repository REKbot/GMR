"""Trim an animation NPZ file to a specific time range.

Usage:
    python scripts/trim_npz.py input.npz output.npz --start 4.0 --end 10.0

The script auto-detects which arrays are time-series (first axis == total
frames) and slices only those. All other arrays (e.g. fps, metadata) are
passed through untouched.
"""

import argparse

import numpy as np


def trim_npz(input_path: str, output_path: str, start_sec: float, end_sec: float):
    data = np.load(input_path, allow_pickle=True)

    # detect fps and total frames
    fps_array = data["fps"]
    fps = float(fps_array[0]) if fps_array.ndim > 0 else float(fps_array)

    # Find the longest time-series axis to determine total frame count
    n_frames = max(v.shape[0] for v in data.values() if v.ndim >= 1 and v.shape[0] > 1)

    start_frame = int(round(start_sec * fps))
    end_frame = int(round(end_sec * fps))

    # Clamp to valid range
    start_frame = max(0, min(start_frame, n_frames))
    end_frame = max(start_frame, min(end_frame, n_frames))

    print(f"FPS: {fps}")
    print(f"Total frames: {n_frames}  ({n_frames / fps:.3f} s)")
    print(
        f"Trimming frames {start_frame}–{end_frame}  "
        f"({start_frame / fps:.3f} s – {end_frame / fps:.3f} s)"
    )
    print(
        f"Output frames: {end_frame - start_frame}  "
        f"({(end_frame - start_frame) / fps:.3f} s)"
    )

    # build output dict
    out = {}
    for key in data.files:
        arr = data[key]
        # Trim any array whose first axis matches total frame count
        if arr.ndim >= 1 and arr.shape[0] == n_frames:
            out[key] = arr[start_frame:end_frame]
            print(f"  trimmed  '{key}': {arr.shape} → {out[key].shape}")
        else:
            out[key] = arr
            print(f"  kept     '{key}': {arr.shape}")

    np.savez(output_path, **out)
    print(f"\nSaved → {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Trim animation NPZ to a time range.")
    parser.add_argument("input", help="Input .npz file")
    parser.add_argument("output", help="Output .npz file")
    parser.add_argument(
        "--start",
        type=float,
        default=0.0,
        help="Start time in seconds (default: 0)",
    )
    parser.add_argument(
        "--end",
        type=float,
        default=10.0,
        help="End time in seconds (default: 10)",
    )
    args = parser.parse_args()

    trim_npz(args.input, args.output, args.start, args.end)
