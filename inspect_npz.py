#!/usr/bin/env python3
"""Inspect NPZ and detect one continuous steady-state block per flap movement segment."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# 1) Put your .npz path here.
NPZ_PATH = Path(r"C:\Users\kmroe\Desktop\5 klasse\hallgrim\S1_30T06_16_R1_LANG_0102.npz")


def _first_index(keys: np.ndarray, name: str) -> int | None:
    matches = np.where(keys == name)[0]
    return int(matches[0]) if len(matches) else None


def _contiguous_true_blocks(mask: np.ndarray, min_len: int = 1) -> list[tuple[int, int]]:
    blocks: list[tuple[int, int]] = []
    start = None
    for i, flag in enumerate(mask):
        if flag and start is None:
            start = i
        elif (not flag) and start is not None:
            if i - start >= min_len:
                blocks.append((start, i))
            start = None
    if start is not None and len(mask) - start >= min_len:
        blocks.append((start, len(mask)))
    return blocks


def _moving_average(x: np.ndarray, window: int) -> np.ndarray:
    window = max(1, int(window))
    kernel = np.ones(window, dtype=float) / window
    return np.convolve(x, kernel, mode="same")


def detect_steady_block_per_movement(
    t: np.ndarray,
    z: np.ndarray,
    flap: np.ndarray,
    movement_segments: list[tuple[int, int]],
    dt: float,
) -> list[tuple[int, int, int, float, float, float]]:
    """Return one continuous steady block per movement.

    Returns tuples:
        (movement_id, start_idx, end_idx, start_time, end_time, steady_mean)
    """
    steady_blocks: list[tuple[int, int, int, float, float, float]] = []

    # Window and duration settings in seconds (converted to samples using dt).
    slope_window_secs = 0.6
    std_window_secs = 6.0
    flap_progress_ratio = 0.85
        f"flap_progress_ratio={flap_progress_ratio:.2f} "
        seg_flap = flap[ms:me]
        peak_gate = min(n - 1, peak_idx + post_peak_delay)

        # Additional gate: only allow detection after flap has reached most of
        # the step amplitude (helps avoid selecting too early in each step).
        flap_start = float(seg_flap[0])
        flap_delta = np.abs(seg_flap - flap_start)
        flap_target = flap_progress_ratio * float(np.max(flap_delta))
        flap_candidates = np.where(flap_delta >= flap_target)[0]
        flap_gate = int(flap_candidates[0]) if len(flap_candidates) else 0

        search_start = max(peak_gate, flap_gate)

    slope_win = max(5, int(round(slope_window_secs / dt)))
    std_win = max(7, int(round(std_window_secs / dt)))
    min_steady = max(8, int(round(min_steady_secs / dt)))

    print(
        "Steady-state settings: "
        f"slope_window={slope_window_secs:.1f}s, "
        f"std_window={std_window_secs:.1f}s, "
        f"min_steady={min_steady_secs:.1f}s "
        "(converted to samples using dt)."
    )

    for move_id, (ms, me) in enumerate(movement_segments, start=1):
        seg_t = t[ms:me]
        seg_z = z[ms:me]
        n = len(seg_z)
        if n < max(std_win, min_steady):
            print(f" Movement {move_id} too short for steady detection.")
            continue

        dz = np.gradient(seg_z, seg_t)
        slope_metric = _moving_average(np.abs(dz), slope_win)
        std_metric = np.sqrt(_moving_average((seg_z - _moving_average(seg_z, slope_win)) ** 2, std_win))

        # Adaptive thresholds per movement segment.
        slope_thr = np.nanquantile(slope_metric, 0.40)
        std_thr = np.nanquantile(std_metric, 0.35)

        candidate = (slope_metric <= slope_thr) & (std_metric <= std_thr)
        blocks = _contiguous_true_blocks(candidate, min_len=min_steady)
        if not blocks:
            print(f" Movement {move_id}: no steady block (try relaxing thresholds).")
            continue

        # Pick a single continuous block per movement: prefer the latest one in the segment.
        best_start, best_end = max(blocks, key=lambda b: (b[1], b[1] - b[0]))

        g_start = ms + best_start
        g_end = ms + best_end
        steady_mean = float(np.mean(z[g_start:g_end]))
        steady_blocks.append((move_id, g_start, g_end, t[g_start], t[g_end - 1], steady_mean))

    return steady_blocks


def main() -> None:
    npz_file = NPZ_PATH
    if not npz_file.exists():
        print(f"NPZ file not found: {npz_file}")
        print("Edit NPZ_PATH in inspect_npz.py and run again.")
        return

    with np.load(npz_file, allow_pickle=True) as archive:
        print(f"Loaded: {npz_file}")

        keys = archive["keys"]
        if keys.dtype.kind in {"S", "O"}:
            keys = np.array([k.decode() if isinstance(k, bytes) else str(k) for k in keys])

        data_obj = archive["data"].item() if archive["data"].shape == () else archive["data"]

        want_names = [
            "Time  1 - default sample rate",
            "z07",
            "WP1",
            "Flap_position",
        ]

        print("\nSelected series")
        for name in want_names:
            idx = _first_index(keys, name)
            if idx is None:
                print(f"- {name}: NOT FOUND")
                continue
            series = np.asarray(data_obj[idx])
            print(f"- {name}: index {idx}, shape={series.shape}, dtype={series.dtype}")
            print(f"  sample [first 5] = {series[:5]}")

        print("\nFull key legend count", len(keys))

        idx_t = _first_index(keys, "Time  1 - default sample rate")
        idx_z = _first_index(keys, "z07")
        idx_w = _first_index(keys, "WP1")
        idx_f = _first_index(keys, "Flap_position")

        if None in [idx_t, idx_z, idx_w, idx_f]:
            print("One or more required keys missing for plotting")
            return

        t = np.asarray(data_obj[idx_t], dtype=float)
        z07 = np.asarray(data_obj[idx_z], dtype=float)
        wp1 = np.asarray(data_obj[idx_w], dtype=float) * 1000.0
        flap = np.asarray(data_obj[idx_f], dtype=float)

        plt.figure(figsize=(11, 6))
        plt.plot(t, flap, label="Flap_position", linewidth=1)
        plt.plot(t, wp1, label="WP1 (x1000 mm)", linewidth=1)
        plt.plot(t, z07, label="z07", linewidth=1)
        plt.xlabel("Time")
        plt.ylabel("Value")
        plt.title("Flap_position, WP1, z07 vs Time")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        dt = np.median(np.diff(t)) if len(t) > 1 else 1.0
        min_flat_secs = 180.0
        min_flat_samples = max(1, int(np.ceil(min_flat_secs / dt)))
        flat_thresh = 0.02

        diff = np.abs(np.diff(flap, prepend=flap[0]))
        flat_mask = diff <= flat_thresh
        flat_segments = []
        for s, e in _contiguous_true_blocks(flat_mask, min_len=min_flat_samples):
            flat_segments.append((s, e, t[e - 1] - t[s]))

        if not flat_segments:
            print("No flat flap segment ~3 min found (threshold/min duration may need tuning)")
            return

        print("\nDetected flat flap segments (index range, duration sec):")
        for i, (s, e, dur) in enumerate(flat_segments, start=1):
            print(f" {i}: {s}-{e} -> {dur:.1f}s (~{dur / 60:.1f}min)")

        movement_segments = []
        last_end = 0
        for (s, e, _) in flat_segments:
            if last_end < s:
                movement_segments.append((last_end, s))
            last_end = e
        if last_end < len(t):
            movement_segments.append((last_end, len(t)))

        print("\nMovement segments (non-flat) for steady-state detection:")
        for i, (ms, me) in enumerate(movement_segments, start=1):
            duration = t[me - 1] - t[ms]
            print(f" {i}: {ms}-{me} -> {duration:.1f}s")

        steady_blocks = detect_steady_block_per_movement(t, z07, flap, movement_segments, dt)

        if not steady_blocks:
            print("No steady-state segments found during movement intervals.")
            return

        print("\nDetected steady-state block per movement:")
        for idx, bs, be, bt0, bt1, mean_v in steady_blocks:
            print(
                f" movement {idx}: global {bs}-{be}, time {bt0:.2f}-{bt1:.2f}, "
                f"duration {bt1 - bt0:.1f}s, mean {mean_v:.3f}"
            )

        fig, ax1 = plt.subplots(figsize=(14, 6))
        ax2 = ax1.twinx()

        ax1.plot(t, z07, label="z07", linewidth=1, color="tab:blue")
        ax2.plot(t, flap, label="Flap_position", linewidth=1, color="tab:orange", alpha=0.7)

        for _, start_i, end_i, start_t, end_t, mean_v in steady_blocks:
            ax1.axvspan(start_t, end_t, color="yellow", alpha=0.25)
            ax1.hlines(mean_v, start_t, end_t, color="red", linewidth=2, linestyle="--")

        ax1.set_xlabel("Time")
        ax1.set_ylabel("z07", color="tab:blue")
        ax2.set_ylabel("Flap_position", color="tab:orange")
        ax1.set_title("z07 steady-state (one continuous block per flap movement)")
        ax1.grid(True, alpha=0.3)

        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()
