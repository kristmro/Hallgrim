#!/usr/bin/env python3
"""Inspect NPZ data and detect steady-state z07 blocks during flap movement."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

NPZ_PATH = Path(r"C:\Users\kmroe\Desktop\5 klasse\hallgrim\S1_30T06_16_R1_LANG_0102.npz")


def first_index(keys: np.ndarray, name: str):
    idx = np.where(keys == name)[0]
    if len(idx) == 0:
        return None
    return int(idx[0])


def contiguous_true_blocks(mask: np.ndarray, min_len: int = 1):
    blocks = []
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


def moving_average(x: np.ndarray, window: int):
    w = max(1, int(window))
    return np.convolve(x, np.ones(w) / w, mode="same")


def infer_period_seconds_from_flap(t, flap, movement_segments, dt):
    """Infer oscillation period directly from flap signal during movement sections."""
    if len(movement_segments) == 0:
        return None

    mask = np.zeros(len(flap), dtype=bool)
    for s, e in movement_segments:
        mask[s:e] = True

    tm = t[mask]
    xm = flap[mask]
    if len(xm) < 10:
        return None

    # Smooth lightly before peak detection.
    smooth_win = max(3, int(round(0.05 / dt)))
    xs = moving_average(xm, smooth_win)

    # Simple local maxima.
    peaks = np.where((xs[1:-1] > xs[:-2]) & (xs[1:-1] >= xs[2:]))[0] + 1
    if len(peaks) >= 4:
        peak_periods = np.diff(tm[peaks])
        peak_periods = peak_periods[peak_periods > 0]
        if len(peak_periods) > 0:
            return float(np.median(peak_periods))

    # Fallback to dominant frequency via FFT.
    x0 = xm - np.mean(xm)
    if np.allclose(x0, 0.0):
        return None

    freqs = np.fft.rfftfreq(len(x0), d=dt)
    power = np.abs(np.fft.rfft(x0)) ** 2
    valid = (freqs > 0.05) & (freqs < 10.0)
    if not np.any(valid):
        return None
    f_dom = freqs[valid][np.argmax(power[valid])]
    if f_dom <= 0:
        return None
    return float(1.0 / f_dom)


def detect_steady_block_per_movement(t, z07, flap, movement_segments, dt, period_secs):
    """Return one steady block per movement section.

    Output rows: (movement_id, start_idx, end_idx, start_time, end_time, mean_z07)
    """
    # Time settings (seconds)
    slope_window_secs = 0.6
    std_window_secs = 6.0
    min_steady_periods = 3.0
    post_peak_delay_secs = 2.0
    flap_progress_ratio = 0.85

    # Convert to samples
    slope_win = max(5, int(round(slope_window_secs / dt)))
    std_win = max(7, int(round(std_window_secs / dt)))
    min_steady_secs = min_steady_periods * period_secs
    min_steady_samples = max(8, int(round(min_steady_secs / dt)))
    post_peak_delay_samples = max(0, int(round(post_peak_delay_secs / dt)))

    print(
        f"Settings: period={period_secs:.3f}s, min_steady_periods={min_steady_periods:.2f}, "
        f"min_steady={min_steady_secs:.3f}s, slope_win={slope_window_secs:.1f}s, "
        f"std_win={std_window_secs:.1f}s, post_peak_delay={post_peak_delay_secs:.1f}s, "
        f"flap_progress_ratio={flap_progress_ratio:.2f}"
    )

    out = []

    for move_id, (ms, me) in enumerate(movement_segments, start=1):
        seg_t = t[ms:me]
        seg_z = z07[ms:me]
        seg_flap = flap[ms:me]

        if len(seg_z) < max(std_win, min_steady_samples):
            print(f"Movement {move_id}: too short")
            continue

        dzdt = np.gradient(seg_z, seg_t)
        slope_metric = moving_average(np.abs(dzdt), slope_win)
        detrended = seg_z - moving_average(seg_z, slope_win)
        std_metric = np.sqrt(moving_average(detrended**2, std_win))

        # Gate 1: after major z07 response peak
        baseline = float(np.median(seg_z))
        peak_idx = int(np.argmax(np.abs(seg_z - baseline)))
        gate_peak = min(len(seg_z) - 1, peak_idx + post_peak_delay_samples)

        # Gate 2: after flap reaches most of its step
        flap_delta = np.abs(seg_flap - float(seg_flap[0]))
        flap_target = flap_progress_ratio * float(np.max(flap_delta))
        flap_hit = np.where(flap_delta >= flap_target)[0]
        gate_flap = int(flap_hit[0]) if len(flap_hit) > 0 else 0

        search_start = max(gate_peak, gate_flap)

        tail_slope = slope_metric[search_start:]
        tail_std = std_metric[search_start:]
        if len(tail_slope) < min_steady_samples:
            print(f"Movement {move_id}: not enough post-gate points")
            continue

        # Primary detection: adaptive thresholds on post-gate tail.
        slope_thr = float(np.nanquantile(tail_slope, 0.75))
        std_thr = float(np.nanquantile(tail_std, 0.75))

        steady_mask = (slope_metric <= slope_thr) & (std_metric <= std_thr)
        steady_mask[:search_start] = False

        blocks = contiguous_true_blocks(steady_mask, min_len=min_steady_samples)

        if blocks:
            # Choose latest (later in movement) and then longest.
            best_start, best_end = max(blocks, key=lambda b: (b[0], b[1] - b[0]))
        else:
            # Fallback: always pick the "most steady" window in the tail,
            # based on combined normalized slope/std score.
            eps = 1e-12
            slope_base = float(np.nanmedian(tail_slope)) + eps
            std_base = float(np.nanmedian(tail_std)) + eps
            score_tail = (tail_slope / slope_base) + (tail_std / std_base)

            score_smooth = moving_average(score_tail, min_steady_samples)
            best_center_tail = int(np.nanargmin(score_smooth))

            tail_start = max(0, best_center_tail - (min_steady_samples // 2))
            tail_end = min(len(score_tail), tail_start + min_steady_samples)
            tail_start = max(0, tail_end - min_steady_samples)

            best_start = search_start + tail_start
            best_end = search_start + tail_end
            print(
                f"Movement {move_id}: fallback steady window selected "
                f"(no threshold block found)."
            )

        gs = ms + best_start
        ge = ms + best_end
        mean_val = float(np.mean(z07[gs:ge]))
        out.append((move_id, gs, ge, t[gs], t[ge - 1], mean_val))

    return out


def main():
    if not NPZ_PATH.exists():
        print(f"NPZ file not found: {NPZ_PATH}")
        print("Edit NPZ_PATH and run again.")
        return

    with np.load(NPZ_PATH, allow_pickle=True) as archive:
        keys = archive["keys"]
        if keys.dtype.kind in {"S", "O"}:
            keys = np.array([k.decode() if isinstance(k, bytes) else str(k) for k in keys])

        data = archive["data"].item() if archive["data"].shape == () else archive["data"]

        required = {
            "Time  1 - default sample rate": None,
            "z07": None,
            "WP1": None,
            "Flap_position": None,
        }

        print("Loaded:", NPZ_PATH)
        for k in required:
            required[k] = first_index(keys, k)
            if required[k] is None:
                print(f"Missing key: {k}")

        if any(required[k] is None for k in required):
            print("One or more required keys missing.")
            return

        t = np.asarray(data[required["Time  1 - default sample rate"]], dtype=float)
        z07 = np.asarray(data[required["z07"]], dtype=float)
        wp1 = np.asarray(data[required["WP1"]], dtype=float) * 1000.0
        flap = np.asarray(data[required["Flap_position"]], dtype=float)

        # Overview plot
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

        # Flat flap detection
        dt = np.median(np.diff(t)) if len(t) > 1 else 1.0
        min_flat_secs = 180.0
        min_flat_samples = max(1, int(np.ceil(min_flat_secs / dt)))
        flat_thresh = 0.02

        flap_diff = np.abs(np.diff(flap, prepend=flap[0]))
        flat_mask = flap_diff <= flat_thresh
        flat_segments = []
        for s, e in contiguous_true_blocks(flat_mask, min_len=min_flat_samples):
            flat_segments.append((s, e, t[e - 1] - t[s]))

        if not flat_segments:
            print("No flat flap segments found. Tune thresholds.")
            return

        print("\nFlat flap segments:")
        for i, (s, e, dur) in enumerate(flat_segments, start=1):
            print(f" {i}: {s}-{e} ({dur:.1f}s)")

        # Movement segments = gaps between flat segments
        movement_segments = []
        last_end = 0
        for s, e, _ in flat_segments:
            if last_end < s:
                movement_segments.append((last_end, s))
            last_end = e
        if last_end < len(t):
            movement_segments.append((last_end, len(t)))

        print("\nMovement segments:")
        for i, (s, e) in enumerate(movement_segments, start=1):
            print(f" {i}: {s}-{e} ({t[e-1]-t[s]:.1f}s)")

        period_secs = infer_period_seconds_from_flap(t, flap, movement_segments, dt)
        if period_secs is None:
            period_secs = 1.0
            print("Could not infer period from flap signal; using default period=1.0s")
        else:
            print(f"Inferred period from flap signal: {period_secs:.3f}s")

        steady_blocks = detect_steady_block_per_movement(
            t, z07, flap, movement_segments, dt, period_secs
        )

        if not steady_blocks:
            print("No steady-state blocks found.")
            return

        print("\nDetected steady-state blocks:")
        for m_id, s, e, t0, t1, mean_z in steady_blocks:
            print(f" movement {m_id}: idx {s}-{e}, time {t0:.2f}-{t1:.2f}, dur {t1-t0:.1f}s, mean {mean_z:.3f}")

        # Final plot
        fig, ax1 = plt.subplots(figsize=(14, 6))
        ax2 = ax1.twinx()

        ax1.plot(t, z07, color="tab:blue", linewidth=1, label="z07")
        ax2.plot(t, flap, color="tab:orange", linewidth=1, alpha=0.7, label="Flap_position")

        for _, s, e, t0, t1, mean_z in steady_blocks:
            ax1.axvspan(t0, t1, color="yellow", alpha=0.25)
            ax1.hlines(mean_z, t0, t1, color="red", linestyle="--", linewidth=2)

        ax1.set_xlabel("Time")
        ax1.set_ylabel("z07", color="tab:blue")
        ax2.set_ylabel("Flap position", color="tab:orange")
        ax1.set_title("z07 steady-state (one block per movement)")
        ax1.grid(True, alpha=0.3)

        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()
