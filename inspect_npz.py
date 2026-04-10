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


def infer_period_seconds_from_segment(seg_t, seg_flap, dt):
    """Infer oscillation period from one movement segment."""
    if len(seg_flap) < 10:
        return None

    # Smooth lightly before peak detection.
    smooth_win = max(3, int(round(0.05 / dt)))
    xs = moving_average(seg_flap, smooth_win)

    # Simple local maxima.
    peaks = np.where((xs[1:-1] > xs[:-2]) & (xs[1:-1] >= xs[2:]))[0] + 1
    if len(peaks) >= 4:
        peak_periods = np.diff(seg_t[peaks])
        peak_periods = peak_periods[peak_periods > 0]
        if len(peak_periods) > 0:
            return float(np.median(peak_periods))

    # Fallback to dominant frequency via FFT.
    x0 = seg_flap - np.mean(seg_flap)
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


def detect_steady_block_per_movement(t, z07, flap, movement_segments, dt):
    """Return one steady block per movement section.

    Output rows:
    (movement_id, start_idx, end_idx, start_time, end_time, mean_z07, period_secs, search_start_idx, search_end_idx)
    """
    # Time settings (seconds)
    slope_window_secs = 3
    std_window_secs = 6.0
    min_steady_periods = 10.0
    ignore_front_seconds = 35.0
    ignore_back_seconds = 50.0

    # Convert static windows to samples.
    slope_win = max(5, int(round(slope_window_secs / dt)))
    std_win = max(7, int(round(std_window_secs / dt)))

    print(
        f"Settings: per-segment period inference enabled, min_steady_periods={min_steady_periods:.2f}, "
        f"slope_win={slope_window_secs:.1f}s, "
        f"std_win={std_window_secs:.1f}s, "
        f"ignore_front_seconds={ignore_front_seconds:.2f}s, "
        f"ignore_back_seconds={ignore_back_seconds:.2f}s"
    )

    out = []
    period_history = []

    for move_id, (ms, me) in enumerate(movement_segments, start=1):
        seg_t = t[ms:me]
        seg_z = z07[ms:me]
        seg_flap = flap[ms:me]

        raw_period = infer_period_seconds_from_segment(seg_t, seg_flap, dt)
        period_secs = raw_period

        # Guard against noisy or unrealistic period estimates.
        valid_period = period_secs is not None and 0.30 <= period_secs <= 3.00
        if not valid_period:
            if period_history:
                period_secs = float(np.median(period_history[-5:]))
                print(
                    f"Movement {move_id}: raw period={raw_period} out of range, "
                    f"using recent median={period_secs:.3f}s"
                )
            else:
                period_secs = 1.0
                print(f"Movement {move_id}: period inference failed, using default 1.0s")
        else:
            period_history.append(period_secs)

        min_steady_secs = min_steady_periods * period_secs
        min_steady_samples = max(8, int(round(min_steady_secs / dt)))
        ignore_front_samples = max(0, int(round(ignore_front_seconds / dt)))
        ignore_back_samples = max(0, int(round(ignore_back_seconds / dt)))
        print(
            f"Movement {move_id}: inferred period={period_secs:.3f}s "
            f"(raw={raw_period}), "
            f"min_steady={min_steady_secs:.3f}s, "
            f"ignore_front={ignore_front_seconds:.3f}s, "
            f"ignore_back={ignore_back_seconds:.3f}s"
        )

        if len(seg_z) < max(std_win, min_steady_samples):
            print(f"Movement {move_id}: too short")
            continue

        dzdt = np.gradient(seg_z, seg_t)
        slope_metric = moving_average(np.abs(dzdt), slope_win)
        detrended = seg_z - moving_average(seg_z, slope_win)
        std_metric = np.sqrt(moving_average(detrended**2, std_win))

        # Fixed-size search window controlled only by front/back ignore seconds.
        search_start = ignore_front_samples
        search_end = len(seg_z) - ignore_back_samples
        if search_end <= search_start:
            print(f"Movement {move_id}: invalid search window (front/back ignore too large)")
            continue
        if (search_end - search_start) < min_steady_samples:
            print(f"Movement {move_id}: search window shorter than min steady duration")
            continue

        tail_slope = slope_metric[search_start:search_end]
        tail_std = std_metric[search_start:search_end]
        if len(tail_slope) < min_steady_samples:
            print(f"Movement {move_id}: not enough post-gate points")
            continue

        # Primary detection: adaptive thresholds on post-gate tail.
        slope_thr = float(np.nanquantile(tail_slope, 0.75))
        std_thr = float(np.nanquantile(tail_std, 0.75))

        steady_mask = (slope_metric <= slope_thr) & (std_metric <= std_thr)
        steady_mask[:search_start] = False
        steady_mask[search_end:] = False

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
        search_start_global = ms + search_start
        search_end_global = ms + search_end
        out.append(
            (
                move_id,
                gs,
                ge,
                t[gs],
                t[ge - 1],
                mean_val,
                period_secs,
                search_start_global,
                search_end_global,
            )
        )

    return out


def plot_movement_segments(t, z07, flap, movement_segments, figsize=(12, 4)):
    import matplotlib.pyplot as plt
    import numpy as _np

    plt.figure(figsize=figsize)
    plt.plot(t, z07, label="z07 (mm)", color="C0")
    plt.plot(t, flap, label="Flap_position (mm)", color="C1", alpha=0.9)

    try:
        ymax = float(_np.nanmax(z07))
        ymin = float(_np.nanmin(z07))
    except Exception:
        ymax, ymin = 1.0, 0.0
    yr = ymax - ymin if ymax != ymin else max(1.0, abs(ymax) * 0.1)

    for i, (s, e) in enumerate(movement_segments, start=1):
        t0 = float(t[s]) if s < len(t) else float(t[-1])
        t1 = float(t[e - 1]) if (e - 1) < len(t) else float(t[-1])
        plt.axvspan(t0, t1, color="orange", alpha=0.25)
        plt.vlines([t0, t1], ymin, ymax, color="k", linestyle="--", linewidth=0.8)
        plt.text((t0 + t1) / 2, ymax - 0.05 * yr, f"mov{i}", ha="center", va="top", fontsize=8)

    plt.xlabel("time (s)")
    plt.legend()
    plt.tight_layout()
    plt.show()


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

        # Movement segments between flat segments, plus an optional first
        # pre-flat movement segment if movement exists before first flat block.
        movement_segments = []

        first_flat_start = flat_segments[0][0]
        pre_first_move_idx = np.where(flap_diff[:first_flat_start] > flat_thresh)[0]
        if len(pre_first_move_idx) > 0:
            pre_start = int(pre_first_move_idx[0])
            if pre_start < first_flat_start:
                movement_segments.append((pre_start, first_flat_start))

        for i in range(len(flat_segments) - 1):
            prev_end = flat_segments[i][1]
            next_start = flat_segments[i + 1][0]
            if prev_end < next_start:
                movement_segments.append((prev_end, next_start))

        print("\nMovement segments:")
        for i, (s, e) in enumerate(movement_segments, start=1):
            print(f" {i}: {s}-{e} ({t[e-1]-t[s]:.1f}s)")

        steady_blocks = detect_steady_block_per_movement(
            t, z07, flap, movement_segments, dt
        )

        if not steady_blocks:
            print("No steady-state blocks found.")
            return

        print("\nDetected steady-state blocks:")
        for m_id, s, e, t0, t1, mean_z, period_s, ss_gs, ss_ge in steady_blocks:
            print(
                f" movement {m_id}: idx {s}-{e}, time {t0:.2f}-{t1:.2f}, "
                f"dur {t1-t0:.1f}s, mean {mean_z:.3f}, period {period_s:.3f}s"
            )

        # Final plot
        try:
            plot_movement_segments(t, z07, flap, movement_segments)
        except Exception as _ex:
            print(f"plot_movement_segments failed: {_ex}")

        # Additional plot: period and steady-state selections together.
        fig2, axp = plt.subplots(figsize=(14, 5))
        axp.plot(t, z07, color="tab:blue", lw=1, label="z07")
        for m_id, s, e, t0, t1, mean_z, period_s, ss_gs, ss_ge in steady_blocks:
            axp.axvspan(t[ss_gs], t[ss_ge - 1], color="tab:green", alpha=0.08)
            axp.axvspan(t0, t1, color="yellow", alpha=0.28)
            axp.hlines(mean_z, t0, t1, color="red", lw=2, linestyles="--")
            axp.text((t0 + t1) / 2, mean_z, f"M{m_id} P={period_s:.2f}s", fontsize=7, ha="center")
        axp.set_title("Per-movement period + selected steady-state window")
        axp.set_xlabel("Time")
        axp.set_ylabel("z07")
        axp.grid(True, alpha=0.25)
        axp.legend(loc="upper right")
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()
