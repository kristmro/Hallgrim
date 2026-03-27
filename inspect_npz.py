#!/usr/bin/env python3
"""Start-simple NPZ inspector: set path, run script, list objects."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# 1) Put your .npz path here.
NPZ_PATH = Path(r"C:\Users\kmroe\Desktop\5 klasse\hallgrim\S1_30T06_16_R1_LANG_0102.npz")


def main() -> None:
    npz_file = NPZ_PATH
    if not npz_file.exists():
        print(f"NPZ file not found: {npz_file}")
        print("Edit NPZ_PATH in inspect_npz.py and run again.")
        return

    with np.load(npz_file, allow_pickle=True) as archive:
        print(f"Loaded: {npz_file}")

        keys = archive["keys"]
        data_obj = archive["data"].item() if archive["data"].shape == () else archive["data"]

        # Find wanted indexes by key names
        want_names = [
            "Time  1 - default sample rate",
            "z07",
            "WP1",
            "Flap_position",
        ]

        print("\nSelected series")
        for name in want_names:
            if name in keys:
                idx = int(np.where(keys == name)[0][0])
                series = data_obj[idx]
                print(f"- {name}: index {idx}, shape={series.shape}, dtype={series.dtype}")
                print(f"  sample [first 5] = {series[:5]}")
            else:
                print(f"- {name}: NOT FOUND")

        # Optional: show all key names listed
        print("\nFull key legend count", len(keys))
        print(keys)

        # Plot selected series vs time
        matches = {name: np.where(keys == name)[0] for name in want_names}
        if all(len(matches[name]) > 0 for name in ["Time  1 - default sample rate", "z07", "WP1", "Flap_position"]):
            t = data_obj[int(matches["Time  1 - default sample rate"][0])]
            z07 = data_obj[int(matches["z07"][0])]
            wp1 = data_obj[int(matches["WP1"][0])] * 1000.0  # convert from meters to millimeters
            flap = data_obj[int(matches["Flap_position"][0])]

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

            # Split data around flat flap regions
            dt = np.median(np.diff(t)) if len(t) > 1 else 1.0
            min_flat_secs = 180.0
            min_flat_samples = max(1, int(np.ceil(min_flat_secs / dt)))
            flat_thresh = 0.02  # mm change threshold per sample (tune if needed)
            diff = np.abs(np.diff(flap, prepend=flap[0]))
            flat_mask = diff <= flat_thresh

            # find flat segments
            flat_segments = []
            start = None
            for i, is_flat in enumerate(flat_mask):
                if is_flat and start is None:
                    start = i
                elif not is_flat and start is not None:
                    end = i
                    if end - start >= min_flat_samples:
                        flat_segments.append((start, end, t[end - 1] - t[start]))
                    start = None
            if start is not None:
                end = len(flat_mask)
                if end - start >= min_flat_samples:
                    flat_segments.append((start, end, t[end - 1] - t[start]))

            if not flat_segments:
                print("No flat flap segment ~3 min found (threshold/min duration may need tuning)")
            else:
                print("\nDetected flat flap segments (index range, duration sec):")
                for idx, (s, e, dur) in enumerate(flat_segments, start=1):
                    print(f" {idx}: {s}-{e} -> {dur:.1f}s (~{dur/60:.1f}min)")

                # movement segments are between flat periods
                movement_segments = []
                last_end = 0
                for (s, e, _) in flat_segments:
                    if last_end < s:
                        movement_segments.append((last_end, s))
                    last_end = e
                if last_end < len(t):
                    movement_segments.append((last_end, len(t)))

                print("\nMovement segments (non-flat) for steady-state detection:")
                for idx, (ms, me) in enumerate(movement_segments, start=1):
                    duration = t[me - 1] - t[ms]
                    print(f" {idx}: {ms}-{me} -> {duration:.1f}s")

                # parameters
                # smaller smoothing window for dynamic motion between flats (e.g., 0.5 s)
                ma_window = max(3, int(round(0.5 / dt)))
                # steady-state window: require at least 10 s of low variance
                ss_window = max(3, int(round(10.0 / dt)))
                steady_thresh = max(0.01, 0.4 * np.std(z07))

                steady_blocks = []
                for idx, (ms, me) in enumerate(movement_segments, start=1):
                    seg_t = t[ms:me]
                    seg_z07 = z07[ms:me]
                    if len(seg_z07) < ss_window:
                        print(f" Movement {idx} too short for steady detection (need >={ss_window}).")
                        continue

                    seg_ma = np.convolve(seg_z07, np.ones(ma_window) / ma_window, mode="same")
                    resid = seg_z07 - seg_ma
                    var_roll = np.convolve(resid**2, np.ones(ss_window) / ss_window, mode="same")
                    std_roll = np.sqrt(var_roll)
                    steady_flag = std_roll <= steady_thresh

                    # collect steady window(s)
                    st_start = None
                    for i, flag in enumerate(steady_flag):
                        if flag and st_start is None:
                            st_start = i
                        elif not flag and st_start is not None:
                            st_end = i
                            if st_end - st_start >= ss_window:
                                steady_blocks.append((idx, ms + st_start, ms + st_end, seg_t[st_start], seg_t[st_end - 1]))
                            st_start = None
                    if st_start is not None and len(seg_z07) - st_start >= ss_window:
                        steady_blocks.append((idx, ms + st_start, ms + len(seg_z07), seg_t[st_start], seg_t[-1]))

                if not steady_blocks:
                    print("No steady-state segments found during movement intervals.")
                else:
                    print("\nDetected steady-state blocks during movement:")
                    for idx, bs, be, bt0, bt1 in steady_blocks:
                        print(f" movement {idx}: global {bs}-{be}, time {bt0:.2f}-{bt1:.2f} -> {bt1 - bt0:.1f}s")

                    plt.figure(figsize=(14, 6))
                    plt.plot(t, z07, label="z07", linewidth=1)
                    plt.plot(t, flap, label="Flap_position", linewidth=1, alpha=0.6)
                    for mb in steady_blocks:
                        _, start_i, end_i, start_t, end_t = mb
                        plt.axvspan(start_t, end_t, color="yellow", alpha=0.3)
                    plt.xlabel("Time")
                    plt.ylabel("z07 / flap")
                    plt.title("z07 steady-state sections during flap motion")
                    plt.legend()
                    plt.grid(True)
                    plt.tight_layout()
                    plt.show()
        else:
            print("One or more required keys missing for plotting")


if __name__ == "__main__":
    main()
