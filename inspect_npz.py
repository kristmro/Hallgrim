#!/usr/bin/env python3
"""Inspect .npz files and plot contained arrays against time."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def build_time_axis(data: np.ndarray, time_key: str | None, arrays: dict[str, np.ndarray]) -> np.ndarray:
    if time_key:
        if time_key not in arrays:
            raise KeyError(f"time key '{time_key}' not found in npz file")
        time = np.asarray(arrays[time_key]).squeeze()
    elif "time" in arrays:
        time = np.asarray(arrays["time"]).squeeze()
    else:
        time = np.arange(data.shape[0])

    if time.ndim != 1:
        raise ValueError("time axis must be one-dimensional after squeeze")
    if len(time) != data.shape[0]:
        raise ValueError(f"time length ({len(time)}) must match first data dimension ({data.shape[0]})")
    return time


def inspect_npz(npz_path: Path) -> dict[str, np.ndarray]:
    with np.load(npz_path) as archive:
        arrays = {key: archive[key] for key in archive.files}

    print(f"Loaded: {npz_path}")
    print("Arrays:")
    for name, arr in arrays.items():
        print(f"  - {name}: shape={arr.shape}, dtype={arr.dtype}")
    return arrays


def plot_array(name: str, arr: np.ndarray, time: np.ndarray, output_dir: Path | None) -> None:
    y = np.asarray(arr)
    if y.ndim == 1:
        series = [y]
    else:
        flattened = y.reshape(y.shape[0], -1)
        series = [flattened[:, i] for i in range(flattened.shape[1])]

    plt.figure(figsize=(10, 4))
    for idx, values in enumerate(series):
        label = None if len(series) == 1 else f"{name}[{idx}]"
        plt.plot(time, values, linewidth=1, label=label)

    plt.title(name)
    plt.xlabel("time")
    plt.ylabel("value")
    if len(series) > 1:
        plt.legend(loc="best", fontsize="small")
    plt.tight_layout()

    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        out_path = output_dir / f"{name}.png"
        plt.savefig(out_path, dpi=150)
        print(f"  saved plot: {out_path}")
        plt.close()
    else:
        plt.show()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect an .npz archive and plot each array against time."
    )
    parser.add_argument("npz_file", type=Path, help="Path to the .npz file")
    parser.add_argument(
        "--time-key",
        default=None,
        help="Name of array to use as time axis. Defaults to 'time' if present, else index.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory to save PNG plots. If omitted, plots open interactively.",
    )
    parser.add_argument(
        "--skip",
        nargs="*",
        default=[],
        help="Array names to skip plotting (for example the time axis key).",
    )
    args = parser.parse_args()

    arrays = inspect_npz(args.npz_file)
    plotted = 0

    for name, arr in arrays.items():
        if name in args.skip:
            continue
        if args.time_key and name == args.time_key:
            continue
        if not args.time_key and name == "time":
            continue
        if arr.ndim == 0:
            print(f"  skipping scalar '{name}'")
            continue

        time = build_time_axis(np.asarray(arr), args.time_key, arrays)
        print(f"  plotting '{name}'")
        plot_array(name, np.asarray(arr), time, args.output_dir)
        plotted += 1

    if plotted == 0:
        print("No plottable arrays found.")


if __name__ == "__main__":
    main()
