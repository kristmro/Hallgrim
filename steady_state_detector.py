#!/usr/bin/env python3
"""Detect and plot steady-state intervals in z07 during flap motion sections.

This script is designed for timeseries where:
- `flap` indicates flap movement/position
- `z07` is a sinus-like response that eventually reaches a steady state

For each contiguous flap-motion section, the script finds contiguous regions where
`z07` appears steady using rolling slope and rolling standard deviation criteria.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


@dataclass
class SteadySegment:
    start_idx: int
    end_idx: int
    mean_value: float


@dataclass
class SectionResult:
    section_start: int
    section_end: int
    steady_segments: List[SteadySegment]


def contiguous_true_ranges(mask: np.ndarray) -> List[Tuple[int, int]]:
    """Return inclusive-exclusive ranges where mask is True."""
    if mask.size == 0:
        return []

    starts = np.where(mask & ~np.r_[False, mask[:-1]])[0]
    ends = np.where(mask & ~np.r_[mask[1:], False])[0] + 1
    return list(zip(starts, ends))


def detect_flap_sections(
    flap: pd.Series,
    min_section_samples: int,
    flap_change_quantile: float,
) -> List[Tuple[int, int]]:
    """Find contiguous sections where flap is moving.

    Motion is estimated from abs first-difference of flap.
    """
    flap_rate = flap.diff().abs().fillna(0.0)
    threshold = flap_rate.quantile(flap_change_quantile)

    # If data are nearly constant, avoid zero-threshold selecting everything.
    if threshold <= 0:
        non_zero = flap_rate[flap_rate > 0]
        threshold = non_zero.quantile(0.25) if not non_zero.empty else 0.0

    moving_mask = flap_rate >= threshold if threshold > 0 else flap_rate > 0

    ranges = contiguous_true_ranges(moving_mask.to_numpy())
    return [(s, e) for (s, e) in ranges if (e - s) >= min_section_samples]


def detect_steady_segments_in_section(
    time: pd.Series,
    z07: pd.Series,
    section_start: int,
    section_end: int,
    window: int,
    slope_quantile: float,
    std_quantile: float,
    min_steady_samples: int,
) -> List[SteadySegment]:
    """Detect contiguous steady intervals within a flap-motion section."""
    sec_t = time.iloc[section_start:section_end].to_numpy(dtype=float)
    sec_z = z07.iloc[section_start:section_end].to_numpy(dtype=float)

    if len(sec_z) < max(window, min_steady_samples):
        return []

    dt = np.gradient(sec_t)
    dzdt = np.gradient(sec_z) / np.where(dt == 0, np.nan, dt)
    slope_abs = np.abs(dzdt)

    s = pd.Series(sec_z)
    rolling_std = s.rolling(window=window, center=True, min_periods=window // 2).std()
    rolling_slope = pd.Series(slope_abs).rolling(
        window=window, center=True, min_periods=window // 2
    ).mean()

    slope_thr = np.nanquantile(rolling_slope, slope_quantile)
    std_thr = np.nanquantile(rolling_std, std_quantile)

    steady_mask = (rolling_slope <= slope_thr) & (rolling_std <= std_thr)
    steady_mask = steady_mask.fillna(False).to_numpy()

    ranges = contiguous_true_ranges(steady_mask)

    results: List[SteadySegment] = []
    for local_start, local_end in ranges:
        if (local_end - local_start) < min_steady_samples:
            continue
        global_start = section_start + local_start
        global_end = section_start + local_end
        mean_val = float(z07.iloc[global_start:global_end].mean())
        results.append(
            SteadySegment(
                start_idx=global_start,
                end_idx=global_end,
                mean_value=mean_val,
            )
        )
    return results


def analyze(
    df: pd.DataFrame,
    time_col: str,
    z_col: str,
    flap_col: str,
    window: int,
    min_section_samples: int,
    min_steady_samples: int,
    flap_change_quantile: float,
    slope_quantile: float,
    std_quantile: float,
) -> List[SectionResult]:
    """Run section and steady-state detection."""
    sections = detect_flap_sections(
        flap=df[flap_col],
        min_section_samples=min_section_samples,
        flap_change_quantile=flap_change_quantile,
    )

    results: List[SectionResult] = []
    for sec_start, sec_end in sections:
        steady = detect_steady_segments_in_section(
            time=df[time_col],
            z07=df[z_col],
            section_start=sec_start,
            section_end=sec_end,
            window=window,
            slope_quantile=slope_quantile,
            std_quantile=std_quantile,
            min_steady_samples=min_steady_samples,
        )
        results.append(
            SectionResult(
                section_start=sec_start,
                section_end=sec_end,
                steady_segments=steady,
            )
        )
    return results


def plot_results(
    df: pd.DataFrame,
    results: List[SectionResult],
    time_col: str,
    z_col: str,
    flap_col: str,
    output_path: str,
) -> None:
    """Plot z07 with flap sections and detected steady-state intervals."""
    t = df[time_col].to_numpy()
    z = df[z_col].to_numpy()
    flap = df[flap_col].to_numpy()

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), sharex=True)

    ax1.plot(t, z, label=z_col, lw=1.5)
    for idx, sec in enumerate(results):
        t0, t1 = t[sec.section_start], t[sec.section_end - 1]
        ax1.axvspan(t0, t1, color="tab:blue", alpha=0.08, label="flap moving" if idx == 0 else None)

        for j, seg in enumerate(sec.steady_segments):
            ts, te = t[seg.start_idx], t[seg.end_idx - 1]
            ax1.axvspan(ts, te, color="tab:green", alpha=0.2, label="steady-state" if (idx == 0 and j == 0) else None)
            ax1.hlines(seg.mean_value, ts, te, color="tab:red", lw=2, linestyles="--", label="steady mean" if (idx == 0 and j == 0) else None)

    ax1.set_ylabel(z_col)
    ax1.set_title("Steady-state detection in z07 within flap-motion sections")
    ax1.grid(alpha=0.25)
    ax1.legend(loc="upper right")

    ax2.plot(t, flap, color="tab:orange", label=flap_col, lw=1.2)
    ax2.set_xlabel(time_col)
    ax2.set_ylabel(flap_col)
    ax2.grid(alpha=0.25)
    ax2.legend(loc="upper right")

    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_csv", help="CSV file containing time, z07, and flap columns")
    parser.add_argument("--time-col", default="time")
    parser.add_argument("--z-col", default="z07")
    parser.add_argument("--flap-col", default="flap")
    parser.add_argument("--window", type=int, default=41, help="Rolling window size (samples)")
    parser.add_argument("--min-section-samples", type=int, default=60)
    parser.add_argument("--min-steady-samples", type=int, default=50)
    parser.add_argument("--flap-change-quantile", type=float, default=0.70)
    parser.add_argument("--slope-quantile", type=float, default=0.35)
    parser.add_argument("--std-quantile", type=float, default=0.35)
    parser.add_argument("--output-plot", default="steady_state_detection.png")
    parser.add_argument("--output-csv", default="steady_segments.csv")
    return parser


def main() -> None:
    args = build_parser().parse_args()

    df = pd.read_csv(args.input_csv)
    for col in [args.time_col, args.z_col, args.flap_col]:
        if col not in df.columns:
            raise ValueError(f"Missing required column '{col}' in {args.input_csv}")

    results = analyze(
        df=df,
        time_col=args.time_col,
        z_col=args.z_col,
        flap_col=args.flap_col,
        window=args.window,
        min_section_samples=args.min_section_samples,
        min_steady_samples=args.min_steady_samples,
        flap_change_quantile=args.flap_change_quantile,
        slope_quantile=args.slope_quantile,
        std_quantile=args.std_quantile,
    )

    rows = []
    for sec_id, sec in enumerate(results, start=1):
        for seg_id, seg in enumerate(sec.steady_segments, start=1):
            rows.append(
                {
                    "section_id": sec_id,
                    "steady_id": seg_id,
                    "section_start_idx": sec.section_start,
                    "section_end_idx": sec.section_end,
                    "steady_start_idx": seg.start_idx,
                    "steady_end_idx": seg.end_idx,
                    "steady_start_time": df[args.time_col].iloc[seg.start_idx],
                    "steady_end_time": df[args.time_col].iloc[seg.end_idx - 1],
                    "steady_mean": seg.mean_value,
                }
            )

    out_df = pd.DataFrame(rows)
    out_df.to_csv(args.output_csv, index=False)

    plot_results(
        df=df,
        results=results,
        time_col=args.time_col,
        z_col=args.z_col,
        flap_col=args.flap_col,
        output_path=args.output_plot,
    )

    print(f"Detected {len(results)} flap sections.")
    print(f"Detected {len(out_df)} steady-state segments in total.")
    print(f"Saved table to: {args.output_csv}")
    print(f"Saved plot to: {args.output_plot}")


if __name__ == "__main__":
    main()
