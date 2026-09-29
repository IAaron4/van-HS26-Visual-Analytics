"""Does the scoring find the incident I already know about?

R43 on 16 February 2026 is visible by eye in figure 4. If the ranking does not surface it
without being told where to look, the algorithm is not doing its job.

    python notebooks/02_model_check.py --month 2026-02
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import duckdb
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import model  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"


def load(month: str, baseline_weeks: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The month to analyse plus the trailing weeks the baseline needs."""
    start = (pd.Period(month, freq="M").start_time - pd.Timedelta(weeks=baseline_weeks)).date()
    end = pd.Period(month, freq="M").end_time.date()

    con = duckdb.connect()
    con.execute("SET threads = 6")
    con.execute("SET memory_limit = '8GB'")
    cube = con.execute(
        f"""
        SELECT * FROM read_parquet('{PROCESSED / "cells.parquet"}')
        WHERE hour >= ? AND hour < ? + INTERVAL 1 DAY
        """,
        [start, end],
    ).df()
    order = con.execute(f"SELECT * FROM read_parquet('{PROCESSED / 'stop_order.parquet'}')").df()
    return cube, order


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--month", default="2026-02")
    ap.add_argument("--z", type=float, default=3.0)
    ap.add_argument("--min-excess-min", type=float, default=3.0)
    ap.add_argument("--min-measured", type=int, default=2)
    ap.add_argument("--min-share-real", type=float, default=0.5)
    ap.add_argument("--cancel-weight", type=float, default=0.0)
    ap.add_argument("--rank-by", default="severity", choices=list(model.RANKINGS))
    args = ap.parse_args()

    params = model.Params(
        z_threshold=args.z,
        min_excess_s=args.min_excess_min * 60,
        min_measured=args.min_measured,
        min_share_real=args.min_share_real,
        cancel_weight=args.cancel_weight,
        rank_by=args.rank_by,
    )

    cube, order = load(args.month, params.baseline_weeks)
    print(f"{len(cube):,} cells loaded, {cube['hour'].min()} to {cube['hour'].max()}")

    cells = model.prepare_cells(cube)
    cells = model.add_baseline(cells, params)
    scored = model.score_cells(cells, params)

    usable = scored["baseline_delay_s"].notna().sum()
    print(f"{usable:,} cells have a baseline ({100 * usable / len(scored):.1f}%)")

    in_month = scored[scored["hour"].dt.strftime("%Y-%m") == args.month]
    episodes = model.find_episodes(in_month, order, params)
    print(f"{len(episodes)} episodes at z >= {params.z_threshold}\n")

    if episodes.empty:
        print("nothing flagged, try a lower threshold")
        return 0

    show = episodes.head(12).copy()
    show["start"] = show["start"].dt.strftime("%d %b %H:%M")
    show["worst_delay_min"] = (show["worst_delay_s"] / 60).round(1)
    show["origin_gain_min"] = (show["origin_gain_s"] / 60).round(1)
    show["hours"] = ((episodes["end"] - episodes["start"]).dt.total_seconds() / 3600 + 1).astype(int)
    show["severity_h"] = (show["severity_s"] / 3600).round(1)
    print(
        show[[
            "line", "start", "hours", "n_stations", "severity_h", "peak_score",
            "worst_delay_min", "origin_station", "origin_gain_min", "share_real",
        ]].to_string(index=False, float_format=lambda v: f"{v:.2f}")
    )

    known = episodes[
        (episodes["line"] == "R43") & (episodes["start"].dt.date == pd.Timestamp("2026-02-16").date())
    ]
    print()
    if known.empty:
        print("R43 on 16 Feb was NOT found, the scoring needs work")
    else:
        rank = int(known.index[0]) + 1
        print(f"R43 on 16 Feb found at rank {rank} of {len(episodes)}")
        print(f"  origin {known.iloc[0]['origin_station']} at {known.iloc[0]['origin_hour']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
