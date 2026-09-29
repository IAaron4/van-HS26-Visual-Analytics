"""First look at the data, and the figures I want for the Data Viz Clinic.

Run after prepare.py:
    python notebooks/01_explore.py --month 2026-02

Figure titles stay descriptive. The argument belongs in the report caption, not on the axes.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import duckdb
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import style  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
FIGURES = ROOT / "figures" / "before"


def fig_delay_by_status(con: duckdb.DuckDBPyConnection, stops: str) -> None:
    """Delay distribution split by whether the recorded time was measured or forecast."""
    rows = con.execute(
        f"""
        SELECT CASE WHEN arr_measured THEN 'measured (REAL)' ELSE 'forecast or estimate' END
                   AS source,
               arr_delay_s / 60.0 AS delay_min
        FROM read_parquet('{stops}')
        WHERE arr_delay_s IS NOT NULL AND arr_delay_s BETWEEN -300 AND 1800
        USING SAMPLE 400000 ROWS
        """
    ).df()

    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    bins = np.linspace(-5, 30, 71)
    for colour, source in zip(style.SERIES, ["measured (REAL)", "forecast or estimate"], strict=False):
        part = rows.loc[rows["source"] == source, "delay_min"]
        ax.hist(part, bins=bins, histtype="step", density=True, linewidth=1.6,
                color=colour, label=f"{source}  n={len(part):,}")

    ax.set_xlabel("arrival delay (minutes)")
    ax.set_ylabel("density")
    ax.set_title("Arrival delay by source of the recorded time")
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(FIGURES / "01_delay_by_status.png")
    plt.close(fig)
    print("  01_delay_by_status.png")


def fig_daily_cancellation(con: duckdb.DuckDBPyConnection, stops: str) -> None:
    """Daily median delay against cancellation rate, split by weekday.

    Pooled, the fit slopes down, which looks like support for the survivorship story. Split, both
    groups slope up. The figure exists to show that the naive reading does not hold.
    """
    daily = con.execute(
        f"""
        SELECT dayofweek(service_day) IN (0, 6)                                   AS weekend,
               median(arr_delay_s) FILTER (arr_measured AND NOT cancelled) / 60.0 AS delay_min,
               100.0 * count(*) FILTER (cancelled) / count(*)                     AS cancel_pct
        FROM read_parquet('{stops}')
        GROUP BY service_day HAVING count(*) > 10000
        """
    ).df()

    fig, ax = plt.subplots(figsize=(5.6, 3.6))
    for colour, (is_weekend, label) in zip(
        style.SERIES, [(False, "Mon to Fri"), (True, "Sat and Sun")], strict=False
    ):
        part = daily[daily["weekend"] == is_weekend]
        ax.scatter(part["cancel_pct"], part["delay_min"], s=34, color=colour,
                   edgecolor=style.SURFACE, linewidth=1.2, zorder=3,
                   label=f"{label}  n={len(part)}")
        slope, intercept = np.polyfit(part["cancel_pct"], part["delay_min"], 1)
        xs = np.linspace(part["cancel_pct"].min(), part["cancel_pct"].max(), 10)
        ax.plot(xs, slope * xs + intercept, color=colour, linewidth=1.1, alpha=0.75, zorder=2)

    pooled = np.polyfit(daily["cancel_pct"], daily["delay_min"], 1)[0]
    ax.set_xlabel("stops cancelled (%)")
    ax.set_ylabel("median delay of the trains that ran (min)")
    ax.set_title("Daily median delay against cancellation rate")
    ax.legend(loc="lower left")
    style.caption(fig, f"Pooled slope {pooled:+.3f} min per percent cancelled; both groups slope up.")
    fig.tight_layout()
    fig.savefig(FIGURES / "02_daily_cancellation.png", bbox_inches="tight")
    plt.close(fig)
    print("  02_daily_cancellation.png")


def fig_incident_profile(con: duckdb.DuckDBPyConnection, stops: str, line: str, day) -> None:
    """One line-day hour by hour, as three panels on a shared time axis.

    Three panels rather than two y-scales on one plot: the alignment of two scales would be
    arbitrary and would suggest a relationship the chart has not measured.
    """
    hourly = con.execute(
        f"""
        SELECT date_part('hour', arr_sched) AS hour,
               median(arr_delay_s) FILTER (arr_measured AND NOT cancelled) / 60.0 AS delay_min,
               100.0 * count(*) FILTER (cancelled) / count(*)                     AS cancel_pct,
               count(*) FILTER (arr_measured AND NOT cancelled)                   AS measured
        FROM read_parquet('{stops}')
        WHERE line = ? AND service_day = ? AND arr_sched IS NOT NULL
        GROUP BY hour HAVING count(*) >= 5 ORDER BY hour
        """,
        [line, day],
    ).df()

    fig, axes = plt.subplots(3, 1, figsize=(6.4, 5.0), sharex=True,
                             gridspec_kw={"height_ratios": [2, 1.4, 1.4], "hspace": 0.18})
    top, mid, bot = axes

    top.axhline(0, color=style.INK_SOFT, linewidth=0.7, alpha=0.45, zorder=1)
    top.plot(hourly["hour"], hourly["delay_min"], "-o", color=style.SERIES[0],
             markeredgecolor=style.SURFACE, markeredgewidth=1.0, zorder=3)
    top.set_ylabel("median delay (min)")
    top.set_title(f"{line} on {day}, by hour")

    mid.fill_between(hourly["hour"], hourly["cancel_pct"], color=style.SERIES[1], alpha=0.45,
                     linewidth=0)
    mid.set_ylabel("cancelled (%)")
    mid.set_ylim(0, 100)

    bot.fill_between(hourly["hour"], hourly["measured"], color=style.INK_SOFT, alpha=0.22,
                     linewidth=0)
    bot.set_ylabel("stops measured")
    bot.set_xlabel("hour of day")
    bot.set_xlim(hourly["hour"].min(), hourly["hour"].max())

    for ax in axes:
        ax.grid(axis="x", visible=False)

    fig.savefig(FIGURES / "03_incident_profile.png", bbox_inches="tight")
    plt.close(fig)
    print("  03_incident_profile.png")


def worst_line_day(con: duckdb.DuckDBPyConnection, cube: str, month: str) -> tuple[str, object]:
    """The worst line-day within the month being plotted.

    The cube covers a whole year, so without the month restriction this returned an incident
    from a different month and the detail figures were drawn on an empty selection.
    """
    found = con.execute(
        f"""
        SELECT line, hour::DATE AS day
        FROM read_parquet('{cube}')
        WHERE line IS NOT NULL AND n_trips >= 3 AND strftime(hour, '%Y-%m') = ?
        GROUP BY line, day HAVING count(*) > 40
        ORDER BY quantile_cont(median_delay_s, 0.9) DESC LIMIT 1
        """,
        [month],
    ).fetchone()
    if found is None:
        raise SystemExit(f"no line-day with enough data in {month}")
    return found


def fig_line_heatmap(con: duckdb.DuckDBPyConnection, cube: str, order: str, line: str, day) -> None:
    """Stations along one line against time, coloured by median delay.

    Delay is a magnitude, so the ramp is one hue light to dark. Hours with no service are left
    in the no-data grey rather than coloured, because on this line a blank hour is often a
    suspension, and rendering it as a low value would say the opposite of what happened.
    """
    grid = con.execute(
        f"""
        SELECT c.station, o.position, date_part('hour', c.hour) AS hour,
               c.median_delay_s / 60.0 AS delay_min
        FROM read_parquet('{cube}') c
        JOIN read_parquet('{order}') o USING (line, bpuic)
        WHERE c.line = ? AND c.hour::DATE = ?
        ORDER BY o.position, hour
        """,
        [line, day],
    ).df()

    pivot = grid.pivot_table(index=["position", "station"], columns="hour",
                             values="delay_min", aggfunc="first").sort_index()
    pivot = pivot.reindex(columns=range(24))

    fig, ax = plt.subplots(figsize=(7.6, max(3.2, 0.19 * len(pivot))))
    im = ax.imshow(np.ma.masked_invalid(pivot.to_numpy()), aspect="auto", cmap=style.DELAY_CMAP,
                   vmin=0, vmax=15, extent=(-0.5, 23.5, len(pivot) - 0.5, -0.5),
                   interpolation="nearest")
    ax.set_yticks(range(len(pivot)))
    ax.set_yticklabels([s for _, s in pivot.index], fontsize=6.5)
    ax.set_xticks(range(0, 24, 3))
    ax.set_xlabel("hour of day")
    ax.set_title(f"{line} on {day}, median delay along the route")
    ax.grid(visible=False)
    ax.tick_params(length=0)

    bar = fig.colorbar(im, ax=ax, pad=0.02, fraction=0.03)
    bar.set_label("median delay (min)")
    bar.outline.set_visible(False)
    style.caption(fig, "Grey means no arrivals recorded in that hour, not an on-time service.")
    fig.savefig(FIGURES / "04_line_heatmap.png", bbox_inches="tight")
    plt.close(fig)
    print("  04_line_heatmap.png")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--month", default="2026-02")
    args = ap.parse_args()

    style.use()
    FIGURES.mkdir(parents=True, exist_ok=True)
    stops = str(PROCESSED / "stops" / args.month / "*.parquet")
    cube = str(PROCESSED / "cube.parquet")
    order = str(PROCESSED / "stop_order.parquet")

    if not list((PROCESSED / "stops" / args.month).glob("*.parquet")):
        print(f"no parquet for {args.month}, run prepare.py first", file=sys.stderr)
        return 1
    if not (Path(cube).exists() and Path(order).exists()):
        print("cube or stop order missing, run prepare.py first", file=sys.stderr)
        return 1

    con = duckdb.connect()
    con.execute("SET threads = 6")
    con.execute("SET memory_limit = '6GB'")

    total, days = con.execute(
        f"SELECT count(*), count(DISTINCT service_day) FROM read_parquet('{stops}')"
    ).fetchone()
    line, day = worst_line_day(con, cube, args.month)
    print(f"{total:,} train stops over {days} days")
    print(f"worst line-day: {line} on {day}\n")

    fig_delay_by_status(con, stops)
    fig_daily_cancellation(con, stops)
    fig_incident_profile(con, stops, line, day)
    fig_line_heatmap(con, cube, order, line, day)
    return 0


if __name__ == "__main__":
    sys.exit(main())
