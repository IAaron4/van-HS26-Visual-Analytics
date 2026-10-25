"""Baseline, anomaly scoring and episode grouping.

No interface code here, so the notebooks, the tests and the app all call the same functions.

These operate on the hourly cube for an analysis window plus the trailing weeks the baseline
needs, not on the whole year. The app never asks for more than that.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

# Scales the interquartile range to a standard deviation on normal data.
IQR_TO_SIGMA = 1.349

# Smallest spread a slot is allowed to claim, in seconds. A slot that always runs to within a
# few seconds would otherwise divide by almost nothing and turn a two minute delay into a
# z-score of 50, which buries real disruptions under statistically unusual trivia.
SPREAD_FLOOR_S = 60.0

# Window lengths precomputed by prepare.build_cells.
BASELINE_WEEKS = (4, 8, 12)

# Offsets from a cell to the neighbours it can share an episode with, as (position, hour). Only
# forward in position, because the reverse link is found when that neighbour is visited. Both
# time directions, because a line runs both ways: the next station along can be served an hour
# earlier or an hour later depending on which way the train is going.
NEIGHBOUR_OFFSETS = ((0, 1), (1, -1), (1, 0), (1, 1))

# Stations missing from the stop order are parked here, spaced two apart so that two unknowns
# are never adjacent to each other or to a real position.
UNKNOWN_POSITION_BASE = -1_000_000

# How episodes can be ordered, and the column each one sorts on. Severity is the default because
# it is the only one that answers "how much harm did this do" rather than "how extreme was one
# moment of it".
RANKINGS = {
    "severity": "severity_s",
    "peak score": "peak_score",
    "reach": "reach",
}


@dataclass(frozen=True)
class Params:
    """The settings a user steers in the interface."""

    z_threshold: float = 3.0
    min_excess_s: float = 180.0
    # Counted in measured arrivals, not scheduled trips. An hourly rural line never has five
    # trips in an hour, so a floor set for an S-Bahn hides every rural line in the country, and
    # those are the ones where a disruption leaves people with no alternative.
    min_measured: int = 2
    min_share_real: float = 0.5
    cancel_weight: float = 0.0
    baseline_weeks: int = 8
    min_baseline_obs: int = 3
    # How much waiting a cancelled train costs, counted in headways. One headway is the natural
    # answer: the passenger waits for the next one. It is a judgement, so it is a control.
    cancel_headway_multiple: float = 1.0
    rank_by: str = "severity"

    def __post_init__(self) -> None:
        if self.baseline_weeks not in BASELINE_WEEKS:
            raise ValueError(f"baseline_weeks must be one of {BASELINE_WEEKS}")
        if self.rank_by not in RANKINGS:
            raise ValueError(f"rank_by must be one of {tuple(RANKINGS)}")


def prepare_cells(cube: pd.DataFrame) -> pd.DataFrame:
    """Add the time keys and the two evidence measures used everywhere downstream."""
    out = cube.copy()
    out["hour"] = pd.to_datetime(out["hour"])
    out["hour_of_week"] = out["hour"].dt.dayofweek * 24 + out["hour"].dt.hour
    # Divide by a Timedelta rather than the raw int64. DuckDB hands back microsecond timestamps,
    # so assuming nanoseconds here silently put six hours of cells in the same slot.
    out["hour_index"] = (out["hour"] - pd.Timestamp("1970-01-01")) // pd.Timedelta(hours=1)
    out["share_real"] = np.where(out["n_trips"] > 0, out["n_measured"] / out["n_trips"], 0.0)
    out["cancel_rate"] = np.where(out["n_trips"] > 0, out["n_cancelled"] / out["n_trips"], 0.0)
    return out


def add_baseline(cells: pd.DataFrame, params: Params = Params()) -> pd.DataFrame:
    """Select the precomputed baseline for the chosen window length.

    Normal is defined per line, station and hour-of-week over the preceding weeks only, so a
    trailing window follows timetable changes on its own and one catastrophic day cannot poison
    the baseline it is meant to be judged against. The windows are computed in prepare.py,
    because doing it here would take minutes on a slice the app has to redraw in under a second.

    A slot with fewer than min_baseline_obs past observations gets no baseline rather than a
    guess, which is what keeps a brand new timetable slot from scoring as an anomaly.
    """
    prefix = f"base{params.baseline_weeks}"
    out = cells.copy()
    enough = out[f"{prefix}_n"] >= params.min_baseline_obs
    out["baseline_delay_s"] = out[f"{prefix}_delay_s"].where(enough)
    out["baseline_iqr_s"] = out[f"{prefix}_iqr_s"].where(enough)
    out["baseline_n"] = out[f"{prefix}_n"]
    return out


def add_baseline_columns(cells: pd.DataFrame, params: Params = Params()) -> pd.DataFrame:
    """Baseline for a frame that already carries one window as base_delay_s and friends.

    The app picks a window when it loads a scenario, so by the time a frame reaches here the
    week suffix is gone. Same rule as add_baseline: too few past observations means no baseline
    rather than a guess.
    """
    out = prepare_cells(cells)
    enough = out["base_n"] >= params.min_baseline_obs
    out["baseline_delay_s"] = out["base_delay_s"].where(enough)
    out["baseline_iqr_s"] = out["base_iqr_s"].where(enough)
    out["baseline_n"] = out["base_n"]
    return out


def score_cells(cells: pd.DataFrame, params: Params = Params()) -> pd.DataFrame:
    """Robust z-score against the baseline, plus the cancellation term.

    Cancelled trips carry no arrival time, so they drop out of the delay median entirely. A line
    that starts cancelling therefore looks like it is recovering. `cancel_weight` lets the user
    decide how much that missing service counts, which is a judgement rather than a constant.
    """
    out = cells.copy()
    spread = (out["baseline_iqr_s"] / IQR_TO_SIGMA).clip(lower=SPREAD_FLOOR_S)

    out["excess_s"] = out["median_delay_s"] - out["baseline_delay_s"]
    out["z_delay"] = out["excess_s"] / spread
    out["z_cancel"] = out["cancel_rate"] * params.cancel_weight
    out["score"] = out["z_delay"].fillna(0) + out["z_cancel"]
    return out


def well_evidenced(cells: pd.DataFrame, params: Params = Params()) -> pd.Series:
    """Whether a cell clears the user's evidence floors.

    Two separate things: how many arrivals were actually measured, and what share of the
    scheduled service that represents. A cell with three trips all measured is better evidence
    than one with eight trips of which three are forecasts.
    """
    return (cells["n_measured"] >= params.min_measured) & (
        cells["share_real"] >= params.min_share_real
    )


def worth_reporting(cells: pd.DataFrame, params: Params = Params()) -> pd.Series:
    """Whether a cell is both statistically unusual and operationally large enough to matter.

    The two gates do different jobs and both are needed. Without the z-score a busy station that
    always runs two minutes down would be flagged every morning. Without the absolute floor, a
    slot that is normally exact turns a ninety second delay into a large z-score, which is true
    and useless. Nobody sends an inspector for ninety seconds.
    """
    return (
        (cells["score"] >= params.z_threshold)
        & (cells["excess_s"] >= params.min_excess_s)
        & well_evidenced(cells, params)
    )


def stop_positions(stop_order: pd.DataFrame) -> dict[tuple[str, int], int]:
    """Map (line, bpuic) to its position along the line."""
    return {
        (row.line, row.bpuic): int(row.position) for row in stop_order.itertuples(index=False)
    }


class _Union:
    """Union-find over cell row positions."""

    def __init__(self, n: int) -> None:
        self.parent = list(range(n))

    def find(self, a: int) -> int:
        while self.parent[a] != a:
            self.parent[a] = self.parent[self.parent[a]]
            a = self.parent[a]
        return a

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra


def flagged_sql(table: str, params: Params) -> str:
    """SQL that scores every cell and returns only the ones worth reporting.

    The formula lives here rather than in the app so the notebooks and the interface cannot
    drift apart. It runs in DuckDB rather than pandas because a scenario is five million cells
    and only about twenty thousand survive the gates. Scoring all of them in pandas, which means
    copying the frame, was most of a four second rebuild.

    The table is expected to carry base_delay_s, base_iqr_s and base_n, already renamed from
    whichever window the user picked.
    """
    return f"""
    WITH scored AS (
        SELECT *,
            median_delay_s - base_delay_s AS excess_s,
            (median_delay_s - base_delay_s)
                / greatest(base_iqr_s / {IQR_TO_SIGMA}, {SPREAD_FLOOR_S}) AS z_delay,
            (n_cancelled::DOUBLE / nullif(n_trips, 0)) * {params.cancel_weight} AS z_cancel
        FROM {table}
        WHERE base_n >= {params.min_baseline_obs}
    )
    SELECT *, coalesce(z_delay, 0) + z_cancel AS score
    FROM scored
    WHERE coalesce(z_delay, 0) + z_cancel >= {params.z_threshold}
      AND excess_s >= {params.min_excess_s}
      AND n_measured >= {params.min_measured}
      AND n_measured::DOUBLE / nullif(n_trips, 0) >= {params.min_share_real}
    """


def find_episodes(
    scored: pd.DataFrame,
    stop_order: pd.DataFrame,
    params: Params = Params(),
) -> pd.DataFrame:
    """Gate the scored cells, then group what survives. The pandas path, used by notebooks."""
    return group_episodes(scored[worth_reporting(scored, params)], stop_order, params)


def cell_severity(cells: pd.DataFrame, params: Params = Params()) -> pd.Series:
    """Train-seconds of delay a cell is responsible for, cancellations included.

    One quantity rather than a weighted blend of unrelated numbers, so it can be read aloud:
    the delay actually carried by the trains that ran, plus the waiting caused by the ones that
    did not. That makes the ranking consider magnitude, duration, spatial extent and service
    volume at once, because every one of them adds cells or adds trains to the sum.

    A cancellation is charged at the headway of that hour, so losing the only train on an hourly
    rural line costs sixty minutes of waiting while losing one of eight on an S-Bahn costs seven
    and a half. Charging a flat penalty instead would have made the ranking a measure of how
    busy a line is.
    """
    trips = cells["n_trips"].replace(0, np.nan)
    headway_s = 3600.0 / trips
    carried = cells["excess_s"].clip(lower=0) * cells["n_measured"]
    waited = headway_s * params.cancel_headway_multiple * cells["n_cancelled"]
    return (carried + waited).fillna(0.0)


def group_episodes(
    flagged: pd.DataFrame,
    stop_order: pd.DataFrame,
    params: Params = Params(),
) -> pd.DataFrame:
    """Merge neighbouring flagged cells into episodes and rank them.

    A single bad cell is not something anyone reports. Cells on the same line that touch in time
    and along the route belong to one incident, which is the unit an analyst writes down.

    Takes cells that have already passed the gates, so the app can do the gating in SQL and hand
    over only the survivors.
    """
    flagged = flagged.reset_index(drop=True).copy()
    if flagged.empty:
        return _empty_episodes()

    flagged["position"] = _resolve_positions(flagged, stop_order)

    union = _Union(len(flagged))
    # A key can hold several cells, because two stations whose median position rounds to the
    # same number share a slot. They belong to the same episode either way.
    index: dict[tuple[str, int, int], list[int]] = {}
    for i, row in enumerate(flagged.itertuples(index=False)):
        index.setdefault((row.line, row.position, row.hour_index), []).append(i)

    for (line, position, hour_index), members in index.items():
        for other in members[1:]:
            union.union(members[0], other)
        for dp, dh in NEIGHBOUR_OFFSETS:
            neighbours = index.get((line, position + dp, hour_index + dh))
            if neighbours:
                union.union(members[0], neighbours[0])

    flagged["severity_s"] = cell_severity(flagged, params)
    flagged["episode"] = [union.find(i) for i in range(len(flagged))]
    return _summarise(flagged, params)


def _resolve_positions(flagged: pd.DataFrame, stop_order: pd.DataFrame) -> list[int]:
    """Position along the line for each cell, with a private slot for unknown stations.

    Giving every unknown station the same sentinel made them all neighbours, so a line's
    unmapped cells collapsed into one episode.
    """
    known = stop_positions(stop_order)
    parked: dict[tuple[str, int], int] = {}
    out = []
    for line, bpuic in zip(flagged["line"], flagged["bpuic"], strict=True):
        position = known.get((line, bpuic))
        if position is None:
            key = (line, bpuic)
            if key not in parked:
                parked[key] = UNKNOWN_POSITION_BASE - 2 * len(parked)
            position = parked[key]
        out.append(position)
    return out


def _summarise(flagged: pd.DataFrame, params: Params = Params()) -> pd.DataFrame:
    """Collapse flagged cells into one row per episode, and name where it started.

    Arrival delay propagates downstream unchanged, so the cells with the worst delay are usually
    just the ones furthest along the route. The cell that added the most delay since its own
    previous stop is the one worth sending someone to look at.
    """
    grouped = flagged.groupby("episode", sort=False)
    episodes = grouped.agg(
        line=("line", "first"),
        start=("hour", "min"),
        end=("hour", "max"),
        n_cells=("line", "size"),
        n_stations=("bpuic", "nunique"),
        peak_score=("score", "max"),
        mean_score=("score", "mean"),
        worst_delay_s=("median_delay_s", "max"),
        n_trips=("n_trips", "sum"),
        n_measured=("n_measured", "sum"),
        n_cancelled=("n_cancelled", "sum"),
        severity_s=("severity_s", "sum"),
    )

    # Where the episode actually ran. A line code covers more than one railway in this data,
    # so naming the episode after the line's busiest route sent a reader to the wrong canton.
    ends = flagged.sort_values("position").groupby("episode")["station"].agg(["first", "last"])
    episodes["from_station"] = ends["first"]
    episodes["to_station"] = ends["last"]

    # Spatial spread times duration. Kept separate from severity so a wide, shallow disruption
    # can still be found by someone looking for one.
    episodes["reach"] = episodes["n_stations"] * (
        (episodes["end"] - episodes["start"]).dt.total_seconds() / 3600 + 1
    )

    # Totals over totals, not the mean of per-cell rates. Averaging the rates let a cell with
    # two trips weigh as much as one with fifty and moved the episode's evidence figures a long
    # way from the truth.
    trips = episodes["n_trips"].replace(0, np.nan)
    episodes["share_real"] = episodes["n_measured"] / trips
    episodes["cancel_rate"] = episodes["n_cancelled"] / trips

    with_gain = flagged[flagged["median_gain_s"].notna()]
    if len(with_gain):
        peak = with_gain.groupby("episode")["median_gain_s"].idxmax()
        origins = (
            flagged.loc[peak, ["episode", "station", "hour", "median_gain_s"]]
            .rename(columns={
                "station": "origin_station",
                "hour": "origin_hour",
                "median_gain_s": "origin_gain_s",
            })
            .set_index("episode")
        )
        episodes = episodes.join(origins)
    else:
        episodes["origin_station"] = None
        episodes["origin_hour"] = pd.NaT
        episodes["origin_gain_s"] = float("nan")

    episodes = episodes.reset_index()
    order = RANKINGS[params.rank_by]
    return episodes.sort_values(order, ascending=False).reset_index(drop=True)[_EPISODE_COLUMNS]


# The member cells are not summarised here. Collecting a station list per episode meant a Python
# call per group, which was most of an eighteen second rebuild. A caller that needs the cells
# filters the scored frame by the episode id instead.
_EPISODE_COLUMNS = [
    "episode", "line", "start", "end", "n_cells", "n_stations", "peak_score", "mean_score",
    "worst_delay_s", "from_station", "to_station", "origin_station", "origin_hour",
    "origin_gain_s",
    "n_trips", "n_measured", "n_cancelled", "share_real", "cancel_rate",
    "severity_s", "reach",
]


def _empty_episodes() -> pd.DataFrame:
    return pd.DataFrame(columns=_EPISODE_COLUMNS)
