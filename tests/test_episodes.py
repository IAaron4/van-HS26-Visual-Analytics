"""Unit tests for episode grouping and summarising, on small hand-built frames."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import model  # noqa: E402


def cell(line="S1", bpuic=8500001, station="A", hour_index=0, score=10.0,
         gain_s=60.0, n_trips=10, n_measured=10, n_cancelled=0, delay_s=600.0,
         excess_s=None):
    """One flagged cell, with everything group_episodes reads."""
    return {
        "line": line,
        "bpuic": bpuic,
        "station": station,
        "hour": pd.Timestamp("2026-02-16") + pd.Timedelta(hours=hour_index),
        "hour_index": hour_index,
        "median_delay_s": delay_s,
        "median_gain_s": gain_s,
        "excess_s": delay_s if excess_s is None else excess_s,
        "score": score,
        "n_trips": n_trips,
        "n_measured": n_measured,
        "n_cancelled": n_cancelled,
        "share_real": n_measured / n_trips,
        "cancel_rate": n_cancelled / n_trips,
    }


def order(entries: list[tuple[str, int, int]]) -> pd.DataFrame:
    """Stop order frame from (line, bpuic, position) triples."""
    return pd.DataFrame(
        [{"line": ln, "bpuic": b, "station": f"stn{p}", "position": p} for ln, b, p in entries]
    )


def test_empty_input_returns_empty_frame() -> None:
    got = model.group_episodes(pd.DataFrame(columns=["line"]), order([]))
    assert got.empty
    assert list(got.columns) == model._EPISODE_COLUMNS


def test_two_cells_same_station_consecutive_hours_merge() -> None:
    flagged = pd.DataFrame([cell(hour_index=5), cell(hour_index=6)])
    got = model.group_episodes(flagged, order([("S1", 8500001, 1)]))
    assert len(got) == 1
    assert got.loc[0, "n_cells"] == 2


def test_two_cells_adjacent_stations_same_hour_merge() -> None:
    flagged = pd.DataFrame([
        cell(bpuic=8500001, hour_index=5),
        cell(bpuic=8500002, hour_index=5),
    ])
    got = model.group_episodes(
        flagged, order([("S1", 8500001, 1), ("S1", 8500002, 2)])
    )
    assert len(got) == 1
    assert got.loc[0, "n_stations"] == 2


def test_distant_cells_stay_separate() -> None:
    flagged = pd.DataFrame([
        cell(bpuic=8500001, hour_index=5),
        cell(bpuic=8500009, hour_index=20),
    ])
    got = model.group_episodes(
        flagged, order([("S1", 8500001, 1), ("S1", 8500009, 9)])
    )
    assert len(got) == 2


def test_different_lines_never_merge() -> None:
    flagged = pd.DataFrame([
        cell(line="S1", bpuic=8500001, hour_index=5),
        cell(line="S2", bpuic=8500002, hour_index=5),
    ])
    got = model.group_episodes(
        flagged, order([("S1", 8500001, 1), ("S2", 8500002, 2)])
    )
    assert len(got) == 2


@pytest.mark.parametrize("hour_offset", [1, -1], ids=["later", "earlier"])
def test_diagonal_neighbours_merge_in_both_directions(hour_offset: int) -> None:
    """A line runs both ways, so the next station can be an hour earlier or an hour later.

    Connecting only the later diagonal splits one incident into two whenever the direction
    happens to run against the position axis.
    """
    flagged = pd.DataFrame([
        cell(bpuic=8500001, hour_index=10),
        cell(bpuic=8500002, hour_index=10 + hour_offset),
    ])
    got = model.group_episodes(
        flagged, order([("S1", 8500001, 1), ("S1", 8500002, 2)])
    )
    assert len(got) == 1, f"diagonal with offset {hour_offset} did not merge"


def test_evidence_is_weighted_by_service() -> None:
    """A cell with two trips must not count as much as one with fifty.

    share_real and cancel_rate describe the episode as a whole, so they have to be totals over
    totals. An unweighted mean of per-cell rates lets a tiny cell drag the number around.
    """
    flagged = pd.DataFrame([
        cell(bpuic=8500001, hour_index=5, n_trips=2, n_measured=2, n_cancelled=0),
        cell(bpuic=8500002, hour_index=5, n_trips=50, n_measured=25, n_cancelled=25),
    ])
    got = model.group_episodes(
        flagged, order([("S1", 8500001, 1), ("S1", 8500002, 2)])
    )
    assert len(got) == 1
    assert got.loc[0, "share_real"] == pytest.approx(27 / 52, abs=1e-6)
    assert got.loc[0, "cancel_rate"] == pytest.approx(25 / 52, abs=1e-6)


def test_origin_is_the_cell_that_added_most_delay() -> None:
    flagged = pd.DataFrame([
        cell(bpuic=8500001, station="start", hour_index=5, gain_s=600.0, delay_s=600.0),
        cell(bpuic=8500002, station="downstream", hour_index=5, gain_s=10.0, delay_s=900.0),
    ])
    got = model.group_episodes(
        flagged, order([("S1", 8500001, 1), ("S1", 8500002, 2)])
    )
    assert got.loc[0, "origin_station"] == "start"
    assert got.loc[0, "worst_delay_s"] == 900.0


def test_episode_with_no_gain_data_still_summarises() -> None:
    flagged = pd.DataFrame([cell(gain_s=float("nan"))])
    got = model.group_episodes(flagged, order([("S1", 8500001, 1)]))
    assert len(got) == 1
    assert pd.isna(got.loc[0, "origin_gain_s"])


def test_station_missing_from_stop_order_does_not_merge_everything() -> None:
    """Unknown stations get a sentinel position. Two of them must not become neighbours."""
    flagged = pd.DataFrame([
        cell(bpuic=8599001, hour_index=5),
        cell(bpuic=8599002, hour_index=5),
    ])
    got = model.group_episodes(flagged, order([("S1", 8500001, 1)]))
    assert len(got) == 2, "cells with unknown positions were merged into one episode"
