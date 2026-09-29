"""Severity is the ranking the analyst sees first, so its behaviour needs pinning down."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import model  # noqa: E402

from test_episodes import cell, order  # noqa: E402


def test_delay_is_charged_per_train_that_ran() -> None:
    """Ten trains five minutes late is ten times one train five minutes late."""
    one = pd.DataFrame([cell(n_trips=1, n_measured=1, excess_s=300.0)])
    ten = pd.DataFrame([cell(n_trips=10, n_measured=10, excess_s=300.0)])
    params = model.Params(cancel_headway_multiple=0.0)
    assert model.cell_severity(ten, params).iloc[0] == pytest.approx(
        10 * model.cell_severity(one, params).iloc[0]
    )


def test_cancellation_is_charged_at_the_headway() -> None:
    """One cancellation on an hourly line costs an hour of waiting."""
    hourly = pd.DataFrame([cell(n_trips=1, n_measured=0, n_cancelled=1, excess_s=0.0)])
    got = model.cell_severity(hourly, model.Params(cancel_headway_multiple=1.0)).iloc[0]
    assert got == pytest.approx(3600.0)


def test_rural_cancellation_costs_more_than_an_urban_one() -> None:
    """The point of using the headway rather than a flat penalty.

    Losing the only train in an hour strands people. Losing one of eight does not, and a flat
    penalty would have said otherwise, making the ranking a measure of how busy a line is.
    """
    rural = pd.DataFrame([cell(n_trips=1, n_measured=0, n_cancelled=1, excess_s=0.0)])
    urban = pd.DataFrame([cell(n_trips=8, n_measured=7, n_cancelled=1, excess_s=0.0)])
    params = model.Params(cancel_headway_multiple=1.0)
    assert (
        model.cell_severity(rural, params).iloc[0]
        > model.cell_severity(urban, params).iloc[0]
    )


def test_cancel_multiple_of_zero_ignores_cancellations() -> None:
    cells = pd.DataFrame([cell(n_trips=4, n_measured=2, n_cancelled=2, excess_s=0.0)])
    assert model.cell_severity(cells, model.Params(cancel_headway_multiple=0.0)).iloc[0] == 0.0


def test_negative_excess_does_not_earn_credit() -> None:
    """A cell running early inside an episode must not subtract from the total."""
    cells = pd.DataFrame([cell(n_trips=4, n_measured=4, excess_s=-600.0)])
    assert model.cell_severity(cells, model.Params(cancel_headway_multiple=0.0)).iloc[0] == 0.0


def test_severity_sums_across_the_episode() -> None:
    flagged = pd.DataFrame([
        cell(bpuic=8500001, hour_index=5, n_trips=2, n_measured=2, excess_s=300.0),
        cell(bpuic=8500002, hour_index=5, n_trips=2, n_measured=2, excess_s=300.0),
    ])
    got = model.group_episodes(
        flagged, order([("S1", 8500001, 1), ("S1", 8500002, 2)]),
        model.Params(cancel_headway_multiple=0.0),
    )
    assert len(got) == 1
    assert got.loc[0, "severity_s"] == pytest.approx(4 * 300.0)


def test_ranking_choice_changes_the_order() -> None:
    """A long shallow episode beats a short sharp one on severity, loses on peak score."""
    wide = [
        cell(line="S1", bpuic=8500000 + i, station=f"w{i}", hour_index=5 + i,
             score=4.0, excess_s=300.0, n_trips=10, n_measured=10)
        for i in range(6)
    ]
    sharp = [cell(line="S2", bpuic=8600001, station="sharp", hour_index=5,
                  score=50.0, excess_s=1200.0, n_trips=1, n_measured=1)]
    positions = [("S1", 8500000 + i, i + 1) for i in range(6)] + [("S2", 8600001, 1)]
    flagged = pd.DataFrame(wide + sharp)

    by_severity = model.group_episodes(
        flagged, order(positions), model.Params(rank_by="severity")
    )
    by_peak = model.group_episodes(
        flagged, order(positions), model.Params(rank_by="peak score")
    )
    assert by_severity.loc[0, "line"] == "S1"
    assert by_peak.loc[0, "line"] == "S2"


def test_unknown_ranking_is_refused() -> None:
    with pytest.raises(ValueError, match="rank_by"):
        model.Params(rank_by="vibes")
