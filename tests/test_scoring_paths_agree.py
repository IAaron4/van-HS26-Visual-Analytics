"""The SQL path and the pandas path must flag the same cells.

The app scores in DuckDB for speed, the notebooks score in pandas for convenience. Two
implementations of one formula drift apart unless something checks, so this checks.
"""

from __future__ import annotations

import sys
from pathlib import Path

import duckdb
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import model  # noqa: E402

PROCESSED = Path(__file__).resolve().parents[1] / "data" / "processed"
CELLS = PROCESSED / "cells.parquet"

pytestmark = pytest.mark.skipif(not CELLS.exists(), reason="run prepare.py first")


@pytest.fixture(scope="module")
def con() -> duckdb.DuckDBPyConnection:
    connection = duckdb.connect()
    connection.execute("SET threads = 4")
    connection.execute(
        f"""
        CREATE TABLE slice AS
        SELECT line, bpuic, station, category, hour,
               median_delay_s, median_gain_s, median_dwell_s,
               n_trips, n_measured, n_cancelled,
               base8_delay_s AS base_delay_s,
               base8_iqr_s   AS base_iqr_s,
               base8_n       AS base_n
        FROM read_parquet('{CELLS}')
        WHERE hour >= DATE '2026-02-01' AND hour < DATE '2026-02-15'
        """
    )
    return connection


@pytest.mark.parametrize(
    "params",
    [
        model.Params(),
        model.Params(z_threshold=5.0, min_excess_s=60.0),
        model.Params(cancel_weight=8.0),
        model.Params(min_measured=4, min_share_real=0.8),
    ],
    ids=["default", "strict-z", "cancel-weighted", "strict-evidence"],
)
def test_same_cells_flagged(con: duckdb.DuckDBPyConnection, params: model.Params) -> None:
    from_sql = con.execute(model.flagged_sql("slice", params)).df()

    everything = con.execute("SELECT * FROM slice").df()
    everything = everything.rename(
        columns={"base_delay_s": "baseline_delay_s", "base_iqr_s": "baseline_iqr_s"}
    )
    everything["baseline_n"] = con.execute("SELECT base_n FROM slice").df()["base_n"]
    enough = everything["baseline_n"] >= params.min_baseline_obs
    everything["baseline_delay_s"] = everything["baseline_delay_s"].where(enough)
    everything["baseline_iqr_s"] = everything["baseline_iqr_s"].where(enough)

    prepped = model.prepare_cells(everything)
    scored = model.score_cells(prepped, params)
    from_pandas = scored[model.worth_reporting(scored, params)]

    sql_keys = set(zip(from_sql["line"], from_sql["bpuic"], from_sql["hour"], strict=True))
    pandas_keys = set(
        zip(from_pandas["line"], from_pandas["bpuic"], from_pandas["hour"], strict=True)
    )

    assert sql_keys == pandas_keys, (
        f"{len(sql_keys - pandas_keys)} only in SQL, "
        f"{len(pandas_keys - sql_keys)} only in pandas"
    )
    assert len(sql_keys) > 0, "nothing flagged, the test proves nothing"
