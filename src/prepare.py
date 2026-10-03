"""Turn raw Ist-Daten archives into a train-only parquet dataset and an hourly cube.

Each monthly archive holds one file per day at roughly 650 MB uncompressed, so days are
streamed one at a time and the extracted copy is deleted straight after.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path

import duckdb

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"

# Both spellings occur: scheduled times carry no seconds, forecasts do.
TIME_FORMATS = "['%d.%m.%Y %H:%M:%S', '%d.%m.%Y %H:%M']"

# v2 includes foreign stops, whose UIC country code is not 85. They stay in the stop level data
# so an international trip keeps its context, but the analysis is Swiss only: the service point
# list is Swiss, so nothing abroad has coordinates or a baseline worth trusting.
SWISS = "bpuic BETWEEN 8500000 AND 8599999"

STOPS_SQL = f"""
WITH src AS (
    SELECT * FROM read_csv(?, header = true, all_varchar = true)
)
SELECT
    try_strptime(BETRIEBSTAG, '%d.%m.%Y')::DATE                AS service_day,
    FAHRT_BEZEICHNER                                            AS trip_id,
    BETREIBER_ABK                                               AS operator,
    LINIEN_TEXT                                                 AS line,
    TRY_CAST(BPUIC AS BIGINT)                                   AS bpuic,
    HALTESTELLEN_NAME                                           AS station,
    try_strptime(ANKUNFTSZEIT, {TIME_FORMATS})                  AS arr_sched,
    try_strptime(AN_PROGNOSE,  {TIME_FORMATS})                  AS arr_actual,
    nullif(upper(AN_PROGNOSE_STATUS), '')                       AS arr_status,
    try_strptime(ABFAHRTSZEIT, {TIME_FORMATS})                  AS dep_sched,
    try_strptime(AB_PROGNOSE,  {TIME_FORMATS})                  AS dep_actual,
    nullif(upper(AB_PROGNOSE_STATUS), '')                       AS dep_status,
    lower(FAELLT_AUS_TF) = 'true'                               AS cancelled
FROM src
WHERE PRODUKT_ID ILIKE 'Zug'
  AND lower(coalesce(DURCHFAHRT_TF, 'false')) <> 'true'
  AND lower(coalesce(ZUSATZFAHRT_TF, 'false')) <> 'true'
"""

# A trip crossing midnight has its actual time stamped with the operating day while the schedule
# carries the real calendar date, so the difference comes out as almost exactly minus 24 hours.
# It is 0.013% of measured arrivals and nearly all of it sits in the midnight hour, but a twelve
# hour excess beats every real incident in any ranking driven by magnitude, so it has to go.
#
# Rather than repair the date, which would mean guessing what was intended, such a row keeps its
# place in the trip count and loses its measurement. It is then handled by the same machinery
# that already says how much of a cell was actually measured.
#
# The window is asymmetric on purpose. A train cannot arrive an hour early, the timetable has
# nowhere near that much slack, while six hours late is rare but real.
PLAUSIBLE_DELAY_S = (-3600, 21600)

DERIVE_SQL = f"""
SELECT
    * EXCLUDE (raw_arr_delay_s, raw_dep_delay_s),
    CASE WHEN raw_arr_delay_s BETWEEN {PLAUSIBLE_DELAY_S[0]} AND {PLAUSIBLE_DELAY_S[1]}
         THEN raw_arr_delay_s END AS arr_delay_s,
    CASE WHEN raw_dep_delay_s BETWEEN {PLAUSIBLE_DELAY_S[0]} AND {PLAUSIBLE_DELAY_S[1]}
         THEN raw_dep_delay_s END AS dep_delay_s,
    -- An empty status means PROGNOSE, so only an explicit REAL is a measurement, and only if
    -- the number it produced is one a train could actually have made.
    arr_status = 'REAL' AND raw_arr_delay_s
        BETWEEN {PLAUSIBLE_DELAY_S[0]} AND {PLAUSIBLE_DELAY_S[1]} AS arr_measured,
    dep_status = 'REAL' AND raw_dep_delay_s
        BETWEEN {PLAUSIBLE_DELAY_S[0]} AND {PLAUSIBLE_DELAY_S[1]} AS dep_measured
FROM (
    SELECT *,
        date_diff('second', arr_sched, arr_actual) AS raw_arr_delay_s,
        date_diff('second', dep_sched, dep_actual) AS raw_dep_delay_s
    FROM stops
    WHERE service_day IS NOT NULL
)
"""


def day_members(archive: Path) -> list[str]:
    """Names of the daily files inside a monthly archive, in date order."""
    with zipfile.ZipFile(archive) as zf:
        return sorted(n for n in zf.namelist() if not n.endswith("/"))


def convert_month(con: duckdb.DuckDBPyConnection, archive: Path, out_dir: Path) -> int:
    """Write one monthly archive to parquet, one file per day. Returns rows written."""
    out_dir.mkdir(parents=True, exist_ok=True)
    total = 0
    with zipfile.ZipFile(archive) as zf:
        for member in day_members(archive):
            day = Path(member).name.split("_")[0]
            target = out_dir / f"{day}.parquet"
            if target.exists():
                total += con.execute(
                    "SELECT count(*) FROM read_parquet(?)", [str(target)]
                ).fetchone()[0]
                continue

            with tempfile.TemporaryDirectory() as tmp:
                extracted = Path(tmp) / Path(member).name
                with zf.open(member) as src, extracted.open("wb") as dst:
                    shutil.copyfileobj(src, dst, length=1 << 22)

                con.execute("CREATE OR REPLACE TEMP TABLE stops AS " + STOPS_SQL, [str(extracted)])
                con.execute(
                    f"COPY ({DERIVE_SQL}) TO '{target}' (FORMAT parquet, COMPRESSION zstd)"
                )

            rows = con.execute("SELECT count(*) FROM read_parquet(?)", [str(target)]).fetchone()[0]
            total += rows
            print(f"  {day}  {rows:>9,} train stops")
    return total


def build_cube(con: duckdb.DuckDBPyConnection, stops_glob: str, target: Path) -> None:
    """Aggregate stop level rows into (line, station, hour) cells.

    Each cell carries three different measurements rather than one. Level is how late a train is
    on arrival, which propagates downstream unchanged and so cannot say where a delay began.
    Gain is how much was added since the previous stop, which is what localises the cause. Dwell
    is how much was added standing at the platform, which separates a boarding problem from a
    running problem. The three disagree often enough to be worth keeping apart.
    """
    con.execute(
        f"""
        COPY (
            WITH base AS (
                SELECT
                    trip_id, line, bpuic, station, arr_sched, cancelled, arr_measured,
                    CASE WHEN arr_measured AND NOT cancelled THEN arr_delay_s END AS delay_s,
                    CASE WHEN arr_measured AND dep_measured AND NOT cancelled
                         THEN dep_delay_s - arr_delay_s END                       AS dwell_s
                FROM read_parquet('{stops_glob}')
                WHERE arr_sched IS NOT NULL AND line IS NOT NULL AND {SWISS}
            ),
            legs AS (
                SELECT *,
                       delay_s - lag(delay_s) OVER (
                           PARTITION BY trip_id ORDER BY arr_sched
                       ) AS gain_s
                FROM base
            )
            SELECT
                line,
                bpuic,
                any_value(station)                        AS station,
                regexp_extract(line, '^([A-Za-z]+)', 1)   AS category,
                date_trunc('hour', arr_sched)             AS hour,
                median(delay_s)                           AS median_delay_s,
                quantile_cont(delay_s, 0.85)              AS p85_delay_s,
                median(gain_s)                            AS median_gain_s,
                median(dwell_s)                           AS median_dwell_s,
                count(*)                                  AS n_trips,
                count(*) FILTER (arr_measured)            AS n_measured,
                count(*) FILTER (cancelled)               AS n_cancelled,
                count(gain_s)                             AS n_gain
            FROM legs
            GROUP BY line, bpuic, category, hour
        ) TO '{target}' (FORMAT parquet, COMPRESSION zstd)
        """
    )


BASELINE_WINDOWS = (4, 8, 12)


def build_cells(con: duckdb.DuckDBPyConnection, cube: Path, target: Path) -> None:
    """The cube plus a rolling baseline per hour-of-week slot, for three window lengths.

    Precomputed here rather than in the app because a three month slice is over five million
    cells across 435,000 slots, which is far too slow to sit behind a widget. The cost is that
    the window length becomes a choice of three rather than a free slider.

    Spread is the interquartile range, not the MAD. A MAD needs each window's own median to
    measure deviations against, which one window pass cannot express, while two quantiles can.
    Both are robust; this one is computable at the scale the interaction needs.
    """
    measures = ",\n".join(
        f"""
            quantile_cont(median_delay_s, 0.5)  OVER w{n} AS base{n}_delay_s,
            quantile_cont(median_delay_s, 0.75) OVER w{n}
              - quantile_cont(median_delay_s, 0.25) OVER w{n} AS base{n}_iqr_s,
            count(median_delay_s) OVER w{n} AS base{n}_n"""
        for n in BASELINE_WINDOWS
    )
    frames = ",\n".join(
        f"""w{n} AS (PARTITION BY line, bpuic, hour_of_week ORDER BY hour
                     ROWS BETWEEN {n} PRECEDING AND 1 PRECEDING)"""
        for n in BASELINE_WINDOWS
    )
    con.execute(
        f"""
        COPY (
            WITH c AS (
                SELECT *, date_part('dow', hour) * 24 + date_part('hour', hour) AS hour_of_week
                FROM read_parquet('{cube}')
            )
            SELECT *,
{measures}
            FROM c
            WINDOW
{frames}
        ) TO '{target}' (FORMAT parquet, COMPRESSION zstd)
        """
    )


def build_stations(con: duckdb.DuckDBPyConnection, points: Path, target: Path) -> None:
    """Station reference with coordinates, one current row per stop id.

    Service points carry a row per validity period, so keep the most recent per number.
    """
    con.execute(
        f"""
        COPY (
            SELECT
                TRY_CAST(number AS BIGINT)      AS bpuic,
                designationOfficial             AS station,
                TRY_CAST(wgs84North AS DOUBLE)  AS lat,
                TRY_CAST(wgs84East AS DOUBLE)   AS lon,
                cantonName                      AS canton,
                municipalityName                AS municipality
            FROM read_csv(?, header = true, all_varchar = true)
            WHERE number IS NOT NULL
            QUALIFY row_number() OVER (PARTITION BY number ORDER BY validFrom DESC) = 1
        ) TO '{target}' (FORMAT parquet, COMPRESSION zstd)
        """,
        [str(points)],
    )


def build_stop_order(con: duckdb.DuckDBPyConnection, stops_glob: str, target: Path) -> None:
    """Derive where each station sits along its line, so episodes can follow the route.

    A line runs in both directions, and averaging positions over both puts the far end of the
    route in the middle, so the order comes from one origin and destination pair only. Stations
    served by other variants fall back to their overall median position, pushed past the main
    route so they never sit between two stations they do not lie between.

    That pair is the variant serving the most stations, not the one running most often. An
    international line like EC is cut at the border by the Swiss filter, and its most frequent
    Swiss fragment is a single stop at Chiasso, which labelled the line "Chiasso to Chiasso"
    while the real route runs to Zürich.

    Positions are numbered per trip and day. FAHRT_BEZEICHNER repeats daily, so partitioning by
    it alone numbered a year of trips in one sequence. The relative order survived that, since
    every station shifted by the same amount each day, but the numbers meant nothing.
    """
    con.execute(
        f"""
        COPY (
            WITH seq AS (
                SELECT trip_id, service_day, line, bpuic, station,
                       row_number() OVER (
                           PARTITION BY trip_id, service_day
                           ORDER BY coalesce(arr_sched, dep_sched)
                       ) AS pos
                FROM read_parquet('{stops_glob}')
                WHERE line IS NOT NULL
                  AND (arr_sched IS NOT NULL OR dep_sched IS NOT NULL)
                  AND {SWISS}
            ),
            ends AS (
                SELECT trip_id, service_day, line,
                       arg_min(bpuic, pos) AS origin, arg_max(bpuic, pos) AS dest
                FROM seq GROUP BY trip_id, service_day, line
            ),
            variants AS (
                SELECT s.line, e.origin, e.dest,
                       count(DISTINCT s.bpuic) AS stations,
                       count(*)                AS stops
                FROM seq s JOIN ends e USING (trip_id, service_day)
                GROUP BY s.line, e.origin, e.dest
            ),
            dominant AS (
                SELECT line, origin, dest FROM variants
                QUALIFY row_number() OVER (
                    PARTITION BY line ORDER BY stations DESC, stops DESC
                ) = 1
            ),
            on_route AS (
                SELECT s.line, s.bpuic,
                       any_value(s.station)                  AS station,
                       CAST(round(median(s.pos)) AS INTEGER) AS position,
                       count(*)                              AS n_observed
                FROM seq s
                JOIN ends e USING (trip_id, service_day)
                JOIN dominant d ON d.line = s.line AND d.origin = e.origin AND d.dest = e.dest
                GROUP BY s.line, s.bpuic
            ),
            route_end AS (
                SELECT line, max(position) AS last_pos FROM on_route GROUP BY line
            ),
            off_route AS (
                SELECT s.line, s.bpuic,
                       any_value(s.station)                  AS station,
                       CAST(round(median(s.pos)) AS INTEGER) AS raw_position,
                       count(*)                              AS n_observed
                FROM seq s
                WHERE NOT EXISTS (
                    SELECT 1 FROM on_route o WHERE o.line = s.line AND o.bpuic = s.bpuic
                )
                GROUP BY s.line, s.bpuic
            )
            SELECT line, bpuic, station, position, n_observed, true AS on_main_route
            FROM on_route
            UNION ALL
            SELECT f.line, f.bpuic, f.station,
                   coalesce(r.last_pos, 0) + f.raw_position AS position,
                   f.n_observed, false AS on_main_route
            FROM off_route f LEFT JOIN route_end r USING (line)
        ) TO '{target}' (FORMAT parquet, COMPRESSION zstd)
        """
    )


def report(con: duckdb.DuckDBPyConnection, stops_glob: str, points: Path | None) -> None:
    """Print the checks that would otherwise fail silently."""
    print("\narrival status breakdown")
    rows = con.execute(
        f"""
        SELECT coalesce(arr_status, '(empty -> PROGNOSE)') AS status,
               count(*) AS n,
               round(100.0 * count(*) / sum(count(*)) OVER (), 1) AS pct
        FROM read_parquet('{stops_glob}')
        GROUP BY 1 ORDER BY n DESC
        """
    ).fetchall()
    for status, n, pct in rows:
        print(f"  {status:<24} {n:>12,}  {pct:>5.1f}%")

    cancelled, total = con.execute(
        f"SELECT count(*) FILTER (cancelled), count(*) FROM read_parquet('{stops_glob}')"
    ).fetchone()
    print(f"\ncancelled stops            {cancelled:>12,}  {100 * cancelled / total:>5.1f}%")

    rejected = con.execute(
        f"""
        SELECT count(*) FROM read_parquet('{stops_glob}')
        WHERE arr_status = 'REAL' AND NOT arr_measured
        """
    ).fetchone()[0]
    print(f"REAL times outside the plausible window, dropped: {rejected:,}")

    stations = PROCESSED / "stations.parquet"
    if not stations.exists():
        return

    swiss, foreign = con.execute(
        f"""
        SELECT count(*) FILTER (WHERE {SWISS}), count(*) FILTER (WHERE NOT {SWISS})
        FROM (SELECT DISTINCT bpuic FROM read_parquet('{stops_glob}'))
        """
    ).fetchone()
    matched = con.execute(
        f"""
        SELECT count(*) FILTER (WHERE p.lat IS NOT NULL)
        FROM (SELECT DISTINCT bpuic FROM read_parquet('{stops_glob}') WHERE {SWISS}) s
        LEFT JOIN read_parquet('{stations}') p USING (bpuic)
        """
    ).fetchone()[0]
    pct = 100 * matched / swiss if swiss else 0
    print(f"\nSwiss stations             {swiss:>12,}   with coordinates {matched:,} ({pct:.1f}%)")
    print(f"foreign stations excluded  {foreign:>12,}")
    if pct < 99:
        print("  unexpected gap, check whether BPUIC carries a SLOID for some operators")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--months", nargs="*", help="limit to these YYYY-MM, default all downloaded")
    ap.add_argument("--memory-limit", default="12GB")
    ap.add_argument("--threads", type=int, default=8)
    args = ap.parse_args()

    archives = sorted((RAW / "istdaten").glob("ist-daten-v2-*.zip"))
    if args.months:
        archives = [a for a in archives if any(m in a.name for m in args.months)]
    if not archives:
        print("no archives found, run fetch.py first", file=sys.stderr)
        return 1

    con = duckdb.connect()
    con.execute(f"SET memory_limit = '{args.memory_limit}'")
    con.execute(f"SET threads = {args.threads}")
    con.execute(f"SET temp_directory = '{PROCESSED / '.duckdb_tmp'}'")

    for archive in archives:
        month = archive.stem.replace("ist-daten-v2-", "")
        print(f"\n{archive.name}")
        rows = convert_month(con, archive, PROCESSED / "stops" / month)
        print(f"  total {rows:,}")

    points = next(iter(sorted((RAW / "servicepoints").glob("*"))), None)
    if points:
        build_stations(con, points, PROCESSED / "stations.parquet")
    else:
        print("no service point file, skipping coordinates", file=sys.stderr)

    stops_glob = str(PROCESSED / "stops" / "*" / "*.parquet")
    build_cube(con, stops_glob, PROCESSED / "cube.parquet")
    build_cells(con, PROCESSED / "cube.parquet", PROCESSED / "cells.parquet")
    build_stop_order(con, stops_glob, PROCESSED / "stop_order.parquet")
    report(con, stops_glob, points)
    print(f"\ncube written to {PROCESSED / 'cube.parquet'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
