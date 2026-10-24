"""Panel app for finding and confirming disruption episodes.

    panel serve src/app.py --show

Picking an episode drives the three views below it: the route against time, the same route on the
map, and the episode hour by hour.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import duckdb
import holoviews as hv
import pandas as pd
import panel as pn
import param
from bokeh.models import HoverTool
from bokeh.themes import Theme
from holoviews import opts
from holoviews.util.transform import lon_lat_to_easting_northing

import incidents
import model
import propagation as prop
import style

pn.extension("tabulator", sizing_mode="stretch_width")
hv.extension("bokeh")
hv.renderer("bokeh").theme = Theme(json=style.bokeh_theme())

PROCESSED = Path(__file__).resolve().parents[1] / "data" / "processed"
CELLS = PROCESSED / "cells.parquet"
STATIONS = PROCESSED / "stations.parquet"
STOP_ORDER = PROCESSED / "stop_order.parquet"
LEGS = PROCESSED / "legs.parquet"
INCIDENT_LOG = Path(__file__).resolve().parents[1] / "data" / "incident_log.jsonl"


def no_scroll_zoom(plot, element) -> None:
    """Keep the wheel zoom off the scroll wheel.

    Bokeh leaves active_scroll on "auto" and then activates the only scroll tool there is, so
    scrolling over a chart zooms it instead of moving the page. The page is taller than a
    screen, so that has to go. The tool stays in the toolbar, one click away.

    The Bokeh logo goes with it. It is the clearest library default on the page.
    """
    plot.state.toolbar.active_scroll = None
    plot.state.toolbar.logo = None


def condition_for(when: pd.Timestamp) -> str:
    """The bucket a leg was fitted in. Must match CONDITION_SQL in prepare.py."""
    hour, weekday = when.hour, when.dayofweek
    if hour < 5 or hour >= 22:
        return "night"
    if weekday >= 5:
        return "weekend"
    if 6 <= hour <= 8 or 16 <= hour <= 18:
        return "weekday peak"
    return "weekday off peak"

# Two contrasting periods rather than the whole year. Summer turns out to be the disrupted one,
# because most Swiss disruption is planned engineering work rather than weather.
SCENARIOS = {
    "winter 25/26": ("2025-12-01", "2026-02-28"),
    "summer 26": ("2026-06-01", "2026-08-31"),
}

# Only the codes I am sure of. Anything else keeps its raw code rather than being given a name
# I guessed at.
CATEGORY_NAMES = {
    "S": "S-Bahn", "R": "Regio", "RE": "RegioExpress", "IR": "InterRegio",
    "IC": "InterCity", "SN": "S-Bahn night", "EC": "EuroCity", "ICE": "ICE",
    "TGV": "TGV", "NJ": "Nightjet", "TER": "TER", "RB": "RegionalBahn",
}
NAMED_GROUPS = 6

# Fixed widths rather than responsive ones. A Bokeh figure set to stretch takes the width of
# the page rather than of the column it sits in, so every chart overflowed by the width of the
# controls. Pinning them is the only thing that held, and it makes screenshots reproducible.
CONTROLS_W = 250
CONTENT_W = 960
PANEL_W = (235, 400, 305)

# An optional override. Set VAN_TILE_URL to a full {Z}/{X}/{Y} template to use a provider that
# wants a key. It is read from the environment so the key never reaches the repository.

TILE_URL = os.environ.get("VAN_TILE_URL")

# Keyless basemaps, so the map works with no account anywhere. OpenStreetMap is the default
# because it is unmistakably a map: its tiles carry about 47 kB of detail where CARTO's pale
# style carries 2 kB, which is easy to mistake for nothing having loaded. If a provider blocks
# you, switch here rather than hunting for a key.
BASEMAPS = {
    "OpenStreetMap": "OSM",
    "Esri streets": "EsriStreet",
    "Esri terrain": "EsriTerrain",
    "OpenTopoMap": "OpenTopoMap",
    "Carto light (very pale)": "CartoLight",
}

# Roughly the country, in degrees, for the view that shows where a line sits in Switzerland.
SWISS_BOUNDS_DEG = (5.8, 45.7, 10.6, 47.9)

# Smallest span the fitted view is allowed to have, in Web Mercator metres. A short line like
# Ponte Tresa to Lugano covers five kilometres, and fitting the view to it leaves a suburb with
# nothing recognisable in it. Keeping a floor means there is always some geography around it.
MIN_SPAN_M = 90_000.0

# Short headers. Spelling the units into every column pushed the headers past their width,
# so the units are said once in the line above the table instead.
TABLE_TITLES = {
    "line": "line", "route": "route", "start": "start", "hours": "hours",
    "stations": "stops", "severity_h": "harm", "score": "score",
    "worst_min": "worst", "started_at": "started at", "gain_min": "added",
    "measured": "measured", "share": "measured %", "judged": "judged",
}

TABLE_COLUMNS = [
    "line", "route", "start", "hours", "stations", "severity_h", "worst_min", "judged",
]


def _label(code: str, pct: float) -> str:
    """Short enough that seven of them fit one row."""
    return f"{CATEGORY_NAMES.get(code, code)} {pct:.0f}%"


class Explorer(param.Parameterized):
    """Filters narrow what is analysed. Parameters change how it is scored."""

    scenario = param.Selector(objects=list(SCENARIOS), default="winter 25/26")
    categories = param.ListSelector(default=[], objects=[])
    canton = param.Selector(default="all", objects=["all"])

    rank_by = param.Selector(
        objects=list(model.RANKINGS), default="severity", label="order the list by",
        doc="Severity is delay carried by the trains that ran plus waiting caused by the ones "
            "that did not. Peak score is the single worst cell. Reach is how far it spread.",
    )
    z_threshold = param.Number(
        3.0, bounds=(1.0, 15.0), step=0.5, label="how far from normal",
        doc="How many times its usual spread an hour has to be off before it counts. Lower "
            "finds more and flags more ordinary days.",
    )
    min_excess_min = param.Number(
        3.0, bounds=(0.0, 20.0), step=0.5, label="at least this many minutes late",
        doc="On top of the threshold, an hour has to be this far above what the stop "
            "normally runs at. Keeps statistically odd but harmless hours out.",
    )
    min_measured = param.Integer(
        2, bounds=(1, 20), label="trains actually measured",
        doc="How many arrivals must carry a real timestamp rather than an estimate before "
            "an hour is trusted. Thinner hours are faded, not hidden.",
    )
    min_share_real = param.Number(
        0.5, bounds=(0.0, 1.0), step=0.05, label="share of them measured",
        doc="The fraction of arrivals in an hour that were measured rather than forecast.",
    )
    cancel_weight = param.Number(
        0.0, bounds=(0.0, 20.0), step=1.0, label="count cancellations this much",
        doc="A cancelled train contributes no delay, so a suspended line goes quiet instead "
            "of spiking. Raise this to score the silence.",
    )
    cancel_headway_multiple = param.Number(
        1.0, bounds=(0.0, 4.0), step=0.25, label="a cancellation costs this long a wait",
        doc="In multiples of the gap between trains that hour. Losing the only train on an "
            "hourly line costs an hour of waiting, one of eight costs seven minutes.",
    )
    baseline_weeks = param.Selector(
        objects=list(model.BASELINE_WEEKS), default=8, label="weeks of history for normal",
        doc="How far back the trailing window looks when working out what this stop at this "
            "hour of this weekday usually does.",
    )

    colour_by = param.Selector(
        objects={"how late it was": "level", "where it was added": "gain"}, default="level",
        label="colour the route by",
        doc="How late carries downstream unchanged, so it cannot say where a delay began. "
            "Where it was added can.",
    )
    rows_by = param.Selector(
        objects={
            "stations that were flagged": "flagged",
            "every station on the line": "all",
            "cantons": "canton",
        },
        default="flagged", label="rows show",
        doc="Cantons lose the order of the route, which is what the view is for, so they are "
            "there for long lines rather than as the default.",
    )
    map_extent = param.Selector(
        objects={"around the line": "line", "all of Switzerland": "country"}, default="line",
        label="map shows",
    )
    basemap = param.Selector(
        objects=list(BASEMAPS), default="OpenStreetMap", label="background",
    )

    # The model. Every one of these refits it rather than filtering its output.
    pooling = param.Selector(
        objects={"each stop to the next": "leg", "one figure per line": "line",
                 "one figure for the country": "global"},
        default="leg", label="fit the model",
        doc="Per stop pair is the most faithful and the noisiest. Per line is steadier and "
            "blurs the difference between a stretch with slack and one without.",
    )
    min_obs = param.Integer(
        60, bounds=(5, 2000), step=5, label="trains behind each fit",
        doc="A stretch with fewer than this many measured runs borrows its line's figure "
            "rather than fitting its own.",
    )
    conditioned = param.Boolean(
        True, label="fit peak and off peak apart",
        doc="A morning peak stretch of track behaves differently from the same track at "
            "midnight. Off, one figure covers the whole day.",
    )
    coverage = param.Selector(
        objects={"half the days": 0.5, "4 days in 5": 0.8, "9 days in 10": 0.9,
                 "19 days in 20": 0.95},
        default=0.8, label="how often the band should hold",
        doc="Measured against real residuals rather than assumed. A true 19 in 20 needs "
            "about ten times the spread here, not two.",
    )
    start_from_actual = param.Boolean(
        True, label="start from what was measured",
        doc="Off, the forecast starts from the what-if delay below instead, so you can ask "
            "what five minutes at the origin would have done.",
    )
    start_delay_min = param.Number(
        5.0, bounds=(0.0, 30.0), step=0.5, label="what-if start, minutes late",
    )

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        missing = [p.name for p in (CELLS, STATIONS, STOP_ORDER, LEGS) if not p.exists()]
        if missing:
            raise SystemExit(f"missing {', '.join(missing)} in data/processed. Run prepare.py.")

        self._con = duckdb.connect()
        self._con.execute("SET threads = 6")
        self._con.execute("SET memory_limit = '8GB'")

        self._stop_order = self._con.execute(f"SELECT * FROM read_parquet('{STOP_ORDER}')").df()
        self._routes = self._route_labels()
        # 12 MB, so it lives in memory and every refit is arithmetic over it.
        self._legs = self._con.execute(f"SELECT * FROM read_parquet('{LEGS}')").df()
        self._coef_key: tuple | None = None
        self._coef: pd.DataFrame | None = None
        self._verdicts = incidents.latest_verdicts(INCIDENT_LOG)
        self._last_prediction: dict[str, float | None] = {}

        # Everything refresh() touches has to exist first: _build_filters sets `categories`,
        # which fires the watcher.
        self._slice_key: tuple | None = None
        self._result_key: tuple | None = None
        self._result: pd.DataFrame = model._empty_episodes()
        self._shown = pd.DataFrame()
        self._n_cells = 0
        self.load_ms = 0.0
        self.score_ms = 0.0
        self.n_flagged = 0

        self.table = pn.widgets.Tabulator(
            pd.DataFrame(columns=TABLE_COLUMNS), height=200, disabled=True,
            show_index=False, selectable=1, theme="simple",
            pagination="local", page_size=5, titles=TABLE_TITLES,
            # Size to content and stretch the last column. Sharing the width equally cut
            # every header to three letters, and sizing to content alone pushed `judged`,
            # the column saying whether this episode has been decided, off the right edge.
            layout="fit_data_stretch",
        )
        self.table.param.watch(lambda _: self._draw_detail(), "selection")
        self.detail = pn.Column(sizing_mode="stretch_width")

        self._build_filters()

    # ---- reference data -------------------------------------------------------------

    def _route_labels(self) -> dict[str, str]:
        """Where each line runs from and to, taken from its main route.

        A line called R43 tells a reader nothing. Brig to Andermatt does.
        """
        main = self._stop_order[self._stop_order["on_main_route"]].sort_values("position")
        ends = main.groupby("line")["station"].agg(["first", "last"])
        return {line: f"{row['first']} to {row['last']}" for line, row in ends.iterrows()}

    def _build_filters(self) -> None:
        """Category groups and cantons, both derived from the data.

        The raw codes over-split: a line whose name has no digits, like IRVAE, keeps the whole
        name as its code, which is how 38 of them appear. Six cover 95% of the data, so those get
        their own entry and the rest become one option rather than a list nobody can read.
        """
        counts = self._con.execute(
            f"""
            SELECT category, count(*) AS n FROM read_parquet('{CELLS}')
            WHERE category IS NOT NULL AND category <> ''
            GROUP BY category ORDER BY n DESC
            """
        ).df()
        total = counts["n"].sum()

        self._groups: dict[str, list[str]] = {}
        for _, row in counts.head(NAMED_GROUPS).iterrows():
            self._groups[_label(row["category"], 100 * row["n"] / total)] = [row["category"]]
        rest = counts.iloc[NAMED_GROUPS:]
        if len(rest):
            share = 100 * rest["n"].sum() / total
            label = f"{len(rest)} other kinds {share:.0f}%"
            self._groups[label] = rest["category"].tolist()

        self.param.categories.objects = list(self._groups)
        self.categories = list(self._groups)

        cantons = self._con.execute(
            f"SELECT DISTINCT canton FROM read_parquet('{STATIONS}') "
            "WHERE canton IS NOT NULL AND canton <> '' ORDER BY canton"
        ).df()["canton"].tolist()
        self.param.canton.objects = ["all"] + cantons

    @property
    def params(self) -> model.Params:
        return model.Params(
            z_threshold=self.z_threshold,
            min_excess_s=self.min_excess_min * 60,
            min_measured=self.min_measured,
            min_share_real=self.min_share_real,
            cancel_weight=self.cancel_weight,
            cancel_headway_multiple=self.cancel_headway_multiple,
            baseline_weeks=self.baseline_weeks,
            rank_by=self.rank_by,
        )

    # ---- data -----------------------------------------------------------------------

    def _load(self) -> None:
        """Hold the scenario in a DuckDB table, reloaded only when it or the window changes."""
        key = (self.scenario, self.baseline_weeks)
        if self._slice_key == key:
            return

        start, end = SCENARIOS[self.scenario]
        weeks = self.baseline_weeks
        started = time.perf_counter()
        self._con.execute(
            f"""
            CREATE OR REPLACE TABLE slice AS
            SELECT c.line, c.bpuic, c.station, c.category, c.hour,
                   c.median_delay_s, c.median_gain_s, c.median_dwell_s,
                   c.n_trips, c.n_measured, c.n_cancelled,
                   c.base{weeks}_delay_s AS base_delay_s,
                   c.base{weeks}_iqr_s   AS base_iqr_s,
                   c.base{weeks}_n       AS base_n,
                   s.canton, s.lat, s.lon
            FROM read_parquet('{CELLS}') c
            LEFT JOIN read_parquet('{STATIONS}') s USING (bpuic)
            WHERE c.hour >= ?::DATE AND c.hour < ?::DATE + INTERVAL 1 DAY
            """,
            [start, end],
        )
        self._n_cells = self._con.execute("SELECT count(*) FROM slice").fetchone()[0]
        self.load_ms = (time.perf_counter() - started) * 1000
        self._slice_key = key

    def _flagged(self) -> pd.DataFrame:
        """Score in DuckDB and bring back only the cells worth reporting.

        Filter values are interpolated rather than bound. They are safe because param only
        accepts values from the objects lists, and those come from the data itself.
        """
        self._load()
        codes = [c for label in self.categories for c in self._groups.get(label, [])]
        where = ["category IN (" + ", ".join(f"'{c}'" for c in codes) + ")"] if codes else ["false"]
        if self.canton != "all":
            where.append(f"canton = '{self.canton}'")
        table = f"(SELECT * FROM slice WHERE {' AND '.join(where)})"
        return self._con.execute(model.flagged_sql(table, self.params)).df()

    def _episodes(self) -> pd.DataFrame:
        """Episodes for the current settings, computed once per change."""
        key = (
            self.scenario, tuple(self.categories), self.canton,
            self.z_threshold, self.min_excess_min, self.min_measured,
            self.min_share_real, self.cancel_weight, self.cancel_headway_multiple,
            self.baseline_weeks, self.rank_by,
        )
        if self._result_key == key:
            return self._result

        started = time.perf_counter()
        flagged = model.prepare_cells(self._flagged())
        self._result = model.group_episodes(flagged, self._stop_order, self.params)
        self.score_ms = (time.perf_counter() - started) * 1000
        self.n_flagged = len(flagged)
        self._result_key = key
        return self._result

    # ---- views ----------------------------------------------------------------------

    @param.depends(
        "scenario", "categories", "canton", "rank_by", "z_threshold", "min_excess_min",
        "min_measured", "min_share_real", "cancel_weight", "cancel_headway_multiple",
        "baseline_weeks", watch=True,
    )
    def refresh(self) -> None:
        found = self._episodes()
        if found.empty:
            self._shown = pd.DataFrame()
            self.table.value = pd.DataFrame(columns=TABLE_COLUMNS)
            self.detail[:] = [pn.pane.Markdown("Nothing flagged. Try a lower threshold.")]
            return

        self._shown = found.head(60).assign(
            route=lambda d: d["line"].map(self._routes).fillna(""),
            day=lambda d: d["start"].dt.date,
            peak_hour=lambda d: d["start"],
            hours=lambda d: ((d["end"] - d["start"]).dt.total_seconds() / 3600 + 1).astype(int),
            stations=lambda d: d["n_stations"],
            severity_h=lambda d: (d["severity_s"] / 3600).round(1),
            score=lambda d: d["peak_score"].round(1),
            worst_min=lambda d: (d["worst_delay_s"] / 60).round(1),
            started_at=lambda d: d["origin_station"],
            gain_min=lambda d: (d["origin_gain_s"] / 60).round(1),
            measured=lambda d: d["n_measured"],
            share=lambda d: d["share_real"].round(2),
            start=lambda d: d["start"].dt.strftime("%d %b %H:%M"),
        )
        self._shown["judged"] = [
            self._verdicts.get(incidents.key_of(r.line, str(r.day), str(r.peak_hour)), "")
            for r in self._shown.itertuples(index=False)
        ]
        self.table.value = self._shown[TABLE_COLUMNS]
        self.table.selection = [0]
        self._draw_detail()

    @param.depends("colour_by", "rows_by", "map_extent", "basemap", "pooling", "min_obs",
                   "conditioned", "coverage", "start_from_actual", "start_delay_min",
                   watch=True)
    def _recolour(self) -> None:
        self._draw_detail()

    def _draw_detail(self) -> None:
        if not self.table.selection or self._shown.empty:
            return
        row = self._shown.iloc[self.table.selection[0]]
        line, day = row["line"], row["day"]

        cells = self._con.execute(
            """
            SELECT station, bpuic, hour, canton, median_delay_s, median_gain_s,
                   n_trips, n_measured, n_cancelled, lat, lon,
                   base_delay_s, base_iqr_s, base_n
            FROM slice WHERE line = ? AND hour::DATE = ?
            """,
            [line, day],
        ).df()
        if cells.empty:
            self.detail[:] = [pn.pane.Markdown("No cells for that line and day.")]
            return

        # Which cells the gates actually flagged, so the heatmap can show only those rows and
        # the tooltip can say how far above normal each one was.
        scored = model.score_cells(model.add_baseline_columns(cells), self.params)
        cells = cells.assign(
            excess_min=(scored["excess_s"] / 60).round(1),
            score=scored["score"].round(1),
            flagged=model.worth_reporting(scored, self.params),
        )

        header = pn.pane.Markdown(
            f"### {line} &nbsp;·&nbsp; {self._routes.get(line, 'route unknown')}"
            f" &nbsp;·&nbsp; {day:%d %B %Y}\n"
            f"Delay was added first at **{row['started_at']}** ({row['gain_min']:.1f} min there). "
            f"{row['stations']} stations over {row['hours']} h, "
            f"{row['severity_h']:.1f} train-hours of delay."
        )
        measure = "how late" if self.colour_by == "level" else "where delay was added"
        self.detail[:] = [
            header,
            self._forecast(cells, line, row),
            self._panels(cells, line, measure),
            self._judgement(row),
        ]

    def _judgement(self, row: pd.Series) -> pn.viewable.Viewable:
        """Record what the analyst concluded, with the settings that produced it.

        This is the loop closing. The algorithm proposes an episode and a prediction, the person
        decides whether it held, and the decision is kept with the model that produced it so it
        can be read back in context later.
        """
        key = incidents.key_of(row["line"], str(row["day"]), str(row["peak_hour"]))
        already = self._verdicts.get(key)

        note = pn.widgets.TextInput(
            placeholder="what happened, if you know", name="", width=430)
        said = pn.pane.Markdown(
            f"Last recorded: **{already}**" if already else "Not judged yet.",
            styles={"font-size": "0.82em", "color": style.INK_SOFT},
        )

        def record(verdict: str):
            def clicked(_event) -> None:
                incidents.append(INCIDENT_LOG, incidents.Entry(
                    line=row["line"], day=str(row["day"]), episode_start=str(row["peak_hour"]),
                    started_at=str(row["started_at"]), verdict=verdict, note=note.value,
                    predicted_end_min=self._last_prediction.get("predicted"),
                    actual_end_min=self._last_prediction.get("actual"),
                    pooling=self.pooling, min_obs=self.min_obs, conditioned=self.conditioned,
                ))
                self._verdicts = incidents.latest_verdicts(INCIDENT_LOG)
                said.object = f"Recorded **{verdict}** just now."
                note.value = ""
            return clicked

        # Separate buttons rather than a radio group. A radio group always has something
        # selected, so the first verdict reads as the tool's own answer before anyone has
        # judged anything, and the study is partly about whether people disagree with it.
        buttons = []
        for choice in incidents.VERDICTS:
            button = pn.widgets.Button(
                name=choice, width=150,
                button_type="primary" if choice == already else "default",
            )
            button.on_click(record(choice))
            buttons.append(button)

        return pn.Column(
            pn.Row(pn.pane.Markdown("**Did the model get it right?**", width=200), *buttons),
            pn.Row(note, said),
            width=CONTENT_W,
        )

    def _panels(self, cells: pd.DataFrame, line: str, measure: str) -> pn.viewable.Viewable:
        """The three panels under the forecast.

        The heatmap gets the largest share because it is the only one of the three with two
        dimensions in it.
        """
        day, route, where = PANEL_W
        return pn.Row(
            pn.Column("**The whole day, hour by hour**", self._profile(cells, day), width=day),
            pn.Column(f"**Along the route, hour by hour**, {measure}",
                      self._heatmap(cells, line, route), width=route),
            pn.Column("**Every stop, worst of the day**",
                      self._map(cells, line, where), width=where),
            width=CONTENT_W,
        )

    def _episode_strip(self, cells: pd.DataFrame, line: str) -> pn.viewable.Viewable:
        """The anomaly detector, kept as one of the three panels under the forecast.

        It is no longer the whole tool. It is how you find the line-days worth asking the
        model about.
        """
        same_line = self._shown[self._shown["line"] == line]
        if same_line.empty:
            return pn.pane.Markdown("No other episodes on this line.")
        shown = same_line[["start", "hours", "stations", "severity_h", "started_at"]].head(12)
        return pn.widgets.Tabulator(
            shown, height=360, disabled=True, show_index=False, theme="simple",
            titles=TABLE_TITLES,
        )

    @property
    def model_params(self) -> prop.Params:
        return prop.Params(
            pooling=self.pooling, min_obs=self.min_obs,
            conditioned=self.conditioned, coverage=self.coverage,
        )

    def coefficients(self) -> pd.DataFrame:
        """Refit, cached on the settings that change the fit."""
        key = (self.pooling, self.min_obs, self.conditioned)
        if self._coef_key != key:
            self._coef = prop.fit(self._legs, self.model_params)
            self._coef_key = key
        return self._coef

    def _forecast(self, cells: pd.DataFrame, line: str, row: pd.Series) -> pn.viewable.Viewable:
        """What the model expected along the route against what actually happened.

        The prediction starts at the stop where the delay was first added, with the delay really
        observed there, and walks the rest of the route. Drawing the observed delays over it
        turns the panel into one question: did this line behave the way it normally does?

        The band widens with every leg because the variance compounds. Where the band swallows
        the observed line the model has stopped saying anything useful, and that should be
        visible rather than hidden behind one confident curve.
        """
        peak_hour = pd.Timestamp(row["peak_hour"])
        condition = condition_for(peak_hour) if self.conditioned else None
        route = prop.route_legs(self._stop_order, line, condition)
        if route.empty:
            return pn.pane.Markdown("No route for this line.")

        snapshot = (
            cells[cells["hour"] == peak_hour]
            .groupby("station", as_index=False)
            .agg(observed=("median_delay_s", "median"), normal=("base_delay_s", "median"))
        )
        observed = dict(zip(snapshot["station"], snapshot["observed"], strict=True))
        normal = dict(zip(snapshot["station"], snapshot["normal"], strict=True))

        # Start where the delay was first added, if that stop is on the route.
        names = [route.iloc[0]["from_name"], *route["to_name"].tolist()]
        on_route = row["started_at"] in names
        origin = row["started_at"] if on_route else names[0]
        start_at = names.index(origin)
        remaining = route.iloc[start_at:].reset_index(drop=True)
        if remaining.empty:
            remaining, start_at = route, 0

        actual_start = observed.get(origin)
        if self.start_from_actual and actual_start is not None and not pd.isna(actual_start):
            start_s = float(actual_start)
            source = f"the {actual_start / 60:.1f} min actually seen at {origin}"
        else:
            start_s = self.start_delay_min * 60
            source = f"a what-if of {self.start_delay_min:.1f} min at {origin}"

        predicted = prop.predict_route(remaining, start_s, self.coefficients(), self.model_params)
        predicted["step"] = range(len(predicted))
        def minutes(lookup: dict, station: str) -> float:
            value = lookup.get(station)
            return float("nan") if value is None or pd.isna(value) else value / 60

        predicted["actual_min"] = [minutes(observed, s) for s in predicted["station"]]
        # What this slot normally runs at. This is the baseline the scoring compares against,
        # and the reason for using robust statistics instead of a learned detector was that it
        # can be drawn. Until now it never was.
        predicted["normal_min"] = [minutes(normal, s) for s in predicted["station"]]
        for column in ("delay_s", "lower_s", "upper_s"):
            predicted[column.replace("_s", "_min")] = (predicted[column] / 60).round(2)

        every = max(1, -(-len(predicted) // 12))
        ticks = [(i, s) for i, s in enumerate(predicted["station"]) if i % every == 0]
        hover = HoverTool(mode="vline", tooltips=[
            ("", "@station"),
            ("model expects", "@delay_min{0.0} min"),
            ("band", "@lower_min{0.0} to @upper_min{0.0} min"),
            ("actually", "@actual_min{0.0} min"),
            ("normal here", "@normal_min{0.0} min"),
        ])
        # delay_min first: the scatter takes its y from the first vdim, and the rest are
        # along for the tooltip.
        vdims = ["delay_min", "station", "lower_min", "upper_min", "actual_min",
                 "normal_min", "a", "level"]

        band = hv.Area(
            predicted, "step", ["lower_min", "upper_min"], label=f"{self.coverage:.0%} band"
        ).opts(color=style.SERIES[0], alpha=0.10, line_alpha=0)
        centre = hv.Curve(predicted, "step", "delay_min", label="model expects").opts(
            color=style.SERIES[0], line_width=style.LINE_W + 0.5)
        marks = hv.Scatter(predicted, "step", vdims).opts(
            color=style.SERIES[0], size=8, tools=[hover],
            line_color=style.SURFACE, line_width=2)
        truth = hv.Curve(
            predicted.dropna(subset=["actual_min"]), "step", "actual_min", label="what happened"
        ).opts(color=style.SERIES[1], line_width=style.LINE_W + 0.5, line_dash="dashed")
        usual = hv.Curve(
            predicted.dropna(subset=["normal_min"]), "step", "normal_min", label="normal here"
        ).opts(color=style.INK_SOFT, line_width=1.4, alpha=0.65, line_dash="dotted")
        zero = hv.HLine(0).opts(color=style.INK_SOFT, line_width=0.8, alpha=0.4)

        # The band can run to five minutes either side of a route that only ever moves by two,
        # and letting it set the y range leaves the comparison the chart exists for in a
        # twentieth of the height. The lines set the range and the band clips at the frame.
        drawn = pd.concat([predicted[c] for c in ("delay_min", "actual_min", "normal_min")])
        pad = max(0.6, 0.15 * (drawn.max() - drawn.min()))
        chart = (zero * band * usual * centre * truth * marks).opts(
            height=360, width=CONTENT_W, show_grid=True, hooks=[no_scroll_zoom],
            toolbar="above", ylim=(drawn.min() - pad, drawn.max() + pad), yticks=8,
            xticks=ticks, xrotation=35,
            xlabel="", ylabel="delay (minutes)", fontscale=style.FONTSCALE,
            title=f"forecast from {origin}",
            legend_position="top", legend_cols=4,
            legend_opts={"orientation": "horizontal", "glyph_height": 14, "glyph_width": 22},
        )
        tail = predicted.iloc[-1]
        self._last_prediction = {
            "predicted": round(float(tail["delay_min"]), 2),
            "actual": (None if pd.isna(tail["actual_min"])
                       else round(float(tail["actual_min"]), 2)),
        }
        elsewhere = "" if on_route else (
            f"The delay was first added at {row['started_at']}, which is not on this route, "
            f"so the forecast starts at {origin} instead and says nothing about the stops "
            f"before it. "
        )
        caption = pn.pane.Markdown(
            f"{elsewhere}Starting from {source}, measured at {peak_hour:%H:%M}. "
            f"The band is calibrated to hold the truth "
            f"{100 * self.coverage:.0f}% of the time and widens with every leg, so where it "
            f"swallows the dashed line the model has run out of things to say.",
            styles={"font-size": "0.82em", "color": style.INK_SOFT, "max-width": "76ch"},
        )
        return pn.Column(pn.pane.HoloViews(chart), caption, width=CONTENT_W)

    def _heatmap(self, cells: pd.DataFrame, line: str,
                 width: int = PANEL_W[1]) -> pn.viewable.Viewable:
        """The route against the hour, coloured by whichever measure is selected.

        A long line has sixty or more stops, which makes an unreadable axis, so the rows are
        whatever `rows_by` asks for: only the stations the gates flagged, every station, or
        rolled up to cantons. Cantons lose the route order, which is the point of the view, so
        they are an option rather than the default.

        Rows are aggregated in pandas first. Handing HeatMap duplicate (hour, station) pairs,
        which happens when two stop ids share a name, makes it aggregate silently and drop the
        value columns, and the tooltip then shows ??? instead of numbers.
        """
        level = self.colour_by == "level"
        field = "median_delay_s" if level else "median_gain_s"
        order = self._stop_order[self._stop_order["line"] == line]
        pos = dict(zip(order["station"], order["position"], strict=True))

        frame = cells.assign(
            hour_of_day=cells["hour"].dt.hour,
            value=(cells[field] / 60).round(2),
            position=cells["station"].map(pos),
        ).dropna(subset=["value"])

        row_label = "canton" if self.rows_by == "canton" else "station"
        if self.rows_by == "canton":
            frame = frame.assign(row=frame["canton"].fillna("unknown"))
            frame["position"] = frame.groupby("row")["position"].transform("median")
        else:
            if self.rows_by == "flagged" and frame["flagged"].any():
                keep = set(frame.loc[frame["flagged"], "station"])
                frame = frame[frame["station"].isin(keep)]
            frame = frame.assign(row=frame["station"])
        frame = frame.dropna(subset=["position"])
        if frame.empty:
            return pn.pane.Markdown("Nothing to draw for this measure.")

        grid = frame.groupby(["row", "hour_of_day"], as_index=False).agg(
            value=("value", "median"), position=("position", "first"),
            trips=("n_trips", "sum"), measured=("n_measured", "sum"),
            cancelled=("n_cancelled", "sum"), excess_min=("excess_min", "max"),
        )
        grid["value"] = grid["value"].round(1)

        # Cells resting on thin or mostly estimated evidence are faded rather than dropped.
        # Hiding them would quietly claim the hour was fine when nothing was measured in it.
        share = (grid["measured"] / grid["trips"].replace(0, float("nan"))).fillna(0.0)
        enough = (grid["measured"] >= self.min_measured) & (share >= self.min_share_real)
        grid["evidence"] = enough.map({True: 1.0, False: 0.45})
        grid["share_measured"] = share.round(2)
        thin = int((~enough).sum())
        rows = grid.sort_values("position", ascending=False)["row"].drop_duplicates().tolist()

        hover = HoverTool(tooltips=[
            ("", "@row"),
            ("hour", "@hour_of_day{0}:00"),
            ("minutes", "@value{0.0}"),
            ("above normal", "@excess_min{0.0} min"),
            ("trains", "@measured of @trips measured, @cancelled cancelled"),
            ("faded means", "thin evidence, @share_measured measured"),
        ])
        heat = hv.HeatMap(
            grid,
            kdims=[hv.Dimension("hour_of_day", label="hour of day"),
                   hv.Dimension("row", label=row_label, values=rows)],
            vdims=["value", "excess_min", "trips", "measured", "cancelled",
                   "evidence", "share_measured"],
        ).opts(
            opts.HeatMap(
                cmap=style.BLUE_RAMP if level else style.GAIN_RAMP,
                clim=(0, 15) if level else (-10, 10),
                alpha=hv.dim("evidence"),
                colorbar=True, tools=[hover], height=400, width=width,
                hooks=[no_scroll_zoom], ylabel="stops in route order",
                # The whole day, always. A range that follows the data makes two episodes
                # look the same width when one lasted three hours and the other twelve.
                xlim=(-0.5, 23.5),
                colorbar_opts={
                    "title": "min late" if level else "min added",
                    "width": 12, "padding": 6, "label_standoff": 5,
                    "major_label_overrides": {15: "15+"} if level else {},
                },
                title="", fontscale=style.FONTSCALE,
                xticks=list(range(0, 24, 3)), toolbar="above",
            )
        )
        note = pn.pane.Markdown(
            f"{len(rows)} of {len(set(cells['station']))} stops shown"
            + (f". {thin} cells are faded: too few trains measured to trust the number"
               if thin else ""),
            styles={"font-size": "0.78em", "color": style.INK_SOFT},
        )
        return pn.Column(pn.pane.HoloViews(heat), note, width=width)

    def _map(self, cells: pd.DataFrame, line: str,
             width: int = PANEL_W[2]) -> pn.viewable.Viewable:
        """Every stop of the line in route order, coloured by its worst hour that day."""
        worst = (
            cells.dropna(subset=["lat", "lon"])
            .groupby("station", as_index=False)
            .agg(lat=("lat", "first"), lon=("lon", "first"),
                 worst_min=("median_delay_s", lambda s: round(s.max() / 60, 1)))
        )
        order = self._stop_order[self._stop_order["line"] == line]
        worst = worst.merge(
            order[["station", "position", "on_main_route"]], on="station", how="left"
        ).sort_values("position")
        if worst.empty:
            return pn.pane.Markdown("No coordinates for this line.")

        x, y = lon_lat_to_easting_northing(worst["lon"].to_numpy(), worst["lat"].to_numpy())
        worst = worst.assign(x=x, y=y)

        # Only the main route gets a connecting line. Stops served by another variant sit off
        # it, and joining them in position order drew a fan across the country rather than a
        # railway.
        main = worst[worst["on_main_route"].fillna(False)]
        route = hv.Path([main[["x", "y"]].to_numpy()] if len(main) > 1 else []).opts(
            color=style.INK_SOFT, line_width=2, alpha=0.45
        )
        hover = HoverTool(tooltips=[("", "@station"), ("worst hour", "@worst_min{0.0} min late")])
        stops = hv.Points(worst, ["x", "y"], ["station", "worst_min"]).opts(
            color="worst_min", cmap=style.BLUE_RAMP, clim=(0, 15), size=12, tools=[hover],
            line_color=style.SURFACE, line_width=2, colorbar=True,
            colorbar_opts={"title": "min late", "width": 12, "padding": 6},
        )
        # Full opacity. Dimming a basemap that is already nearly white left nothing to see.
        if TILE_URL:
            tiles = hv.Tiles(TILE_URL)
        else:
            tiles = getattr(hv.element.tiles, BASEMAPS[self.basemap])()
        tiles = tiles.opts(alpha=0.78)
        xlim, ylim = self._map_bounds(worst)
        return pn.pane.HoloViews(
            (tiles * route * stops).opts(
                height=400, width=width, fontscale=style.FONTSCALE,
                xaxis=None, yaxis=None,
                hooks=[no_scroll_zoom], toolbar="above",
                xlim=xlim, ylim=ylim, title="",
            ),
            width=width,
        )

    def _map_bounds(self, stops: pd.DataFrame) -> tuple[tuple, tuple]:
        """The view the map opens on, in Web Mercator metres.

        Fitting tightly to a five kilometre line gives a suburb with no landmarks in it, so the
        fitted view has a floor on its span and some padding. The country view is there for
        when the question is where in Switzerland this is rather than which stops are involved.
        """
        if self.map_extent == "country":
            west, south, east, north = SWISS_BOUNDS_DEG
            x, y = lon_lat_to_easting_northing([west, east], [south, north])
            return (x[0], x[1]), (y[0], y[1])

        cx, cy = stops["x"].mean(), stops["y"].mean()
        span = max(
            stops["x"].max() - stops["x"].min(),
            stops["y"].max() - stops["y"].min(),
            MIN_SPAN_M,
        ) * 1.25
        return (cx - span / 2, cx + span / 2), (cy - span / 2, cy + span / 2)

    def _profile(self, cells: pd.DataFrame,
                 width: int = PANEL_W[0]) -> pn.viewable.Viewable:
        """The line hour by hour: how late, how much cancelled, how much was measured.

        Three stacked plots on a shared hour axis rather than two scales on one, because
        aligning two scales would suggest a relationship the chart has not measured.
        """
        hourly = (
            cells.assign(hour_of_day=cells["hour"].dt.hour)
            .groupby("hour_of_day", as_index=False)
            .agg(delay_min=("median_delay_s", lambda s: s.median() / 60),
                 trips=("n_trips", "sum"), measured=("n_measured", "sum"),
                 cancelled=("n_cancelled", "sum"))
        )
        hourly["cancelled_pct"] = (
            100 * hourly["cancelled"] / hourly["trips"].replace(0, float("nan"))
        ).fillna(0).round(1)
        hourly["delay_min"] = hourly["delay_min"].round(1)

        hover = HoverTool(tooltips=[
            ("hour", "@hour_of_day{0}:00"),
            ("median delay", "@delay_min{0.0} min"),
            ("cancelled", "@cancelled of @trips (@cancelled_pct{0.0}%)"),
            ("measured", "@measured arrivals"),
        ])
        vdims = ["trips", "measured", "cancelled", "cancelled_pct", "delay_min"]

        def band(column: str, label: str, colour: str, ylim=None, last: bool = False):
            """A filled line with visible markers, because a bare Area will not hover."""
            area = hv.Area(hourly, "hour_of_day", column).opts(
                color=colour, alpha=0.18, line_alpha=0)
            line = hv.Curve(hourly, "hour_of_day", column).opts(color=colour, line_width=2)
            dots = hv.Scatter(hourly, "hour_of_day", [column, *vdims]).opts(
                color=colour, size=5, tools=[hover], line_color=style.SURFACE, line_width=1)
            # ylim is only passed when there is one. HoloViews rejects None, and it does so at
            # render time rather than when the object is built.
            limit = {"ylim": ylim} if ylim is not None else {}
            return (area * line * dots).opts(
                width=width, height=135, ylabel=label, fontscale=0.75,
                xlabel="hour of day" if last else "", xticks=list(range(0, 24, 3)),
                show_grid=True, toolbar=None, default_tools=[], **limit,
            )

        return pn.Column(
            pn.pane.HoloViews(band("delay_min", "delay (min)", style.SERIES[0])),
            pn.pane.HoloViews(band("cancelled_pct", "cancelled (%)", style.SERIES[1],
                                   ylim=(0, 100))),
            pn.pane.HoloViews(band("measured", "measured", style.INK_SOFT, last=True)),
            width=width,
        )

    @param.depends(
        "scenario", "categories", "canton", "rank_by", "z_threshold", "min_excess_min",
        "min_measured", "min_share_real", "cancel_weight", "cancel_headway_multiple",
        "baseline_weeks",
    )
    def status(self) -> pn.viewable.Viewable:
        found = self._episodes()
        return pn.pane.Markdown(
            f"{self._n_cells:,} cells &nbsp;|&nbsp; {self.n_flagged:,} flagged &nbsp;|&nbsp; "
            f"top {len(self._shown)} of {len(found):,} episodes shown &nbsp;|&nbsp; "
            f"load {self.load_ms:.0f} ms, rescore {self.score_ms:.0f} ms"
        )


def sidebar(explorer: Explorer) -> list:
    """The controls, grouped by the question each group answers.

    Finding episodes comes first because it produces what the table shows, and the model comes
    second because it explains the episode you then pick.
    """
    return [
        "### Which days to look at",
        explorer.param.rank_by,
        explorer.param.z_threshold,
        explorer.param.min_excess_min,
        explorer.param.min_measured,
        explorer.param.min_share_real,
        explorer.param.cancel_weight,
        explorer.param.cancel_headway_multiple,
        explorer.param.baseline_weeks,
        pn.layout.Divider(),
        "### What the model expects",
        pn.pane.Markdown("Each of these refits the model rather than filtering its output.",
                         styles={"font-size": "0.85em", "color": style.INK_SOFT}),
        explorer.param.pooling,
        explorer.param.min_obs,
        explorer.param.conditioned,
        explorer.param.coverage,
        explorer.param.start_from_actual,
        explorer.param.start_delay_min,
        pn.layout.Divider(),
        "### Where the delay came from",
        explorer.param.colour_by,
        explorer.param.rows_by,
        pn.layout.Divider(),
        "### Map",
        explorer.param.map_extent,
        explorer.param.basemap,
    ]


def main_column(explorer: Explorer) -> list:
    """Filters, then the list, then whatever the selected row opens up, then the counters."""
    return [
        pn.FlexBox(
            pn.widgets.Select.from_param(explorer.param.scenario, width=140),
            pn.widgets.Select.from_param(explorer.param.canton, width=130),
            pn.widgets.CheckBoxGroup.from_param(
                explorer.param.categories, inline=True, name="Train category"),
            align_items="start", gap="18px",
        ),
        pn.layout.Divider(margin=(2, 0, 8, 0)),
        pn.pane.Markdown(
            "Each row is one line on one day the algorithm thinks went wrong. "
            "Harm is train-hours of delay, worst is the worst hour in minutes. "
            "Pick one to see what the model expected.",
            styles={"font-size": "0.86em", "color": style.INK_SOFT},
        ),
        explorer.table,
        explorer.detail,
        pn.layout.Divider(margin=(16, 0, 4, 0)),
        explorer.status,
    ]


PAGE_CSS = f"""
body {{ background: {style.SURFACE}; margin: 0; }}
.van-title {{
    font-size: 1.15rem; font-weight: 600; color: {style.INK};
    padding: 14px 0 10px 0; border-bottom: 1px solid {style.RULE}; margin-bottom: 14px;
}}
.van-controls {{ border-right: 1px solid {style.RULE}; padding-right: 18px; }}
"""


def layout() -> pn.viewable.Viewable:
    """Controls beside the page rather than in a template sidebar.

    FastListTemplate opens its sidebar after the first paint. Bokeh has measured the main
    area by then and keeps the wider figure, so every chart overflowed the column it sits in
    by about the width of the sidebar. A plain row is measured once and is right.
    """
    pn.config.raw_css.append(PAGE_CSS)
    explorer = Explorer()
    explorer.refresh()
    return pn.Row(
        pn.Column(*sidebar(explorer), width=CONTROLS_W, css_classes=["van-controls"]),
        pn.Column(
            pn.pane.HTML("<div class='van-title'>Swiss train disruptions</div>"),
            *main_column(explorer),
            width=CONTENT_W,
        ),
        width=CONTROLS_W + CONTENT_W + 40,
        margin=(0, 16),
    )


if pn.state.curdoc is not None or "panel" in Path(sys.argv[0]).name:
    layout().servable()
