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
from holoviews import opts
from holoviews.util.transform import lon_lat_to_easting_northing

import model
import style

pn.extension("tabulator", sizing_mode="stretch_width")
hv.extension("bokeh")

PROCESSED = Path(__file__).resolve().parents[1] / "data" / "processed"
CELLS = PROCESSED / "cells.parquet"
STATIONS = PROCESSED / "stations.parquet"
STOP_ORDER = PROCESSED / "stop_order.parquet"

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

TABLE_COLUMNS = [
    "line", "route", "start", "hours", "stations", "severity_h",
    "score", "worst_min", "started_at", "gain_min", "measured", "share",
]


def _label(code: str, pct: float) -> str:
    name = CATEGORY_NAMES.get(code)
    return f"{name} ({code}) {pct:.0f}%" if name else f"{code} {pct:.0f}%"


class Explorer(param.Parameterized):
    """Filters narrow what is analysed. Parameters change how it is scored."""

    scenario = param.Selector(objects=list(SCENARIOS), default="winter 25/26")
    categories = param.ListSelector(default=[], objects=[])
    canton = param.Selector(default="all", objects=["all"])

    rank_by = param.Selector(objects=list(model.RANKINGS), default="severity")
    z_threshold = param.Number(3.0, bounds=(1.0, 15.0), step=0.5)
    min_excess_min = param.Number(3.0, bounds=(0.0, 20.0), step=0.5)
    min_measured = param.Integer(2, bounds=(1, 20))
    min_share_real = param.Number(0.5, bounds=(0.0, 1.0), step=0.05)
    cancel_weight = param.Number(0.0, bounds=(0.0, 20.0), step=1.0)
    cancel_headway_multiple = param.Number(1.0, bounds=(0.0, 4.0), step=0.25)
    baseline_weeks = param.Selector(objects=list(model.BASELINE_WEEKS), default=8)

    colour_by = param.Selector(
        objects={"how late it was": "level", "where it was added": "gain"}, default="level"
    )
    rows_by = param.Selector(
        objects={
            "stations that were flagged": "flagged",
            "every station on the line": "all",
            "cantons": "canton",
        },
        default="flagged",
    )
    map_extent = param.Selector(
        objects={"around the line": "line", "all of Switzerland": "country"}, default="line"
    )
    basemap = param.Selector(objects=list(BASEMAPS), default="OpenStreetMap")

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        missing = [p.name for p in (CELLS, STATIONS, STOP_ORDER) if not p.exists()]
        if missing:
            raise SystemExit(f"missing {', '.join(missing)} in data/processed. Run prepare.py.")

        self._con = duckdb.connect()
        self._con.execute("SET threads = 6")
        self._con.execute("SET memory_limit = '8GB'")

        self._stop_order = self._con.execute(f"SELECT * FROM read_parquet('{STOP_ORDER}')").df()
        self._routes = self._route_labels()

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
            pd.DataFrame(columns=TABLE_COLUMNS), height=235, disabled=True,
            show_index=False, selectable=1, theme="simple",
            pagination="local", page_size=6,
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
            label = f"everything else ({len(rest)} kinds) {share:.0f}%"
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
        self.table.value = self._shown[TABLE_COLUMNS]
        self.table.selection = [0]
        self._draw_detail()

    @param.depends("colour_by", "rows_by", "map_extent", "basemap", watch=True)
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
        self.detail[:] = [
            header,
            pn.Row(self._heatmap(cells, line), self._map(cells, line), height=420),
            self._profile(cells),
        ]

    def _heatmap(self, cells: pd.DataFrame, line: str) -> pn.viewable.Viewable:
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
        rows = grid.sort_values("position", ascending=False)["row"].drop_duplicates().tolist()

        hover = HoverTool(tooltips=[
            ("", "@row"),
            ("hour", "@hour_of_day{0}:00"),
            ("minutes", "@value{0.0}"),
            ("above normal", "@excess_min{0.0} min"),
            ("trains", "@measured of @trips measured, @cancelled cancelled"),
        ])
        heat = hv.HeatMap(
            grid,
            kdims=[hv.Dimension("hour_of_day", label="hour of day"),
                   hv.Dimension("row", label=row_label, values=rows)],
            vdims=["value", "excess_min", "trips", "measured", "cancelled"],
        ).opts(
            opts.HeatMap(
                cmap="Blues" if level else "RdBu_r",
                clim=(0, 15) if level else (-10, 10),
                colorbar=True, tools=[hover], height=400, responsive=True,
                title="minutes late" if level else "minutes added since the previous stop",
                fontscale=0.75, xticks=list(range(0, 24, 3)), toolbar="above",
            )
        )
        return pn.pane.HoloViews(heat, sizing_mode="stretch_width")

    def _map(self, cells: pd.DataFrame, line: str) -> pn.viewable.Viewable:
        """Every stop of the line in route order, coloured by its worst hour that day."""
        worst = (
            cells.dropna(subset=["lat", "lon"])
            .groupby("station", as_index=False)
            .agg(lat=("lat", "first"), lon=("lon", "first"),
                 worst_min=("median_delay_s", lambda s: round(s.max() / 60, 1)))
        )
        order = self._stop_order[self._stop_order["line"] == line][["station", "position"]]
        worst = worst.merge(order, on="station", how="left").sort_values("position")
        if worst.empty:
            return pn.pane.Markdown("No coordinates for this line.")

        x, y = lon_lat_to_easting_northing(worst["lon"].to_numpy(), worst["lat"].to_numpy())
        worst = worst.assign(x=x, y=y)

        route = hv.Path([worst[["x", "y"]].to_numpy()]).opts(
            color=style.INK_SOFT, line_width=2, alpha=0.45
        )
        hover = HoverTool(tooltips=[("", "@station"), ("worst hour", "@worst_min{0.0} min late")])
        stops = hv.Points(worst, ["x", "y"], ["station", "worst_min"]).opts(
            color="worst_min", cmap="Blues", clim=(0, 15), size=9, tools=[hover],
            line_color="#444444", line_width=0.8, colorbar=False,
        )
        # Full opacity. Dimming a basemap that is already nearly white left nothing to see.
        if TILE_URL:
            tiles = hv.Tiles(TILE_URL)
        else:
            tiles = getattr(hv.element.tiles, BASEMAPS[self.basemap])()
        xlim, ylim = self._map_bounds(worst)
        return pn.pane.HoloViews(
            (tiles * route * stops).opts(
                height=400, responsive=True, fontscale=0.8, xaxis=None, yaxis=None,
                xlim=xlim, ylim=ylim, title=f"{line}: every stop",
            ),
            sizing_mode="stretch_width",
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

    def _profile(self, cells: pd.DataFrame) -> pn.viewable.Viewable:
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
                responsive=True, height=135, ylabel=label, fontscale=0.75,
                xlabel="hour of day" if last else "", xticks=list(range(0, 24, 3)),
                show_grid=True, gridstyle={"grid_line_alpha": 0.25},
                toolbar=None, default_tools=[], **limit,
            )

        panels = (
            band("delay_min", "delay (min)", style.SERIES[0])
            + band("cancelled_pct", "cancelled (%)", style.SERIES[1], ylim=(0, 100))
            + band("measured", "measured", style.INK_SOFT, last=True)
        )
        return pn.pane.HoloViews(panels.cols(1), sizing_mode="stretch_width")

    @param.depends(
        "scenario", "categories", "canton", "rank_by", "z_threshold", "min_excess_min",
        "min_measured", "min_share_real", "cancel_weight", "cancel_headway_multiple",
        "baseline_weeks",
    )
    def status(self) -> pn.viewable.Viewable:
        found = self._episodes()
        return pn.pane.Markdown(
            f"{self._n_cells:,} cells &nbsp;|&nbsp; {self.n_flagged:,} flagged &nbsp;|&nbsp; "
            f"{len(found):,} episodes &nbsp;|&nbsp; load {self.load_ms:.0f} ms &nbsp;|&nbsp; "
            f"rescore **{self.score_ms:.0f} ms**"
        )


def layout() -> pn.viewable.Viewable:
    explorer = Explorer()
    explorer.refresh()

    filters = pn.Row(
        pn.widgets.Select.from_param(explorer.param.scenario, width=140),
        pn.widgets.CheckBoxGroup.from_param(explorer.param.categories, inline=True),
        pn.widgets.Select.from_param(explorer.param.canton, width=130),
    )
    parameters = pn.Column(
        "### Parameters",
        pn.pane.Markdown("How it scores, not what it shows.", styles={"font-size": "0.85em"}),
        explorer.param.rank_by,
        explorer.param.z_threshold,
        explorer.param.min_excess_min,
        explorer.param.min_measured,
        explorer.param.min_share_real,
        explorer.param.cancel_weight,
        explorer.param.cancel_headway_multiple,
        explorer.param.baseline_weeks,
        pn.layout.Divider(),
        "### Heatmap",
        explorer.param.colour_by,
        explorer.param.rows_by,
        pn.layout.Divider(),
        "### Map",
        explorer.param.map_extent,
        explorer.param.basemap,
        width=250,
    )
    return pn.template.FastListTemplate(
        title="Swiss train disruptions",
        sidebar=[parameters],
        main=[pn.Column(filters, explorer.status, explorer.table, explorer.detail)],
    )


if pn.state.curdoc is not None or "panel" in Path(sys.argv[0]).name:
    layout().servable()
