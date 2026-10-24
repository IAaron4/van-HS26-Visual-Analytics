"""Every view has to survive being rendered, not merely built.

Building a HoloViews object validates almost nothing. Options are checked when the plot is
created, so a check that stops at construction passes while the served page throws. Every
interface bug this project has had escaped that way: ylim=None, an empty Dimension label, a
parameter never passed through. These tests render through Bokeh the way Panel does.
"""

from __future__ import annotations

import sys
from pathlib import Path

import holoviews as hv
import pandas as pd
import pytest
from bokeh.document import Document

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

PROCESSED = ROOT / "data" / "processed"
NEEDED = [PROCESSED / n for n in
          ("cells.parquet", "stations.parquet", "stop_order.parquet", "legs.parquet")]

pytestmark = pytest.mark.skipif(
    not all(p.exists() for p in NEEDED), reason="run prepare.py first"
)


@pytest.fixture(scope="module")
def explorer():
    import app
    hv.extension("bokeh")
    ex = app.Explorer()
    ex.refresh()
    return ex


@pytest.fixture(scope="module")
def selected(explorer):
    """The top episode plus the cells its detail views are drawn from."""
    import app
    row = explorer._shown.iloc[0]
    cells = explorer._con.execute(
        """
        SELECT station, bpuic, hour, canton, median_delay_s, median_gain_s,
               n_trips, n_measured, n_cancelled, lat, lon, base_delay_s, base_iqr_s, base_n
        FROM slice WHERE line = ? AND hour::DATE = ?
        """,
        [row["line"], row["day"]],
    ).df()
    scored = app.model.score_cells(app.model.add_baseline_columns(cells), explorer.params)
    return row, cells.assign(
        excess_min=(scored["excess_s"] / 60).round(1),
        score=scored["score"].round(1),
        flagged=app.model.worth_reporting(scored, explorer.params),
    )


def render(pane) -> None:
    """Render whatever a panel holds, the way Panel does when it serves."""
    obj = getattr(pane, "object", None)
    if isinstance(pane, pn_types()):
        for child in pane:
            render(child)
        return
    if obj is not None and not isinstance(obj, str):
        hv.renderer("bokeh").get_plot(obj)


def pn_types():
    import panel as pn
    return (pn.Column, pn.Row)


def test_the_episode_table_has_rows(explorer) -> None:
    assert len(explorer.table.value) > 0
    assert "route" in explorer.table.value.columns


def test_the_forecast_renders(explorer, selected) -> None:
    row, cells = selected
    render(explorer._forecast(cells, row["line"], row))


def test_the_forecast_draws_every_layer_against_the_delay_axis(explorer, selected) -> None:
    """A text column first in the vdims silently becomes the y axis.

    hv.Scatter takes its y from vdims[0], so listing station names there plots the markers on a
    categorical axis and buries the chart under one tick label per stop. It still renders, so
    only the dtype catches it.
    """
    row, cells = selected
    chart = explorer._forecast(cells, row["line"], row)[0].object
    for element in chart.traverse(lambda e: e, [hv.Element]):
        if not element.vdims:
            continue
        y = element.vdims[0]
        assert pd.api.types.is_numeric_dtype(element.dimension_values(y).dtype), \
            f"{type(element).__name__} puts {y.name} on the y axis"


@pytest.mark.parametrize("pooling", ["leg", "line", "global"])
def test_the_forecast_renders_at_every_pooling_level(explorer, selected, pooling) -> None:
    row, cells = selected
    explorer.pooling = pooling
    render(explorer._forecast(cells, row["line"], row))


def test_the_forecast_renders_as_a_what_if(explorer, selected) -> None:
    row, cells = selected
    explorer.start_from_actual = False
    explorer.start_delay_min = 12.0
    render(explorer._forecast(cells, row["line"], row))
    explorer.start_from_actual = True


@pytest.mark.parametrize("rows_by", ["flagged", "all", "canton"])
@pytest.mark.parametrize("colour_by", ["level", "gain"])
def test_the_heatmap_renders_every_way(explorer, selected, rows_by, colour_by) -> None:
    row, cells = selected
    explorer.rows_by, explorer.colour_by = rows_by, colour_by
    render(explorer._heatmap(cells, row["line"]))


@pytest.mark.parametrize("extent", ["line", "country"])
def test_the_map_renders(explorer, selected, extent) -> None:
    row, cells = selected
    explorer.map_extent = extent
    render(explorer._map(cells, row["line"]))


def test_every_basemap_renders(explorer, selected) -> None:
    import app
    row, cells = selected
    for name in app.BASEMAPS:
        explorer.basemap = name
        render(explorer._map(cells, row["line"]))


def test_hover_fields_all_exist(explorer, selected) -> None:
    """A tooltip pointing at a column that is not there shows ??? instead of a number."""
    import re
    row, cells = selected
    heat = explorer._heatmap(cells, row["line"])[0].object
    columns = set(heat.data.columns)
    tips = heat.opts.get("plot").kwargs["tools"][0].tooltips
    refs = {r for _, text in tips for r in re.findall(r"@(\w+)", text)}
    assert not refs - columns, f"tooltip references missing columns: {refs - columns}"


def test_no_chart_hijacks_the_scroll_wheel(explorer, selected) -> None:
    """Bokeh activates the wheel zoom on its own, and then the page cannot be scrolled."""
    row, cells = selected
    panes = [explorer._forecast(cells, row["line"], row)[0],
             explorer._heatmap(cells, row["line"])[0],
             explorer._map(cells, row["line"])]
    for pane in panes:
        plot = hv.renderer("bokeh").get_plot(pane.object)
        assert plot.state.toolbar.active_scroll is None, \
            f"{type(pane.object).__name__} still zooms on scroll"


def test_the_detail_column_has_the_four_panels(explorer) -> None:
    explorer.table.selection = [0]
    kinds = [type(o).__name__ for o in explorer.detail]
    assert len(explorer.detail) == 4, f"expected header, forecast, judgement, strip: {kinds}"


def test_the_whole_page_builds(explorer) -> None:
    import app
    app.layout().server_doc(doc=Document())


def test_selecting_several_episodes_keeps_working(explorer) -> None:
    for i in range(min(3, len(explorer.table.value))):
        explorer.table.selection = [i]
        for pane in explorer.detail:
            render(pane)
        assert not isinstance(explorer.detail[0], type(None))


def test_a_line_with_no_coordinates_does_not_crash(explorer, selected) -> None:
    row, cells = selected
    blank = cells.assign(lat=pd.NA, lon=pd.NA)
    pane = explorer._map(blank, row["line"])
    render(pane)
