"""Shared figure styling, so every figure in the report reads as one set."""

from __future__ import annotations

import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SOFT = "#52514e"
RULE = "#e2e0da"
# Sits just off the surface: visible as "nothing recorded", but never heavier than the
# lightest step of the ramp, which would read as more delay than an on-time hour.
NO_DATA = "#e9e7e1"

# Categorical slots, assigned in this order and never cycled.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]

# One hue, light to dark. Magnitude only.
BLUE_RAMP = [
    "#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7",
    "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b",
]

DELAY_CMAP = LinearSegmentedColormap.from_list("delay", BLUE_RAMP)
DELAY_CMAP.set_bad(NO_DATA)

# Blue to orange through a neutral grey. The two poles have to read as opposite, which two
# cool hues do not, and the midpoint has to read as nothing.
GAIN_RAMP = [
    "#2a78d6", "#5598e7", "#9ec5f4", "#cde2fb", NO_DATA,
    "#f8d2c2", "#f3ac8b", "#ef8a5f", "#eb6834",
]

FONT = "Segoe UI, DejaVu Sans, system-ui, sans-serif"

# One type scale and one line weight across the app, so four charts do not each pick their own.
FONTSCALE = 0.8
LINE_W = 2.0


def bokeh_theme() -> dict:
    """The same house style for the app that `use` gives the report figures.

    Bokeh ships italic axis labels, a box around the plot and mid-grey gridlines, none of which
    match the figures. Setting it here means the app and the report look like one piece of work.
    """
    text = {"text_font": FONT, "text_font_style": "normal"}
    return {"attrs": {
        "Plot": {
            "background_fill_color": SURFACE,
            "border_fill_color": SURFACE,
            "outline_line_color": None,
        },
        "Axis": {
            "axis_line_color": None,
            "major_tick_line_color": RULE,
            "minor_tick_line_color": None,
            "axis_label_text_color": INK_SOFT,
            "axis_label_text_font": FONT,
            "axis_label_text_font_style": "normal",
            "axis_label_text_font_size": "11px",
            "major_label_text_color": INK_SOFT,
            "major_label_text_font": FONT,
            "major_label_text_font_size": "11px",
        },
        "Grid": {"grid_line_color": RULE, "grid_line_width": 0.8, "grid_line_dash": "solid"},
        "Title": {"text_color": INK, "text_font_size": "13px", **text},
        "Legend": {
            "background_fill_color": SURFACE,
            "background_fill_alpha": 0.8,
            "border_line_color": None,
            "label_text_color": INK_SOFT,
            "label_text_font": FONT,
            "label_text_font_size": "11px",
            "spacing": 2,
            "padding": 6,
        },
        "ColorBar": {
            "background_fill_color": SURFACE,
            "bar_line_color": None,
            "major_tick_line_color": None,
            "major_label_text_color": INK_SOFT,
            "major_label_text_font": FONT,
            "major_label_text_font_size": "10px",
            "title_text_color": INK_SOFT,
            "title_text_font": FONT,
            "title_text_font_style": "normal",
            "title_text_font_size": "11px",
        },
    }}


def use() -> None:
    """Apply the house style to matplotlib."""
    mpl.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 150,
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "font.family": "sans-serif",
        "font.sans-serif": ["Segoe UI", "DejaVu Sans"],
        "font.size": 8.5,
        "text.color": INK,
        "axes.labelcolor": INK_SOFT,
        "axes.edgecolor": RULE,
        "axes.linewidth": 0.8,
        "axes.titlesize": 9.5,
        "axes.titleweight": "regular",
        "axes.titlelocation": "left",
        "axes.titlepad": 8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": RULE,
        "grid.linewidth": 0.6,
        "grid.linestyle": "-",
        "xtick.color": INK_SOFT,
        "ytick.color": INK_SOFT,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "legend.frameon": False,
        "legend.fontsize": 8,
        "lines.linewidth": LINE_W,
        "lines.markersize": 4.5,
    })


def caption(fig, text: str) -> None:
    """A short note under the figure, for the thing a reader would otherwise misread.

    Placed just outside the figure so savefig with a tight bounding box picks it up without
    leaving a gap the size of the axes.
    """
    fig.text(0.01, -0.015, text, fontsize=7, color=INK_SOFT, ha="left", va="top")
