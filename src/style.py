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


def use() -> None:
    """Apply the house style to matplotlib."""
    mpl.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 150,
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans"],
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
        "lines.linewidth": 1.6,
        "lines.markersize": 4.5,
    })


def caption(fig, text: str) -> None:
    """A short note under the figure, for the thing a reader would otherwise misread.

    Placed just outside the figure so savefig with a tight bounding box picks it up without
    leaving a gap the size of the axes.
    """
    fig.text(0.01, -0.015, text, fontsize=7, color=INK_SOFT, ha="left", va="top")
