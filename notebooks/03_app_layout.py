"""Wireframe of the planned interface, to have something to point at in the clinic.

    python notebooks/03_app_layout.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import style  # noqa: E402

FIGURES = Path(__file__).resolve().parents[1] / "figures" / "before"

# x, y, width, height, heading, body lines
PANELS = [
    (0.0, 6.2, 10.0, 1.05, "Filters", [
        "which data is in play, applied to every view below",
        "scenario: winter 25/26 | summer 26     dates     category     line     canton",
    ]),
    (0.0, 0.0, 2.4, 6.0, "Parameters", [
        "how it scores, not",
        "what it shows",
        "",
        "z threshold",
        "minimum excess",
        "minimum measured",
        "share measured",
        "cancellation weight",
        "baseline window",
        "grain",
    ]),
    (2.6, 4.3, 7.4, 1.7, "Episode list", [
        "ranked candidates, worst first",
        "score  origin  duration  stations  measured  share",
        "confirm  /  reject  /  note a cause",
    ]),
    (2.6, 1.7, 4.3, 2.4, "Line and time heatmap", [
        "stations along the route",
        "against hour of day",
        "",
        "toggle:  level  |  gain",
    ]),
    (7.1, 1.7, 2.9, 2.4, "Map", [
        "the selected episode",
        "on the network",
        "",
        "origin marked",
    ]),
    (2.6, 0.0, 7.4, 1.4, "Evidence for the selected cell", [
        "observed against baseline, with the spread the baseline claims",
        "how many arrivals were measured, and what share of the service that is",
    ]),
]

ARROWS = [
    ((1.2, 6.2), (1.2, 6.02), ""),
    ((6.3, 6.2), (6.3, 6.02), ""),
    ((4.7, 4.3), (4.7, 4.1), "selection"),
    ((8.5, 4.3), (8.5, 4.1), ""),
    ((4.7, 1.7), (4.7, 1.4), ""),
]


def main() -> int:
    style.use()
    fig, ax = plt.subplots(figsize=(8.4, 5.8))

    for x, y, w, h, heading, body in PANELS:
        ax.add_patch(Rectangle((x, y), w, h, facecolor="#f3f1ec", edgecolor=style.RULE,
                               linewidth=1.0))
        ax.text(x + 0.16, y + h - 0.26, heading, fontsize=9, color=style.INK, weight="medium")
        for i, row in enumerate(body):
            ax.text(x + 0.16, y + h - 0.55 - i * 0.25, row, fontsize=7, color=style.INK_SOFT)

    for (x0, y0), (x1, y1), label in ARROWS:
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=9,
                                     color=style.SERIES[0], linewidth=1.2))
        if label:
            ax.text(x0 + 0.12, (y0 + y1) / 2, label, fontsize=7, color=style.SERIES[0],
                    va="center")

    ax.set_xlim(-0.15, 10.15)
    ax.set_ylim(-0.3, 7.6)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("Planned layout")
    style.caption(fig, "Filters narrow what is analysed. Parameters change how it is scored. "
                       "Picking an episode drives the three views below it.")
    fig.savefig(FIGURES / "05_app_layout.png", bbox_inches="tight")
    plt.close(fig)
    print("  05_app_layout.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
