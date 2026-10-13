"""The record of what the analyst decided.

Stage 2 asks how the system keeps a record of what was done, and this is it. Each entry says
which episode was judged, what the model predicted, what actually happened, what the settings
were at the time, and what the person concluded. Keeping the settings matters: a verdict only
means something if you can see the model that produced it.

Stored as JSON lines, appended and never rewritten, so a session cannot lose earlier entries and
the file stays readable without a tool.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

VERDICTS = ("model was right", "model was wrong", "not a real disruption", "cannot tell")

LOG_COLUMNS = [
    "decided_at", "line", "day", "started_at", "verdict", "note",
    "predicted_end_min", "actual_end_min", "pooling", "min_obs", "conditioned",
]


@dataclass(frozen=True)
class Entry:
    """One judgement. The model settings travel with it so it can be read back in context."""

    line: str
    day: str
    episode_start: str
    started_at: str
    verdict: str
    note: str = ""
    predicted_end_min: float | None = None
    actual_end_min: float | None = None
    pooling: str = "leg"
    min_obs: int = 60
    conditioned: bool = True
    decided_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds")
    )

    def __post_init__(self) -> None:
        if self.verdict not in VERDICTS:
            raise ValueError(f"verdict must be one of {VERDICTS}")


def key_of(line: str, day: str, episode_start: str) -> str:
    """Stable identifier for an episode, so a second look finds the earlier verdict."""
    return f"{line}|{day}|{episode_start}"


def append(path: Path, entry: Entry) -> None:
    """Add one judgement. Appending rather than rewriting means nothing earlier can be lost."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(asdict(entry), ensure_ascii=False) + "\n")


def load(path: Path) -> pd.DataFrame:
    """Every judgement so far, newest first. A missing or part written file is not an error."""
    if not path.exists():
        return pd.DataFrame(columns=LOG_COLUMNS)

    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            # A half written final line, from a session that stopped mid append. Everything
            # before it is still good, which is the reason for appending in the first place.
            continue
    if not rows:
        return pd.DataFrame(columns=LOG_COLUMNS)

    frame = pd.DataFrame(rows)
    frame["key"] = [
        key_of(r["line"], r["day"], r["episode_start"]) for r in rows
    ]
    return frame.sort_values("decided_at", ascending=False).reset_index(drop=True)


def latest_verdicts(path: Path) -> dict[str, str]:
    """The current verdict per episode, where changing your mind replaces the old one."""
    log = load(path)
    if log.empty:
        return {}
    newest = log.drop_duplicates(subset="key", keep="first")
    return dict(zip(newest["key"], newest["verdict"], strict=True))
