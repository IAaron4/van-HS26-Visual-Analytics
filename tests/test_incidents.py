"""The incident log is the record of what was done, so it must not lose or garble entries."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import incidents  # noqa: E402


def entry(**over) -> incidents.Entry:
    base = {
        "line": "R43", "day": "2026-02-16", "episode_start": "2026-02-16T08:00:00",
        "started_at": "Fiesch", "verdict": "model was right",
    }
    return incidents.Entry(**{**base, **over})


def test_an_entry_survives_a_round_trip(tmp_path: Path) -> None:
    log = tmp_path / "log.jsonl"
    incidents.append(log, entry(note="snow, line suspended from 13:00"))
    back = incidents.load(log)
    assert len(back) == 1
    assert back.iloc[0]["note"] == "snow, line suspended from 13:00"
    assert back.iloc[0]["line"] == "R43"


def test_entries_accumulate_rather_than_replace(tmp_path: Path) -> None:
    log = tmp_path / "log.jsonl"
    for i in range(5):
        incidents.append(log, entry(day=f"2026-02-{10 + i:02d}"))
    assert len(incidents.load(log)) == 5


def test_a_missing_log_is_empty_not_an_error(tmp_path: Path) -> None:
    out = incidents.load(tmp_path / "nothing.jsonl")
    assert out.empty
    assert list(out.columns) == incidents.LOG_COLUMNS


def test_a_half_written_last_line_does_not_lose_the_rest(tmp_path: Path) -> None:
    """A session killed mid append must not take the earlier judgements with it."""
    log = tmp_path / "log.jsonl"
    incidents.append(log, entry(day="2026-02-16"))
    incidents.append(log, entry(day="2026-02-17"))
    with log.open("a", encoding="utf-8") as handle:
        handle.write('{"line": "R43", "day": "2026-02-18", "verd')
    back = incidents.load(log)
    assert len(back) == 2


def test_changing_your_mind_replaces_the_verdict(tmp_path: Path) -> None:
    log = tmp_path / "log.jsonl"
    incidents.append(log, entry(verdict="model was right", decided_at="2026-02-16T09:00:00"))
    incidents.append(log, entry(verdict="model was wrong", decided_at="2026-02-16T10:00:00"))
    current = incidents.latest_verdicts(log)
    key = incidents.key_of("R43", "2026-02-16", "2026-02-16T08:00:00")
    assert current[key] == "model was wrong"
    assert len(incidents.load(log)) == 2, "the earlier judgement is still on the record"


def test_different_episodes_get_different_keys() -> None:
    a = incidents.key_of("R43", "2026-02-16", "2026-02-16T08:00:00")
    b = incidents.key_of("R43", "2026-02-16", "2026-02-16T14:00:00")
    assert a != b


def test_the_settings_are_kept_with_the_verdict(tmp_path: Path) -> None:
    """A verdict means nothing without the model that produced it."""
    log = tmp_path / "log.jsonl"
    incidents.append(log, entry(pooling="global", min_obs=500, conditioned=False))
    row = incidents.load(log).iloc[0]
    assert row["pooling"] == "global"
    assert row["min_obs"] == 500
    assert not row["conditioned"]  # pandas stores this as numpy.bool_, so no identity check


def test_an_invented_verdict_is_refused() -> None:
    with pytest.raises(ValueError, match="verdict"):
        entry(verdict="probably fine")


def test_unicode_in_a_note_survives(tmp_path: Path) -> None:
    log = tmp_path / "log.jsonl"
    incidents.append(log, entry(note="Störung zwischen Brig und Visp, Zürich HB betroffen"))
    assert "Störung" in incidents.load(log).iloc[0]["note"]
