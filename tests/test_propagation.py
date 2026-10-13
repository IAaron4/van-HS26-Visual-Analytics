"""The propagation model is fitted from sums rather than rows, so the sums have to be right."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import propagation as prop  # noqa: E402


def stats(x, y, line="S1", frm=1, to=2, condition="weekday peak") -> dict:
    """Sufficient statistics for one leg, the way prepare.build_legs writes them."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    return {
        "line": line, "from_stop": frm, "to_stop": to, "condition": condition,
        "from_name": f"stop{frm}", "to_name": f"stop{to}",
        "n": len(x), "sx": x.sum(), "sy": y.sum(),
        "sxy": float((x * y).sum()), "sxx": float((x * x).sum()), "syy": float((y * y).sum()),
        "booked_min": 3.0,
    }


def line_through(x, a, b, noise=0.0, seed=0):
    rng = np.random.default_rng(seed)
    return a * np.asarray(x, dtype=float) + b + rng.normal(0, noise, len(x))


def test_recovers_a_known_slope() -> None:
    x = np.arange(0, 600, 10, dtype=float)
    y = line_through(x, 0.8, 30.0)
    fitted = prop.fit(pd.DataFrame([stats(x, y)]))
    assert fitted.iloc[0]["a"] == pytest.approx(0.8, abs=1e-6)
    assert fitted.iloc[0]["b"] == pytest.approx(30.0, abs=1e-6)


def test_pooling_equals_refitting_on_the_combined_data() -> None:
    """The whole design rests on this.

    Pooling adds sufficient statistics together instead of going back to the observations. If
    that is not identical to fitting the combined data, every pooled coefficient is wrong.
    """
    rng = np.random.default_rng(1)
    xa, xb = rng.uniform(-60, 900, 400), rng.uniform(-60, 900, 250)
    ya, yb = line_through(xa, 0.9, 10, 40, seed=2), line_through(xb, 0.7, 25, 40, seed=3)

    legs = pd.DataFrame([
        stats(xa, ya, frm=1, to=2, condition="weekday peak"),
        stats(xb, yb, frm=2, to=3, condition="weekday peak"),
    ])
    pooled = prop.fit(legs, prop.Params(pooling="line", conditioned=False)).iloc[0]

    direct_a, direct_b = np.polyfit(np.concatenate([xa, xb]), np.concatenate([ya, yb]), 1)
    assert pooled["a"] == pytest.approx(direct_a, rel=1e-9)
    assert pooled["b"] == pytest.approx(direct_b, rel=1e-9)


def test_residual_spread_matches_a_direct_computation() -> None:
    rng = np.random.default_rng(4)
    x = rng.uniform(0, 600, 500)
    y = line_through(x, 0.85, 12, noise=50, seed=5)
    fitted = prop.fit(pd.DataFrame([stats(x, y)])).iloc[0]

    a, b = np.polyfit(x, y, 1)
    residual = y - (a * x + b)
    expected = np.sqrt((residual**2).sum() / (len(x) - 2))
    assert fitted["sigma"] == pytest.approx(expected, rel=1e-6)


def test_a_leg_with_no_spread_in_its_input_carries_the_delay_forward() -> None:
    """Every train arrived equally late, so the leg pins no slope. Carrying it on is honest."""
    fitted = prop.fit(pd.DataFrame([stats([120.0] * 50, [90.0] * 50)])).iloc[0]
    assert fitted["a"] == 1.0
    assert fitted["b"] == 0.0


def test_a_thin_leg_falls_back_to_its_line() -> None:
    rng = np.random.default_rng(6)
    fat_x = rng.uniform(0, 600, 500)
    legs = pd.DataFrame([
        stats(fat_x, line_through(fat_x, 0.6, 5, 30, seed=7), frm=1, to=2),
        stats([10.0, 400.0, 20.0, 380.0, 15.0, 390.0], [12.0, 390.0, 25.0, 370.0, 18.0, 380.0],
              frm=2, to=3),
    ])
    fitted = prop.fit(legs, prop.Params(min_obs=100, conditioned=False))
    levels = fitted["level"].tolist()
    assert "line" in levels, f"expected a fallback, got {levels}"
    assert fitted.loc[fitted["level"] == "leg", "n"].min() >= 100


def test_every_leg_gets_a_coefficient() -> None:
    rng = np.random.default_rng(8)
    x = rng.uniform(0, 600, 300)
    legs = pd.DataFrame([
        stats(x, line_through(x, 0.9, 4, 30, seed=9), frm=i, to=i + 1) for i in range(1, 6)
    ])
    for pooling in prop.POOLING:
        fitted = prop.fit(legs, prop.Params(pooling=pooling, conditioned=False))
        assert fitted["a"].notna().all(), f"{pooling} left a leg without a slope"


def route(n_legs: int) -> pd.DataFrame:
    stops = pd.DataFrame({
        "line": "S1", "bpuic": range(1, n_legs + 2),
        "station": [f"stop{i}" for i in range(1, n_legs + 2)],
        "position": range(1, n_legs + 2), "on_main_route": True,
    })
    return prop.route_legs(stops, "S1")


def coefficients(a: float, sigma: float, n_legs: int) -> pd.DataFrame:
    frame = route(n_legs).assign(a=a, b=0.0, sigma=sigma, n=500, level="leg")
    return frame.set_index(["line", "from_stop", "to_stop"])[["a", "b", "sigma", "n", "level"]]


def test_a_shedding_route_brings_the_delay_down() -> None:
    legs, coef = route(4), coefficients(0.8, 10.0, 4)
    out = prop.predict_route(legs, 600.0, coef)
    assert len(out) == 5
    assert out["delay_s"].iloc[-1] == pytest.approx(600.0 * 0.8**4)
    assert out["delay_s"].is_monotonic_decreasing


def test_the_interval_widens_with_every_leg() -> None:
    out = prop.predict_route(route(5), 300.0, coefficients(1.0, 20.0, 5))
    width = out["upper_s"] - out["lower_s"]
    assert width.iloc[0] == 0.0, "the starting delay is known, not predicted"
    assert width.is_monotonic_increasing
    assert width.iloc[-1] > width.iloc[1]


def test_an_unknown_leg_carries_the_delay_rather_than_guessing() -> None:
    coef = coefficients(0.5, 5.0, 3).iloc[:1]
    out = prop.predict_route(route(3), 400.0, coef)
    assert out["level"].tolist()[1:] == ["leg", "carried", "carried"]
    assert out["delay_s"].iloc[-1] == pytest.approx(400.0 * 0.5)


def test_an_empty_route_is_not_an_error() -> None:
    assert prop.predict_route(pd.DataFrame(), 300.0, coefficients(1.0, 1.0, 2)).empty


@pytest.mark.parametrize("bad", [{"pooling": "vibes"}, {"min_obs": 2}])
def test_nonsense_parameters_are_refused(bad: dict) -> None:
    with pytest.raises(ValueError):
        prop.Params(**bad)
