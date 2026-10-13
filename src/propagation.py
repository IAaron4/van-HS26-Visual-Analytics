"""How a delay carries along a route, fitted from the data.

For each leg of a line, a straight line between the delay a train arrives with and the delay it
leaves with:

    delay_out = a * delay_in + b

`a` is how much the leg carries forward. Below one it sheds delay, above one it amplifies. It
averages about 0.91 and tracks the slack in the timetable, so a leg booked ten minutes sheds far
more than one booked two.

Nothing here imports Panel, so the notebooks, the tests and the app fit the same way.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

LEG_KEYS = ["line", "from_stop", "to_stop"]

# The columns prepare.build_legs writes, enough to fit a least squares line without the rows.
STAT_COLUMNS = ["n", "sx", "sy", "sxy", "sxx", "syy"]

POOLING = ("leg", "line", "global")

# A fit needs at least this many points before a slope means anything at all, whatever the user
# asks for.
ABSOLUTE_MIN_N = 5


@dataclass(frozen=True)
class Params:
    """What the user steers. Every one of these refits the model."""

    pooling: str = "leg"
    min_obs: int = 60
    conditioned: bool = True
    interval_sigma: float = 1.96

    def __post_init__(self) -> None:
        if self.pooling not in POOLING:
            raise ValueError(f"pooling must be one of {POOLING}")
        if self.min_obs < ABSOLUTE_MIN_N:
            raise ValueError(f"min_obs must be at least {ABSOLUTE_MIN_N}")


def _fit_from_stats(stats: pd.DataFrame) -> pd.DataFrame:
    """Least squares slope, intercept and residual spread from sufficient statistics.

    The five sums carry everything a straight line fit needs, so pooling groups together is
    adding rows up rather than going back to eleven million observations.
    """
    n, sx, sy = stats["n"], stats["sx"], stats["sy"]
    sxy, sxx, syy = stats["sxy"], stats["sxx"], stats["syy"]

    var_x = n * sxx - sx * sx
    # A leg whose arriving delays are all identical pins no slope at all. Carrying the delay
    # forward unchanged is the honest answer there, not a slope fitted to a single point.
    degenerate = (var_x <= 0) | (n < ABSOLUTE_MIN_N)

    a = np.where(degenerate, 1.0, (n * sxy - sx * sy) / var_x.where(var_x > 0, 1.0))
    b = np.where(degenerate, 0.0, (sy - a * sx) / n)

    # Residual sum of squares, straight from the sums.
    rss = syy - b * sy - a * sxy
    dof = (n - 2).clip(lower=1)
    sigma = np.sqrt(np.maximum(rss, 0.0) / dof)

    return pd.DataFrame(
        {"a": a, "b": b, "sigma": np.where(degenerate, np.nan, sigma), "n": n.to_numpy()},
        index=stats.index,
    )


def fit(legs: pd.DataFrame, params: Params = Params()) -> pd.DataFrame:
    """Fit every leg, pooling as asked and falling back when a leg is too thin.

    Pooling is the bias and variance control the user holds. Per leg is the most faithful and the
    noisiest. Per line is steadier and blurs the difference between a leg with slack and one
    without. One global coefficient is stable and crude.

    A leg below `min_obs` does not get its own slope. It inherits the line's, and the line falls
    back to global, so every leg always has a usable coefficient and the fallback is recorded.
    """
    group = LEG_KEYS + (["condition"] if params.conditioned else [])
    per_leg = _fit_from_stats(legs.groupby(group, observed=True)[STAT_COLUMNS].sum())

    line_group = ["line"] + (["condition"] if params.conditioned else [])
    per_line = _fit_from_stats(legs.groupby(line_group, observed=True)[STAT_COLUMNS].sum())
    overall = _fit_from_stats(legs[STAT_COLUMNS].sum().to_frame().T)

    if params.pooling == "global":
        chosen = per_leg.assign(**_broadcast(overall.iloc[0]), level="global")
        return chosen[["a", "b", "sigma", "n", "level"]]

    out = per_leg.reset_index()
    if params.pooling == "line":
        source = per_line.reset_index()
        merged = out[group].merge(source, on=line_group, how="left")
        merged["level"] = "line"
        return _finish(merged, out[group], overall)

    thin = out["n"] < params.min_obs
    fallback = out.loc[thin, group].merge(per_line.reset_index(), on=line_group, how="left")
    out["level"] = np.where(thin, "line", "leg")
    for column in ("a", "b", "sigma", "n"):
        out.loc[thin, column] = fallback[column].to_numpy()
    return _finish(out, out[group], overall)


def _broadcast(row: pd.Series) -> dict:
    return {"a": row["a"], "b": row["b"], "sigma": row["sigma"], "n": row["n"]}


def _finish(frame: pd.DataFrame, keys: pd.DataFrame, overall: pd.DataFrame) -> pd.DataFrame:
    """Fill anything still missing from the global fit, so no leg is left without a coefficient."""
    out = frame.copy()
    missing = out["a"].isna()
    if missing.any():
        row = overall.iloc[0]
        for column in ("a", "b", "sigma", "n"):
            out.loc[missing, column] = row[column]
        out.loc[missing, "level"] = "global"
    out["sigma"] = out["sigma"].fillna(overall.iloc[0]["sigma"])
    return out.set_index(list(keys.columns))[["a", "b", "sigma", "n", "level"]]


def predict_route(
    route: pd.DataFrame,
    start_delay_s: float,
    coefficients: pd.DataFrame,
    params: Params = Params(),
) -> pd.DataFrame:
    """Walk a delay along the remaining stops, carrying its uncertainty with it.

    Each leg multiplies the delay by its own coefficient and adds its own scatter, so the
    interval widens stop by stop. That widening is the useful part: a prediction five stops ahead
    deserves less confidence than one stop ahead, and the chart should say so rather than drawing
    one confident line all the way to the terminus.

    `route` is the ordered legs of the journey, with the same keys the coefficients carry.
    """
    if route.empty:
        return pd.DataFrame(
            columns=["station", "bpuic", "delay_s", "lower_s", "upper_s", "a", "level"]
        )

    delay = float(start_delay_s)
    variance = 0.0
    rows = [{
        "station": route.iloc[0]["from_name"], "bpuic": route.iloc[0]["from_stop"],
        "delay_s": delay, "lower_s": delay, "upper_s": delay, "a": np.nan, "level": "start",
    }]

    for leg in route.itertuples(index=False):
        key = tuple(getattr(leg, k) for k in coefficients.index.names)
        if key not in coefficients.index:
            a, b, sigma, level = 1.0, 0.0, float("nan"), "carried"
        else:
            row = coefficients.loc[key]
            a, b, sigma, level = row["a"], row["b"], row["sigma"], row["level"]

        delay = a * delay + b
        variance = a * a * variance + (0.0 if np.isnan(sigma) else sigma**2)
        spread = params.interval_sigma * np.sqrt(variance)
        rows.append({
            "station": leg.to_name, "bpuic": leg.to_stop, "delay_s": delay,
            "lower_s": delay - spread, "upper_s": delay + spread, "a": a, "level": level,
        })

    return pd.DataFrame(rows)


def route_legs(stop_order: pd.DataFrame, line: str, condition: str | None = None) -> pd.DataFrame:
    """The ordered legs of a line's main route, ready to hand to predict_route."""
    stops = (
        stop_order[(stop_order["line"] == line) & stop_order["on_main_route"]]
        .sort_values("position")
        .reset_index(drop=True)
    )
    if len(stops) < 2:
        return pd.DataFrame(columns=["line", "from_stop", "to_stop", "from_name", "to_name"])

    legs = pd.DataFrame({
        "line": line,
        "from_stop": stops["bpuic"].to_numpy()[:-1],
        "to_stop": stops["bpuic"].to_numpy()[1:],
        "from_name": stops["station"].to_numpy()[:-1],
        "to_name": stops["station"].to_numpy()[1:],
    })
    if condition is not None:
        legs["condition"] = condition
    return legs
