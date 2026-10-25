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

# How a leg turns the delay it receives into the delay it passes on. All three are fitted from
# the same sums, so switching between them is instant and the comparison is like for like.
#
#   fitted   delay_out = a * delay_in + b, both fitted
#   drift    delay_out = delay_in + b, the leg adds a fixed amount whatever arrives
#   hold     delay_out = delay_in, the persistence baseline the model has to beat
#
# Held out on February: fitted 22.2 s, hold 43.4 s. Drift sits between them.
METHODS = ("fitted", "drift", "hold")

# How many sigma the band actually needs to reach a given coverage, measured on 1136 predicted
# stops across both scenarios rather than taken from the normal distribution.
#
# The theory says 1.96 sigma covers 95%. Measured, it covers 72%, and reaching a true 95% takes
# about ten sigma. The residuals are nothing like normal: 21% of stops land beyond 3 sigma where
# a normal says 0.3%, and the worst is 52 sigma. Delay is event driven, so most stops sit close
# to the fit and a few are thrown a very long way by something the model cannot see.
#
# Fitted on winter and checked on summer, where the same multipliers hold to within a few
# percent, so these are not tuned to one season.
COVERAGE_MULTIPLIER = {0.5: 1.1, 0.68: 1.8, 0.8: 3.2, 0.9: 5.2, 0.95: 10.2}

# A fit needs at least this many points before a slope means anything at all, whatever the user
# asks for.
ABSOLUTE_MIN_N = 5


@dataclass(frozen=True)
class Params:
    """What the user steers. Every one of these refits the model."""

    pooling: str = "leg"
    method: str = "fitted"
    min_obs: int = 60
    conditioned: bool = True
    # How often the band should actually contain the truth. Not a sigma count, because the
    # residuals are too heavy tailed for sigma to mean what it usually means.
    coverage: float = 0.8

    def __post_init__(self) -> None:
        if self.pooling not in POOLING:
            raise ValueError(f"pooling must be one of {POOLING}")
        if self.method not in METHODS:
            raise ValueError(f"method must be one of {METHODS}")
        if self.min_obs < ABSOLUTE_MIN_N:
            raise ValueError(f"min_obs must be at least {ABSOLUTE_MIN_N}")
        if self.coverage not in COVERAGE_MULTIPLIER:
            raise ValueError(f"coverage must be one of {tuple(COVERAGE_MULTIPLIER)}")

    @property
    def band_sigma(self) -> float:
        """The multiplier that makes the band mean what it claims."""
        return COVERAGE_MULTIPLIER[self.coverage]


def _fit_from_stats(stats: pd.DataFrame, method: str = "fitted") -> pd.DataFrame:
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

    if method == "hold":
        a = np.ones(len(n))
        b = np.zeros(len(n))
    elif method == "drift":
        # The slope is pinned at one, so the only thing fitted is how much the leg adds.
        a = np.ones(len(n))
        b = ((sy - sx) / n).to_numpy()
    else:
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
    per_leg = _fit_from_stats(legs.groupby(group, observed=True)[STAT_COLUMNS].sum(),
                              params.method)

    line_group = ["line"] + (["condition"] if params.conditioned else [])
    per_line = _fit_from_stats(legs.groupby(line_group, observed=True)[STAT_COLUMNS].sum(),
                               params.method)
    overall = _fit_from_stats(legs[STAT_COLUMNS].sum().to_frame().T, params.method)

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
    # A leg the fit never saw carries the delay on, which is the honest point estimate. Giving
    # it no scatter was not honest: the band then read as certain exactly where the model knows
    # least. It gets the typical leg's scatter instead.
    unknown_sigma = (
        float(np.nanmedian(coefficients["sigma"])) if len(coefficients) else 0.0
    )
    rows = [{
        "station": route.iloc[0]["from_name"], "bpuic": route.iloc[0]["from_stop"],
        "delay_s": delay, "lower_s": delay, "upper_s": delay, "a": np.nan, "level": "start",
    }]

    for leg in route.itertuples(index=False):
        key = tuple(getattr(leg, k) for k in coefficients.index.names)
        if key not in coefficients.index:
            a, b, sigma, level = 1.0, 0.0, unknown_sigma, "carried"
        else:
            row = coefficients.loc[key]
            a, b, sigma, level = row["a"], row["b"], row["sigma"], row["level"]

        delay = a * delay + b
        variance = a * a * variance + (0.0 if np.isnan(sigma) else sigma**2)
        spread = params.band_sigma * np.sqrt(variance)
        rows.append({
            "station": leg.to_name, "bpuic": leg.to_stop, "delay_s": delay,
            "lower_s": delay - spread, "upper_s": delay + spread, "a": a, "level": level,
        })

    return pd.DataFrame(rows)


def route_legs(
    stop_order: pd.DataFrame,
    line: str,
    condition: str | None = None,
    stations: set | None = None,
) -> pd.DataFrame:
    """The ordered legs of a route, ready to hand to predict_route.

    Without `stations` this is the line's main route. With it, the route runs through those
    stops instead, which is what an episode needs when the line code covers more than one
    railway and the episode is not on the busiest of them.
    """
    on_line = stop_order["line"] == line
    keep = stop_order["station"].isin(stations) if stations else stop_order["on_main_route"]
    stops = stop_order[on_line & keep].sort_values("position").reset_index(drop=True)
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
