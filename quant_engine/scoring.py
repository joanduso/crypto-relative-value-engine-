from __future__ import annotations

import pandas as pd


def normalize_series(series: pd.Series, *, invert: bool = False) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce").fillna(0.0)
    minimum = float(values.min()) if not values.empty else 0.0
    maximum = float(values.max()) if not values.empty else 0.0

    if maximum == minimum:
        normalized = pd.Series(50.0, index=values.index, dtype="float64")
    else:
        normalized = (values - minimum) / (maximum - minimum) * 100.0

    if invert:
        normalized = 100.0 - normalized
    return normalized.clip(lower=0.0, upper=100.0)


def build_opportunity_score_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()

    scored = frame.copy()
    scored["base_score_normalized"] = normalize_series(scored["base_score"])
    scored["volatility_score"] = normalize_series(scored["realized_volatility"])
    scored["liquidity_score"] = normalize_series(scored["quote_volume"])
    scored["opportunity_score"] = (
        0.4 * scored["base_score_normalized"]
        + 0.3 * scored["alignment_score"]
        + 0.2 * scored["volatility_score"]
        + 0.1 * scored["liquidity_score"]
    ).round(2)
    return scored
