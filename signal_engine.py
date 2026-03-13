from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import pandas as pd


class EngineMode(str, Enum):
    COPILOT = "COPILOT"
    AUTO_SAFE = "AUTO_SAFE"


@dataclass(frozen=True)
class ModeThresholds:
    zscore_entry: float
    max_realized_volatility: float
    min_quote_volume: float
    min_spread_stability: float
    max_abs_funding: float
    fee_bps: float
    top_n: int


MODE_THRESHOLDS: dict[EngineMode, ModeThresholds] = {
    EngineMode.COPILOT: ModeThresholds(
        zscore_entry=1.75,
        max_realized_volatility=1.40,
        min_quote_volume=2_500_000.0,
        min_spread_stability=0.45,
        max_abs_funding=0.0015,
        fee_bps=8.0,
        top_n=5,
    ),
    EngineMode.AUTO_SAFE: ModeThresholds(
        zscore_entry=2.40,
        max_realized_volatility=0.95,
        min_quote_volume=7_500_000.0,
        min_spread_stability=0.65,
        max_abs_funding=0.0008,
        fee_bps=10.0,
        top_n=5,
    ),
}


def _confidence_score(row: pd.Series, thresholds: ModeThresholds) -> float:
    z_component = min(abs(row["z_score"]) / (thresholds.zscore_entry * 1.5), 1.0)
    stability_component = min(max(row["spread_stability_score"], 0.0), 1.0)
    vol_component = 1.0 - min(row["realized_volatility"] / max(thresholds.max_realized_volatility, 1e-9), 1.0)
    edge_component = min(max(row["edge_after_fees_pct"] / 3.0, 0.0), 1.0)
    return float((0.35 * z_component + 0.25 * stability_component + 0.20 * vol_component + 0.20 * edge_component) * 100.0)


def _direction_from_deviation(deviation_pct: float) -> str:
    return "LONG" if deviation_pct < 0 else "SHORT"


def _market_relative_signal_quality(row: pd.Series) -> str:
    market_score = float(row["market_opportunity_score"])
    if market_score >= 92:
        return "A1"
    if market_score >= 84:
        return "A2"
    if market_score >= 76:
        return "B1"
    if market_score >= 68:
        return "B2"
    if market_score >= 60:
        return "C1"
    if market_score >= 50:
        return "C2"
    if market_score >= 40:
        return "D1"
    return "D2"


def build_ranked_universe(features_df: pd.DataFrame, mode: EngineMode) -> pd.DataFrame:
    thresholds = MODE_THRESHOLDS[mode]
    if features_df.empty:
        return pd.DataFrame()

    latest = features_df.sort_values("timestamp").groupby("symbol", as_index=False).tail(1).copy()
    latest = latest.dropna(subset=["fair_value", "z_score", "realized_volatility", "spread_stability_score"])
    if latest.empty:
        return latest

    latest["abs_z_score"] = latest["z_score"].abs()
    latest["edge_after_fees_pct"] = latest["deviation_pct"].abs() - (thresholds.fee_bps / 100.0)
    latest["suggested_direction"] = latest["deviation_pct"].apply(_direction_from_deviation)
    latest["suggested_entry"] = latest["alt_price"]

    passes = (
        (latest["abs_z_score"] >= thresholds.zscore_entry)
        & (latest["realized_volatility"] <= thresholds.max_realized_volatility)
        & (latest["quote_volume"] >= thresholds.min_quote_volume)
        & (latest["spread_stability_score"] >= thresholds.min_spread_stability)
    )

    if "funding_rate" in latest.columns:
        latest["funding_rate"] = latest["funding_rate"].fillna(0.0)
        passes &= latest["funding_rate"].abs() <= thresholds.max_abs_funding
    else:
        latest["funding_rate"] = 0.0

    latest["confidence_score"] = latest.apply(_confidence_score, axis=1, thresholds=thresholds)
    latest["passes_filters"] = passes
    latest["confidence_rank_pct"] = latest["confidence_score"].rank(pct=True, method="average") * 100.0
    latest["edge_rank_pct"] = latest["edge_after_fees_pct"].rank(pct=True, method="average") * 100.0
    latest["z_rank_pct"] = latest["abs_z_score"].rank(pct=True, method="average") * 100.0
    latest["stability_rank_pct"] = latest["spread_stability_score"].rank(pct=True, method="average") * 100.0
    latest["liquidity_rank_pct"] = latest["quote_volume"].rank(pct=True, method="average") * 100.0
    latest["vol_rank_pct"] = (1.0 - latest["realized_volatility"].rank(pct=True, method="average")).clip(lower=0.0) * 100.0
    latest["market_opportunity_score"] = (
        0.28 * latest["confidence_rank_pct"]
        + 0.22 * latest["edge_rank_pct"]
        + 0.18 * latest["z_rank_pct"]
        + 0.12 * latest["stability_rank_pct"]
        + 0.10 * latest["liquidity_rank_pct"]
        + 0.10 * latest["vol_rank_pct"]
    )
    latest.loc[latest["passes_filters"], "market_opportunity_score"] += 5.0
    latest["market_opportunity_score"] = latest["market_opportunity_score"].clip(upper=100.0)
    latest["signal_quality"] = latest.apply(_market_relative_signal_quality, axis=1)
    latest = latest.sort_values(
        by=["market_opportunity_score", "confidence_score", "edge_after_fees_pct"],
        ascending=[False, False, False],
    )

    return latest.rename(
        columns={
            "alt_price": "current_price",
            "fair_value": "expected_fair_value",
        }
    )[
        [
            "timestamp",
            "symbol",
            "current_price",
            "expected_fair_value",
            "deviation_pct",
            "z_score",
            "spread_stability_score",
            "suggested_direction",
            "suggested_entry",
            "quote_volume",
            "realized_volatility",
            "funding_rate",
            "edge_after_fees_pct",
            "confidence_score",
            "market_opportunity_score",
            "signal_quality",
            "passes_filters",
        ]
    ]


def build_opportunity_table(features_df: pd.DataFrame, mode: EngineMode) -> pd.DataFrame:
    thresholds = MODE_THRESHOLDS[mode]
    ranked = build_ranked_universe(features_df, mode)
    if ranked.empty:
        return ranked
    return ranked.loc[ranked["passes_filters"]].head(thresholds.top_n).copy()
