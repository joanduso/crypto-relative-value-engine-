from __future__ import annotations

import os
from dataclasses import dataclass
from enum import Enum

import pandas as pd


class EngineMode(str, Enum):
    COPILOT = "COPILOT"
    COPILOT_RELAXED = "COPILOT_RELAXED"
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
    EngineMode.COPILOT_RELAXED: ModeThresholds(
        zscore_entry=1.10,
        max_realized_volatility=1.75,
        min_quote_volume=750_000.0,
        min_spread_stability=0.35,
        max_abs_funding=0.0025,
        fee_bps=8.0,
        top_n=8,
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


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def resolve_mode_thresholds(mode: EngineMode) -> ModeThresholds:
    base = MODE_THRESHOLDS[mode]
    prefix = f"ENGINE_{mode.value}_"
    return ModeThresholds(
        zscore_entry=_env_float(f"{prefix}ZSCORE_ENTRY", _env_float("ENGINE_ZSCORE_ENTRY", base.zscore_entry)),
        max_realized_volatility=_env_float(
            f"{prefix}MAX_REALIZED_VOLATILITY",
            _env_float("ENGINE_MAX_REALIZED_VOLATILITY", base.max_realized_volatility),
        ),
        min_quote_volume=_env_float(
            f"{prefix}MIN_QUOTE_VOLUME",
            _env_float("ENGINE_MIN_QUOTE_VOLUME", base.min_quote_volume),
        ),
        min_spread_stability=_env_float(
            f"{prefix}MIN_SPREAD_STABILITY",
            _env_float("ENGINE_MIN_SPREAD_STABILITY", base.min_spread_stability),
        ),
        max_abs_funding=_env_float(
            f"{prefix}MAX_ABS_FUNDING",
            _env_float("ENGINE_MAX_ABS_FUNDING", base.max_abs_funding),
        ),
        fee_bps=_env_float(f"{prefix}FEE_BPS", _env_float("ENGINE_FEE_BPS", base.fee_bps)),
        top_n=_env_int(f"{prefix}TOP_N", _env_int("ENGINE_TOP_N", base.top_n)),
    )


def _confidence_score(row: pd.Series, thresholds: ModeThresholds) -> float:
    z_component = min(abs(row["z_score"]) / (thresholds.zscore_entry * 1.5), 1.0)
    stability_component = min(max(row["spread_stability_score"], 0.0), 1.0)
    vol_component = 1.0 - min(row["realized_volatility"] / max(thresholds.max_realized_volatility, 1e-9), 1.0)
    edge_component = min(max(row["edge_after_fees_pct"] / 3.0, 0.0), 1.0)
    return float((0.35 * z_component + 0.25 * stability_component + 0.20 * vol_component + 0.20 * edge_component) * 100.0)


def _direction_from_deviation(deviation_pct: float) -> str:
    return "LONG" if deviation_pct < 0 else "SHORT"


def market_score_to_quality(market_score: float) -> str:
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


def _market_relative_signal_quality(row: pd.Series) -> str:
    market_score = float(row["market_opportunity_score"])
    return market_score_to_quality(market_score)


def prepare_signal_frame(features_df: pd.DataFrame, mode: EngineMode) -> pd.DataFrame:
    thresholds = resolve_mode_thresholds(mode)
    if features_df.empty:
        return pd.DataFrame()

    prepared = features_df.copy()
    prepared = prepared.dropna(subset=["fair_value", "z_score", "realized_volatility", "spread_stability_score"])
    if prepared.empty:
        return prepared

    prepared["abs_z_score"] = prepared["z_score"].abs()
    prepared["edge_after_fees_pct"] = prepared["deviation_pct"].abs() - (thresholds.fee_bps / 100.0)
    prepared["suggested_direction"] = prepared["deviation_pct"].apply(_direction_from_deviation)
    prepared["suggested_entry"] = prepared["alt_price"]

    prepared["passes_zscore_filter"] = prepared["abs_z_score"] >= thresholds.zscore_entry
    prepared["passes_volatility_filter"] = prepared["realized_volatility"] <= thresholds.max_realized_volatility
    prepared["passes_liquidity_filter"] = prepared["quote_volume"] >= thresholds.min_quote_volume
    prepared["passes_stability_filter"] = prepared["spread_stability_score"] >= thresholds.min_spread_stability

    if "funding_rate" in prepared.columns:
        prepared["funding_rate"] = prepared["funding_rate"].fillna(0.0)
        prepared["passes_funding_filter"] = prepared["funding_rate"].abs() <= thresholds.max_abs_funding
    else:
        prepared["funding_rate"] = 0.0
        prepared["passes_funding_filter"] = True

    passes = (
        prepared["passes_zscore_filter"]
        & prepared["passes_volatility_filter"]
        & prepared["passes_liquidity_filter"]
        & prepared["passes_stability_filter"]
        & prepared["passes_funding_filter"]
    )

    prepared["confidence_score"] = prepared.apply(_confidence_score, axis=1, thresholds=thresholds)
    prepared["passes_filters"] = passes
    failure_reasons: list[str] = []
    for row in prepared.itertuples(index=False):
        reasons: list[str] = []
        if not row.passes_zscore_filter:
            reasons.append("zscore")
        if not row.passes_volatility_filter:
            reasons.append("volatility")
        if not row.passes_liquidity_filter:
            reasons.append("liquidity")
        if not row.passes_stability_filter:
            reasons.append("stability")
        if not row.passes_funding_filter:
            reasons.append("funding")
        failure_reasons.append("OK" if not reasons else ",".join(reasons))
    prepared["filter_failure_reason"] = failure_reasons
    return prepared


def build_ranked_universe(features_df: pd.DataFrame, mode: EngineMode) -> pd.DataFrame:
    if features_df.empty:
        return pd.DataFrame()

    latest = features_df.sort_values("timestamp").groupby("symbol", as_index=False).tail(1).copy()
    latest = prepare_signal_frame(latest, mode)
    if latest.empty:
        return latest

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
            "alt_open",
            "alt_high",
            "alt_low",
            "current_price",
            "expected_fair_value",
            "deviation_pct",
            "z_score",
            "spread_stability_score",
            "alt_ema_20",
            "alt_ema_50",
            "alt_rsi_14",
            "alt_ema_20_slope_pct",
            "reclaim_ema_20",
            "breakout_5_pct",
            "higher_low_3",
            "short_term_momentum_3",
            "micro_reversal_score",
            "suggested_direction",
            "suggested_entry",
            "quote_volume",
            "realized_volatility",
            "funding_rate",
            "edge_after_fees_pct",
            "confidence_score",
            "passes_zscore_filter",
            "passes_volatility_filter",
            "passes_liquidity_filter",
            "passes_stability_filter",
            "passes_funding_filter",
            "filter_failure_reason",
            "market_opportunity_score",
            "signal_quality",
            "passes_filters",
        ]
    ]


def build_opportunity_table(features_df: pd.DataFrame, mode: EngineMode) -> pd.DataFrame:
    thresholds = resolve_mode_thresholds(mode)
    ranked = build_ranked_universe(features_df, mode)
    if ranked.empty:
        return ranked
    return ranked.loc[ranked["passes_filters"]].head(thresholds.top_n).copy()


def build_opportunity_table_from_ranked(ranked: pd.DataFrame, mode: EngineMode) -> pd.DataFrame:
    thresholds = resolve_mode_thresholds(mode)
    if ranked.empty:
        return ranked.copy()
    return ranked.loc[ranked["passes_filters"]].head(thresholds.top_n).copy()
