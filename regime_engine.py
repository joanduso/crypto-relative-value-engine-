from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from signal_engine import EngineMode, market_score_to_quality


@dataclass(frozen=True)
class RegimeState:
    btc_regime: str
    btc_directional_score: float
    btc_price_vs_ema200_pct: float
    btc_ema50_vs_ema200_pct: float
    btc_ema200_slope_pct: float
    breadth_deterioration_pct: float
    breadth_stress_score: float
    return_dispersion_pct: float
    avg_alt_correlation: float


def _quality_from_market_score(market_score: float) -> str:
    return market_score_to_quality(market_score)


def _btc_regime_state(market_df: pd.DataFrame) -> RegimeState:
    btc = market_df.loc[market_df["symbol"] == "BTCUSDT", ["timestamp", "close"]].copy()
    if btc.empty:
        return RegimeState("range", 0.0, 0.0, 0.0, 0.0, 50.0, 50.0, 0.0, 0.0)

    btc["timestamp"] = pd.to_datetime(btc["timestamp"], utc=True, errors="coerce")
    btc = btc.dropna(subset=["timestamp"]).sort_values("timestamp")
    btc_4h = (
        btc.set_index("timestamp")["close"]
        .resample("4h")
        .last()
        .dropna()
        .to_frame(name="close")
        .reset_index()
    )
    if len(btc_4h) < 210:
        return RegimeState("range", 0.0, 0.0, 0.0, 0.0, 50.0, 50.0, 0.0, 0.0)

    close = pd.to_numeric(btc_4h["close"], errors="coerce")
    ema50 = close.ewm(span=50, adjust=False).mean()
    ema200 = close.ewm(span=200, adjust=False).mean()
    ema50_slope = ema50.pct_change(3) * 100.0
    ema200_slope = ema200.pct_change(3) * 100.0

    latest_price = float(close.iloc[-1])
    latest_ema50 = float(ema50.iloc[-1])
    latest_ema200 = float(ema200.iloc[-1])
    price_vs_ema200_pct = (latest_price / max(latest_ema200, 1e-9) - 1.0) * 100.0
    ema50_vs_ema200_pct = (latest_ema50 / max(latest_ema200, 1e-9) - 1.0) * 100.0
    ema200_slope_pct = float(ema200_slope.iloc[-1]) if pd.notna(ema200_slope.iloc[-1]) else 0.0
    ema50_slope_pct = float(ema50_slope.iloc[-1]) if pd.notna(ema50_slope.iloc[-1]) else 0.0

    directional_score = 0.0
    directional_score += 35.0 if latest_price > latest_ema200 else -35.0
    directional_score += 25.0 if latest_ema50 > latest_ema200 else -25.0
    directional_score += 20.0 if ema50_slope_pct > 0 else -20.0
    directional_score += 20.0 if ema200_slope_pct > 0 else -20.0

    if directional_score >= 35.0 and latest_ema50 > latest_ema200 and ema200_slope_pct > 0:
        regime = "trending_bullish"
    elif directional_score <= -35.0 and latest_ema50 < latest_ema200 and ema200_slope_pct < 0:
        regime = "trending_bearish"
    else:
        regime = "range"

    alt_prices = (
        market_df.loc[~market_df["symbol"].isin(["BTCUSDT", "ETHUSDT"]), ["timestamp", "symbol", "close"]]
        .pivot(index="timestamp", columns="symbol", values="close")
        .sort_index()
    )
    if alt_prices.empty:
        breadth_deterioration_pct = 50.0
        breadth_stress_score = 50.0
        return_dispersion_pct = 0.0
        avg_alt_correlation = 0.0
    else:
        horizon = min(12, max(len(alt_prices) - 1, 1))
        trailing_returns = alt_prices.pct_change(periods=horizon).iloc[-1].dropna()
        ema20 = alt_prices.ewm(span=20, adjust=False).mean().iloc[-1]
        last_prices = alt_prices.iloc[-1]
        below_ema_pct = float((last_prices < ema20).mean() * 100.0) if not ema20.empty else 50.0
        short_momentum = alt_prices.pct_change(periods=3).iloc[-1].dropna()
        short_momentum_pct = float((short_momentum < 0).mean() * 100.0) if not short_momentum.empty else 50.0
        if trailing_returns.empty:
            breadth_deterioration_pct = 50.0
            breadth_stress_score = 50.0
            return_dispersion_pct = 0.0
        else:
            breadth_deterioration_pct = float((trailing_returns < 0).mean() * 100.0)
            return_dispersion_pct = float(trailing_returns.std(ddof=0) * 100.0)
            breadth_stress_score = (
                0.45 * breadth_deterioration_pct
                + 0.35 * below_ema_pct
                + 0.20 * short_momentum_pct
            )

        corr_window = min(24, len(alt_prices))
        corr_source = alt_prices.pct_change().tail(corr_window).dropna(axis=1, how="all")
        if corr_source.shape[1] < 2 or corr_source.empty:
            avg_alt_correlation = 0.0
        else:
            corr = corr_source.corr()
            mask = np.triu(np.ones(corr.shape), k=1).astype(bool)
            values = corr.where(mask).stack()
            avg_alt_correlation = float(values.mean()) if not values.empty else 0.0

    return RegimeState(
        btc_regime=regime,
        btc_directional_score=round(directional_score, 2),
        btc_price_vs_ema200_pct=round(price_vs_ema200_pct, 4),
        btc_ema50_vs_ema200_pct=round(ema50_vs_ema200_pct, 4),
        btc_ema200_slope_pct=round(ema200_slope_pct, 4),
        breadth_deterioration_pct=round(breadth_deterioration_pct, 2),
        breadth_stress_score=round(breadth_stress_score, 2),
        return_dispersion_pct=round(return_dispersion_pct, 4),
        avg_alt_correlation=round(avg_alt_correlation, 4),
    )


def apply_regime_overlay(
    ranked_universe: pd.DataFrame,
    market_df: pd.DataFrame,
    mode: EngineMode = EngineMode.COPILOT,
) -> tuple[pd.DataFrame, RegimeState]:
    regime_state = _btc_regime_state(market_df)
    if ranked_universe.empty:
        return ranked_universe.copy(), regime_state

    out = ranked_universe.copy()
    out["btc_regime"] = regime_state.btc_regime
    out["btc_directional_score"] = regime_state.btc_directional_score
    out["btc_price_vs_ema200_pct"] = regime_state.btc_price_vs_ema200_pct
    out["btc_ema50_vs_ema200_pct"] = regime_state.btc_ema50_vs_ema200_pct
    out["btc_ema200_slope_pct"] = regime_state.btc_ema200_slope_pct
    out["breadth_deterioration_pct"] = regime_state.breadth_deterioration_pct
    out["breadth_stress_score"] = regime_state.breadth_stress_score
    out["return_dispersion_pct"] = regime_state.return_dispersion_pct
    out["avg_alt_correlation"] = regime_state.avg_alt_correlation

    long_mask = out["suggested_direction"].astype(str).str.upper() == "LONG"
    micro_reversal = pd.to_numeric(out.get("micro_reversal_score"), errors="coerce").fillna(0.0)
    alt_rsi = pd.to_numeric(out.get("alt_rsi_14"), errors="coerce").fillna(50.0)
    ema20_slope = pd.to_numeric(out.get("alt_ema_20_slope_pct"), errors="coerce").fillna(0.0)
    breakout_5 = pd.to_numeric(out.get("breakout_5_pct"), errors="coerce").fillna(0.0)
    higher_low = out.get("higher_low_3", False)
    higher_low = pd.Series(higher_low).fillna(False).astype(bool)
    market_stress = (
        regime_state.breadth_stress_score >= 67.5
        and regime_state.avg_alt_correlation >= 0.55
    )
    hard_veto_long = (
        long_mask
        & (regime_state.btc_regime == "trending_bearish")
        & (regime_state.btc_directional_score <= -35.0)
        & market_stress
        & (regime_state.btc_price_vs_ema200_pct <= -1.0)
    )

    validation_score = (
        (micro_reversal / 8.0)
        + (alt_rsi >= 50.0).astype(float)
        + (ema20_slope > 0).astype(float)
        + (breakout_5 > -0.15).astype(float)
        + higher_low.astype(float)
    )

    required_confirmation = 1.5
    if regime_state.btc_regime == "range":
        required_confirmation = 2.5
    elif regime_state.btc_regime == "trending_bearish":
        required_confirmation = 3.25

    entry_validation_passed = (~long_mask) | (validation_score >= required_confirmation)
    if mode is EngineMode.COPILOT:
        regime_filter_passed = ~hard_veto_long
    else:
        regime_filter_passed = (~hard_veto_long) & entry_validation_passed

    size_multiplier = np.ones(len(out), dtype=float)
    confidence_penalty = np.zeros(len(out), dtype=float)
    exposure_cap = np.full(len(out), 1.0, dtype=float)
    validation_buffer = validation_score - required_confirmation

    if regime_state.btc_regime == "range":
        strong_range_long = long_mask & (validation_buffer >= 1.0)
        marginal_range_long = long_mask & (validation_buffer >= 0.0) & (validation_buffer < 1.0)
        weak_range_long = long_mask & (validation_buffer < 0.0)

        if mode is EngineMode.COPILOT:
            size_multiplier = np.where(strong_range_long, 1.0, size_multiplier)
            size_multiplier = np.where(marginal_range_long, 1.0, size_multiplier)
            size_multiplier = np.where(weak_range_long, 0.0, size_multiplier)

            confidence_penalty = np.where(strong_range_long | marginal_range_long, 0.0, confidence_penalty)
            confidence_penalty = np.where(weak_range_long, 4.0, confidence_penalty)

            exposure_cap = np.where(strong_range_long | marginal_range_long, 1.0, exposure_cap)
            exposure_cap = np.where(weak_range_long, 0.0, exposure_cap)
        else:
            size_multiplier = np.where(strong_range_long, 1.0, size_multiplier)
            size_multiplier = np.where(marginal_range_long, 0.85, size_multiplier)
            size_multiplier = np.where(weak_range_long, 0.0, size_multiplier)

            confidence_penalty = np.where(strong_range_long, 0.0, confidence_penalty)
            confidence_penalty = np.where(marginal_range_long, 3.0, confidence_penalty)
            confidence_penalty = np.where(weak_range_long, 8.0, confidence_penalty)

            exposure_cap = np.where(strong_range_long, 1.0, exposure_cap)
            exposure_cap = np.where(marginal_range_long, 0.85, exposure_cap)
            exposure_cap = np.where(weak_range_long, 0.0, exposure_cap)
    elif regime_state.btc_regime == "trending_bearish":
        elite_bear_long = (
            long_mask
            & (validation_buffer >= 1.25)
            & (regime_state.breadth_stress_score < 75.0)
            & (regime_state.btc_price_vs_ema200_pct > -0.75)
        )
        good_bear_long = long_mask & (validation_buffer >= 0.5) & ~elite_bear_long
        weak_bear_long = long_mask & ~elite_bear_long & ~good_bear_long

        if mode is EngineMode.COPILOT:
            size_multiplier = np.where(elite_bear_long, 1.0, size_multiplier)
            size_multiplier = np.where(good_bear_long, 0.8, size_multiplier)
            size_multiplier = np.where(weak_bear_long, 0.35, size_multiplier)

            confidence_penalty = np.where(elite_bear_long, 0.0, confidence_penalty)
            confidence_penalty = np.where(good_bear_long, 4.0, confidence_penalty)
            confidence_penalty = np.where(weak_bear_long, 10.0, confidence_penalty)

            exposure_cap = np.where(elite_bear_long, 1.0, exposure_cap)
            exposure_cap = np.where(good_bear_long, 0.8, exposure_cap)
            exposure_cap = np.where(weak_bear_long, 0.35, exposure_cap)
        else:
            size_multiplier = np.where(elite_bear_long, 0.8, size_multiplier)
            size_multiplier = np.where(good_bear_long, 0.55, size_multiplier)
            size_multiplier = np.where(weak_bear_long, 0.0, size_multiplier)

            confidence_penalty = np.where(elite_bear_long, 4.0, confidence_penalty)
            confidence_penalty = np.where(good_bear_long, 9.0, confidence_penalty)
            confidence_penalty = np.where(weak_bear_long, 16.0, confidence_penalty)

            exposure_cap = np.where(elite_bear_long, 0.8, exposure_cap)
            exposure_cap = np.where(good_bear_long, 0.55, exposure_cap)
            exposure_cap = np.where(weak_bear_long, 0.0, exposure_cap)

    size_multiplier = np.where(hard_veto_long, 0.0, size_multiplier)
    exposure_cap = np.where(hard_veto_long, 0.0, exposure_cap)

    out["regime_hard_veto_long"] = hard_veto_long
    out["entry_validation_passed"] = entry_validation_passed
    out["entry_validation_score"] = np.round(validation_score, 4)
    out["regime_filter_passed"] = regime_filter_passed
    out["regime_position_size_multiplier"] = np.round(size_multiplier, 4)
    out["regime_exposure_cap_pct"] = np.round(exposure_cap * 100.0, 2)
    out["regime_confirmation_required"] = required_confirmation
    out["base_confidence_score"] = pd.to_numeric(out["confidence_score"], errors="coerce").fillna(0.0)
    out["base_market_opportunity_score"] = pd.to_numeric(out["market_opportunity_score"], errors="coerce").fillna(0.0)
    out["base_signal_quality"] = out["signal_quality"].astype(str)

    execution_status = np.full(len(out), "OK", dtype=object)
    execution_status = np.where(hard_veto_long, "BLOCKED", execution_status)
    if mode is EngineMode.COPILOT:
        execution_status = np.where(
            (execution_status == "OK") & long_mask & ~entry_validation_passed & (size_multiplier < 0.5),
            "HIGH_RISK",
            execution_status,
        )
        execution_status = np.where(
            (execution_status == "OK") & long_mask & (~entry_validation_passed | (size_multiplier < 0.999)),
            "CAUTION",
            execution_status,
        )
    else:
        execution_status = np.where(long_mask & ~entry_validation_passed, "BLOCKED", execution_status)
        execution_status = np.where(
            (execution_status == "OK") & long_mask & (size_multiplier < 0.5),
            "HIGH_RISK",
            execution_status,
        )
        execution_status = np.where(
            (execution_status == "OK") & long_mask & (size_multiplier < 0.999),
            "CAUTION",
            execution_status,
        )
    out["execution_status"] = execution_status
    out["execution_allowed"] = out["execution_status"] != "BLOCKED"

    if mode is EngineMode.COPILOT:
        out["passes_filters"] = out["passes_filters"]
        out["confidence_score"] = out["base_confidence_score"]
        out["market_opportunity_score"] = out["base_market_opportunity_score"]
        out["signal_quality"] = out["base_signal_quality"]
    else:
        out["passes_filters"] = out["passes_filters"] & out["regime_filter_passed"]
        out["confidence_score"] = (out["base_confidence_score"] - confidence_penalty).clip(lower=0.0)
        out["market_opportunity_score"] = (
            out["base_market_opportunity_score"]
            - confidence_penalty * 0.75
        ).clip(lower=0.0)
        out["signal_quality"] = out["market_opportunity_score"].apply(_quality_from_market_score)

    out["regime_score_penalty"] = np.round(confidence_penalty * 0.75, 4)

    out = out.sort_values(
        by=["passes_filters", "market_opportunity_score", "confidence_score", "edge_after_fees_pct"],
        ascending=[False, False, False, False],
    ).reset_index(drop=True)
    return out, regime_state
