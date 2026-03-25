from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from signal_engine import EngineMode, resolve_mode_thresholds


@dataclass(frozen=True)
class BacktestConfig:
    hours_per_year: int = 24 * 365


def _build_positions(features_df: pd.DataFrame, mode: EngineMode) -> pd.DataFrame:
    if features_df.empty or "symbol" not in features_df.columns or "timestamp" not in features_df.columns:
        return pd.DataFrame()
    thresholds = resolve_mode_thresholds(mode)
    bt = features_df.sort_values(["symbol", "timestamp"]).copy()
    bt = bt.dropna(subset=["alt_price", "z_score", "realized_volatility", "spread_stability_score"])
    bt["abs_z_score"] = bt["z_score"].abs()
    bt["position"] = 0

    mask = (
        (bt["abs_z_score"] >= thresholds.zscore_entry)
        & (bt["realized_volatility"] <= thresholds.max_realized_volatility)
        & (bt["quote_volume"] >= thresholds.min_quote_volume)
        & (bt["spread_stability_score"] >= thresholds.min_spread_stability)
        & (bt["funding_rate"].fillna(0.0).abs() <= thresholds.max_abs_funding)
    )
    bt.loc[mask & (bt["z_score"] < 0), "position"] = 1
    bt.loc[mask & (bt["z_score"] > 0), "position"] = -1
    bt["asset_return"] = bt.groupby("symbol")["alt_price"].pct_change().fillna(0.0)
    bt["strategy_return"] = bt.groupby("symbol")["position"].shift(1).fillna(0.0) * bt["asset_return"]
    fee_rate = thresholds.fee_bps / 10_000.0
    turnover = bt.groupby("symbol")["position"].diff().abs().fillna(bt["position"].abs())
    bt["net_strategy_return"] = bt["strategy_return"] - (turnover * fee_rate)
    return bt


def _trade_statistics(bt: pd.DataFrame, config: BacktestConfig) -> dict[str, float]:
    if bt.empty:
        return {
            "total_return_pct": 0.0,
            "sharpe": 0.0,
            "max_drawdown_pct": 0.0,
            "win_rate_pct": 0.0,
            "avg_holding_hours": 0.0,
            "pnl_after_fees_pct": 0.0,
        }

    aggregated = bt.groupby("timestamp")["net_strategy_return"].mean().fillna(0.0)
    equity_curve = (1.0 + aggregated).cumprod()
    total_return = (equity_curve.iloc[-1] - 1.0) * 100.0
    running_max = equity_curve.cummax()
    drawdown = equity_curve / running_max - 1.0
    max_drawdown = drawdown.min() * 100.0
    mean_ret = aggregated.mean()
    std_ret = aggregated.std(ddof=0)
    sharpe = float((mean_ret / std_ret) * np.sqrt(config.hours_per_year)) if std_ret and not np.isnan(std_ret) else 0.0

    active = bt.loc[bt.groupby("symbol")["position"].shift(1).fillna(0.0) != 0, "net_strategy_return"]
    win_rate = float((active > 0).mean() * 100.0) if not active.empty else 0.0

    holding_lengths: list[int] = []
    for _, symbol_df in bt.groupby("symbol"):
        run_length = 0
        for position in symbol_df["position"]:
            if position != 0:
                run_length += 1
            elif run_length:
                holding_lengths.append(run_length)
                run_length = 0
        if run_length:
            holding_lengths.append(run_length)

    return {
        "total_return_pct": float(total_return),
        "sharpe": sharpe,
        "max_drawdown_pct": float(max_drawdown),
        "win_rate_pct": win_rate,
        "avg_holding_hours": float(np.mean(holding_lengths)) if holding_lengths else 0.0,
        "pnl_after_fees_pct": float(bt["net_strategy_return"].sum() * 100.0),
    }


def compare_mode_backtests(
    features_df: pd.DataFrame,
    config: BacktestConfig | None = None,
) -> pd.DataFrame:
    cfg = config or BacktestConfig()
    rows: list[dict[str, float | str]] = []
    for mode in (EngineMode.COPILOT, EngineMode.COPILOT_RELAXED, EngineMode.AUTO_SAFE):
        bt = _build_positions(features_df, mode)
        stats = _trade_statistics(bt, cfg)
        rows.append({"mode": mode.value, **stats})
    return pd.DataFrame(rows)


def _prepare_regime_frame(market_df: pd.DataFrame) -> pd.DataFrame:
    if market_df.empty:
        return pd.DataFrame()

    prices = market_df.pivot(index="timestamp", columns="symbol", values="close").sort_index()
    if "BTCUSDT" not in prices.columns:
        return pd.DataFrame(index=prices.index)

    btc = prices["BTCUSDT"].dropna()
    btc_4h = btc.resample("4h").last().dropna()
    ema50_4h = btc_4h.ewm(span=50, adjust=False).mean()
    ema200_4h = btc_4h.ewm(span=200, adjust=False).mean()
    ema50_slope = ema50_4h.pct_change(3)
    ema200_slope = ema200_4h.pct_change(3)

    regime_4h = pd.DataFrame(index=btc_4h.index)
    regime_4h["btc_price_vs_ema200_pct"] = (btc_4h / ema200_4h - 1.0) * 100.0
    regime_4h["btc_ema50_vs_ema200_pct"] = (ema50_4h / ema200_4h - 1.0) * 100.0
    regime_4h["btc_ema200_slope_pct"] = ema200_slope * 100.0
    regime_4h["btc_directional_score"] = (
        np.where(btc_4h > ema200_4h, 35.0, -35.0)
        + np.where(ema50_4h > ema200_4h, 25.0, -25.0)
        + np.where(ema50_slope > 0, 20.0, -20.0)
        + np.where(ema200_slope > 0, 20.0, -20.0)
    )
    regime_4h["btc_regime"] = "range"
    regime_4h.loc[
        (regime_4h["btc_directional_score"] >= 35.0)
        & (ema50_4h > ema200_4h)
        & (ema200_slope > 0),
        "btc_regime",
    ] = "trending_bullish"
    regime_4h.loc[
        (regime_4h["btc_directional_score"] <= -35.0)
        & (ema50_4h < ema200_4h)
        & (ema200_slope < 0),
        "btc_regime",
    ] = "trending_bearish"

    regime = regime_4h.reindex(prices.index, method="ffill")
    alt_prices = prices.drop(columns=[col for col in ["BTCUSDT", "ETHUSDT"] if col in prices.columns], errors="ignore")
    alt_returns = alt_prices.pct_change()
    trailing_returns = alt_prices.pct_change(periods=12)
    ema20 = alt_prices.ewm(span=20, adjust=False).mean()
    below_ema_pct = (alt_prices < ema20).mean(axis=1) * 100.0
    short_momentum_pct = (alt_prices.pct_change(periods=3) < 0).mean(axis=1) * 100.0
    regime["breadth_deterioration_pct"] = (trailing_returns < 0).mean(axis=1) * 100.0
    regime["breadth_stress_score"] = (
        0.45 * regime["breadth_deterioration_pct"]
        + 0.35 * below_ema_pct
        + 0.20 * short_momentum_pct
    )
    regime["return_dispersion_pct"] = trailing_returns.std(axis=1, ddof=0) * 100.0

    avg_corr: list[float] = []
    for idx in range(len(alt_returns)):
        window = alt_returns.iloc[max(0, idx - 23): idx + 1].dropna(axis=1, how="all")
        if window.shape[1] < 2 or window.empty:
            avg_corr.append(0.0)
            continue
        corr = window.corr()
        mask = np.triu(np.ones(corr.shape), k=1).astype(bool)
        values = corr.where(mask).stack()
        avg_corr.append(float(values.mean()) if not values.empty else 0.0)
    regime["avg_alt_correlation"] = avg_corr
    return regime.reset_index().rename(columns={"index": "timestamp"})


def _apply_regime_overlay_to_positions(bt: pd.DataFrame, market_df: pd.DataFrame, mode: EngineMode) -> pd.DataFrame:
    if bt.empty:
        return bt.copy()

    regime = _prepare_regime_frame(market_df)
    if regime.empty:
        return bt.copy()

    out = bt.merge(regime, on="timestamp", how="left")
    out["btc_regime"] = out["btc_regime"].fillna("range")
    out["breadth_deterioration_pct"] = pd.to_numeric(out["breadth_deterioration_pct"], errors="coerce").fillna(50.0)
    out["breadth_stress_score"] = pd.to_numeric(out["breadth_stress_score"], errors="coerce").fillna(50.0)
    out["avg_alt_correlation"] = pd.to_numeric(out["avg_alt_correlation"], errors="coerce").fillna(0.0)
    out["btc_directional_score"] = pd.to_numeric(out["btc_directional_score"], errors="coerce").fillna(0.0)
    out["btc_price_vs_ema200_pct"] = pd.to_numeric(out["btc_price_vs_ema200_pct"], errors="coerce").fillna(0.0)

    long_mask = out["position"] > 0
    micro_reversal = pd.to_numeric(out.get("micro_reversal_score"), errors="coerce").fillna(0.0)
    alt_rsi = pd.to_numeric(out.get("alt_rsi_14"), errors="coerce").fillna(50.0)
    ema20_slope = pd.to_numeric(out.get("alt_ema_20_slope_pct"), errors="coerce").fillna(0.0)
    breakout_5 = pd.to_numeric(out.get("breakout_5_pct"), errors="coerce").fillna(0.0)
    higher_low = pd.Series(out.get("higher_low_3", False)).fillna(False).astype(bool)

    hard_veto_long = (
        long_mask
        & (out["btc_regime"] == "trending_bearish")
        & (out["btc_directional_score"] <= -35.0)
        & (out["breadth_stress_score"] >= 67.5)
        & (out["avg_alt_correlation"] >= 0.55)
        & (out["btc_price_vs_ema200_pct"] <= -1.0)
    )
    validation_score = (
        (micro_reversal / 8.0)
        + (alt_rsi >= 50.0).astype(float)
        + (ema20_slope > 0).astype(float)
        + (breakout_5 > -0.15).astype(float)
        + higher_low.astype(float)
    )
    required_confirmation = np.where(
        out["btc_regime"] == "trending_bearish",
        3.25,
        np.where(out["btc_regime"] == "range", 2.5, 1.5),
    )
    validation_passed = (~long_mask) | (validation_score >= required_confirmation)
    size_multiplier = np.ones(len(out), dtype=float)
    validation_buffer = validation_score - required_confirmation

    range_mask = out["btc_regime"] == "range"
    strong_range_long = long_mask & range_mask & (validation_buffer >= 1.0)
    marginal_range_long = long_mask & range_mask & (validation_buffer >= 0.0) & (validation_buffer < 1.0)
    weak_range_long = long_mask & range_mask & (validation_buffer < 0.0)

    bear_mask = out["btc_regime"] == "trending_bearish"
    elite_bear_long = (
        long_mask
        & bear_mask
        & (validation_buffer >= 1.25)
        & (out["breadth_stress_score"] < 75.0)
        & (out["btc_price_vs_ema200_pct"] > -0.75)
    )
    good_bear_long = long_mask & bear_mask & (validation_buffer >= 0.5) & ~elite_bear_long
    weak_bear_long = long_mask & bear_mask & ~elite_bear_long & ~good_bear_long

    if mode is EngineMode.COPILOT:
        size_multiplier = np.where(strong_range_long, 1.0, size_multiplier)
        size_multiplier = np.where(marginal_range_long, 1.0, size_multiplier)
        size_multiplier = np.where(weak_range_long, 0.0, size_multiplier)
        size_multiplier = np.where(elite_bear_long, 1.0, size_multiplier)
        size_multiplier = np.where(good_bear_long, 0.8, size_multiplier)
        size_multiplier = np.where(weak_bear_long, 0.35, size_multiplier)
    else:
        size_multiplier = np.where(strong_range_long, 1.0, size_multiplier)
        size_multiplier = np.where(marginal_range_long, 0.85, size_multiplier)
        size_multiplier = np.where(weak_range_long, 0.0, size_multiplier)
        size_multiplier = np.where(elite_bear_long, 0.8, size_multiplier)
        size_multiplier = np.where(good_bear_long, 0.55, size_multiplier)
        size_multiplier = np.where(weak_bear_long, 0.0, size_multiplier)

    size_multiplier = np.where(hard_veto_long, 0.0, size_multiplier)

    adjusted_position = out["position"].astype(float) * size_multiplier
    adjusted_position = np.where(validation_passed, adjusted_position, 0.0)
    adjusted_position = np.where(hard_veto_long, 0.0, adjusted_position)
    out["position"] = adjusted_position
    turnover = pd.Series(out["position"]).groupby(out["symbol"]).diff().abs().fillna(np.abs(out["position"]))
    fee_rate = out["fee_rate"].iloc[0] if "fee_rate" in out.columns and not out.empty else 0.0
    out["strategy_return"] = pd.Series(out["position"]).groupby(out["symbol"]).shift(1).fillna(0.0) * out["asset_return"]
    out["net_strategy_return"] = out["strategy_return"] - (turnover * fee_rate)
    return out


def _build_positions_with_fee(features_df: pd.DataFrame, mode: EngineMode) -> pd.DataFrame:
    bt = _build_positions(features_df, mode)
    if bt.empty:
        return bt
    bt["fee_rate"] = resolve_mode_thresholds(mode).fee_bps / 10_000.0
    return bt


def compare_regime_overlay_backtests(
    market_df: pd.DataFrame,
    features_df: pd.DataFrame,
    config: BacktestConfig | None = None,
) -> pd.DataFrame:
    cfg = config or BacktestConfig()
    rows: list[dict[str, float | str]] = []
    for mode in (EngineMode.COPILOT, EngineMode.COPILOT_RELAXED, EngineMode.AUTO_SAFE):
        base_bt = _build_positions_with_fee(features_df, mode)
        overlay_bt = _apply_regime_overlay_to_positions(base_bt.copy(), market_df, mode)

        for label, bt in (("base", base_bt), ("regime_overlay", overlay_bt)):
            stats = _trade_statistics(bt, cfg)
            active_exposure_pct = float((bt["position"].abs() > 0).mean() * 100.0) if not bt.empty else 0.0
            trades = float(bt.groupby("symbol")["position"].diff().abs().fillna(bt["position"].abs()).sum() / 2.0) if not bt.empty else 0.0
            long_share_pct = float((bt["position"] > 0).mean() * 100.0) if not bt.empty else 0.0
            rows.append(
                {
                    "mode": mode.value,
                    "variant": label,
                    **stats,
                    "active_exposure_pct": active_exposure_pct,
                    "trade_count_est": trades,
                    "long_share_pct": long_share_pct,
                }
            )
    return pd.DataFrame(rows)
