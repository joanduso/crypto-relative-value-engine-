from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from signal_engine import EngineMode, MODE_THRESHOLDS


@dataclass(frozen=True)
class BacktestConfig:
    hours_per_year: int = 24 * 365


def _build_positions(features_df: pd.DataFrame, mode: EngineMode) -> pd.DataFrame:
    if features_df.empty or "symbol" not in features_df.columns or "timestamp" not in features_df.columns:
        return pd.DataFrame()
    thresholds = MODE_THRESHOLDS[mode]
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
    for mode in (EngineMode.COPILOT, EngineMode.AUTO_SAFE):
        bt = _build_positions(features_df, mode)
        stats = _trade_statistics(bt, cfg)
        rows.append({"mode": mode.value, **stats})
    return pd.DataFrame(rows)
