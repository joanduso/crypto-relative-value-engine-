from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from interval_profiles import interval_to_minutes
from risk_engine import RiskLimits, RiskState, attach_trade_plan
from signal_engine import EngineMode, prepare_signal_frame


CONFIDENCE_BUCKETS = (0.0, 45.0, 55.0, 65.0, 75.0, 85.0, 101.0)


@dataclass(frozen=True)
class CalibrationStats:
    overall_win_rate_pct: float
    overall_avg_return_pct: float
    sample_size: int
    bucket_table: pd.DataFrame


def _confidence_bucket(series: pd.Series) -> pd.Series:
    labels = [f"{int(CONFIDENCE_BUCKETS[idx])}-{int(CONFIDENCE_BUCKETS[idx + 1])}" for idx in range(len(CONFIDENCE_BUCKETS) - 1)]
    return pd.cut(
        pd.to_numeric(series, errors="coerce"),
        bins=CONFIDENCE_BUCKETS,
        labels=labels,
        right=False,
        include_lowest=True,
    ).astype("string")


def _holding_bars(interval: str, max_holding_hours: int) -> int:
    minutes = max(interval_to_minutes(interval), 1)
    return max(int((max_holding_hours * 60) / minutes), 1)


def _simulate_historical_outcomes(signal_df: pd.DataFrame, interval: str, max_holding_hours: int) -> pd.DataFrame:
    if signal_df.empty:
        return signal_df.copy()

    max_bars = _holding_bars(interval, max_holding_hours)
    results: list[pd.DataFrame] = []

    for _, symbol_df in signal_df.groupby("symbol", sort=False):
        symbol_df = symbol_df.sort_values("timestamp").reset_index(drop=True).copy()
        prices = pd.to_numeric(symbol_df["current_price"], errors="coerce").to_numpy(dtype=float)
        highs = pd.to_numeric(symbol_df.get("alt_high", symbol_df["current_price"]), errors="coerce").fillna(symbol_df["current_price"]).to_numpy(dtype=float)
        lows = pd.to_numeric(symbol_df.get("alt_low", symbol_df["current_price"]), errors="coerce").fillna(symbol_df["current_price"]).to_numpy(dtype=float)
        directions = symbol_df["suggested_direction"].astype(str).to_numpy()
        entries = pd.to_numeric(symbol_df["suggested_entry"], errors="coerce").to_numpy(dtype=float)
        stops = pd.to_numeric(symbol_df["suggested_stop_loss"], errors="coerce").to_numpy(dtype=float)
        takes = pd.to_numeric(symbol_df["suggested_take_profit"], errors="coerce").to_numpy(dtype=float)

        realized_returns: list[float] = []
        win_labels: list[float] = []
        exit_labels: list[str] = []

        for idx in range(len(symbol_df)):
            future_end = min(idx + max_bars + 1, len(symbol_df))
            future_prices = prices[idx + 1:future_end]
            future_highs = highs[idx + 1:future_end]
            future_lows = lows[idx + 1:future_end]
            if future_prices.size == 0:
                realized_returns.append(np.nan)
                win_labels.append(np.nan)
                exit_labels.append("insufficient_future")
                continue

            direction = directions[idx]
            entry = entries[idx]
            stop = stops[idx]
            take = takes[idx]
            outcome_return = np.nan
            exit_reason = "timeout"

            for future_price, future_high, future_low in zip(future_prices, future_highs, future_lows):
                if direction == "LONG":
                    if future_high >= take:
                        outcome_return = (take / max(entry, 1e-9) - 1.0) * 100.0
                        exit_reason = "take_profit"
                        break
                    if future_low <= stop:
                        outcome_return = (stop / max(entry, 1e-9) - 1.0) * 100.0
                        exit_reason = "stop_loss"
                        break
                else:
                    if future_low <= take:
                        outcome_return = (1.0 - take / max(entry, 1e-9)) * 100.0
                        exit_reason = "take_profit"
                        break
                    if future_high >= stop:
                        outcome_return = (1.0 - stop / max(entry, 1e-9)) * 100.0
                        exit_reason = "stop_loss"
                        break

            if np.isnan(outcome_return):
                final_price = future_prices[-1]
                if direction == "LONG":
                    outcome_return = (final_price / max(entry, 1e-9) - 1.0) * 100.0
                else:
                    outcome_return = (1.0 - final_price / max(entry, 1e-9)) * 100.0

            realized_returns.append(outcome_return)
            win_labels.append(1.0 if outcome_return > 0 else 0.0)
            exit_labels.append(exit_reason)

        symbol_df["historical_realized_return_pct"] = realized_returns
        symbol_df["historical_win_label"] = win_labels
        symbol_df["historical_exit_reason"] = exit_labels
        results.append(symbol_df)

    return pd.concat(results, ignore_index=True) if results else signal_df.iloc[0:0].copy()


def calibrate_from_history(
    features_df: pd.DataFrame,
    ranked_universe: pd.DataFrame,
    *,
    mode: EngineMode,
    interval: str,
    risk_limits: RiskLimits,
) -> tuple[pd.DataFrame, CalibrationStats | None]:
    if features_df.empty or ranked_universe.empty:
        return ranked_universe.copy(), None

    historical = prepare_signal_frame(features_df, mode)
    historical = historical.loc[historical["passes_filters"]].copy()
    if historical.empty:
        return ranked_universe.copy(), None

    historical = historical.rename(
        columns={
            "alt_price": "current_price",
            "fair_value": "expected_fair_value",
        }
    )

    required_history_columns = [
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
        "suggested_direction",
        "suggested_entry",
        "quote_volume",
        "realized_volatility",
        "funding_rate",
        "edge_after_fees_pct",
        "confidence_score",
        "passes_filters",
    ]
    historical = historical[required_history_columns].copy()
    historical["suggested_stop_loss"] = np.nan
    historical["suggested_take_profit"] = np.nan
    historical["suggested_position_size"] = np.nan

    historical = attach_trade_plan(
        historical,
        limits=RiskLimits(
            max_concurrent_positions=999,
            max_daily_loss_pct=1.0,
            max_weekly_drawdown_pct=1.0,
            max_risk_per_trade_pct=risk_limits.max_risk_per_trade_pct,
            max_holding_hours=risk_limits.max_holding_hours,
            stop_loss_pct=risk_limits.stop_loss_pct,
            take_profit_pct=risk_limits.take_profit_pct,
        ),
        state=RiskState(equity=100000.0),
    )
    historical = _simulate_historical_outcomes(historical, interval=interval, max_holding_hours=risk_limits.max_holding_hours)
    historical = historical.dropna(subset=["historical_realized_return_pct", "historical_win_label"]).copy()
    if historical.empty:
        return ranked_universe.copy(), None

    historical["confidence_bucket"] = _confidence_bucket(historical["confidence_score"])
    overall_win_rate = float(historical["historical_win_label"].mean() * 100.0)
    overall_avg_return = float(historical["historical_realized_return_pct"].mean())
    prior_mean = historical["historical_win_label"].mean()
    prior_strength = 12.0

    bucket_table = (
        historical.groupby(["suggested_direction", "confidence_bucket"], dropna=False)
        .agg(
            bucket_sample_size=("historical_win_label", "size"),
            bucket_win_rate=("historical_win_label", "mean"),
            bucket_avg_return_pct=("historical_realized_return_pct", "mean"),
        )
        .reset_index()
    )
    bucket_table["calibrated_win_rate_pct"] = (
        ((bucket_table["bucket_win_rate"] * bucket_table["bucket_sample_size"]) + (prior_mean * prior_strength))
        / (bucket_table["bucket_sample_size"] + prior_strength)
    ) * 100.0

    enriched = ranked_universe.copy()
    enriched["confidence_bucket"] = _confidence_bucket(enriched["confidence_score"])
    enriched = enriched.merge(
        bucket_table[
            [
                "suggested_direction",
                "confidence_bucket",
                "bucket_sample_size",
                "bucket_avg_return_pct",
                "calibrated_win_rate_pct",
            ]
        ],
        on=["suggested_direction", "confidence_bucket"],
        how="left",
    )
    enriched["bucket_sample_size"] = pd.to_numeric(enriched["bucket_sample_size"], errors="coerce").fillna(0).astype(int)
    enriched["bucket_avg_return_pct"] = pd.to_numeric(enriched["bucket_avg_return_pct"], errors="coerce").fillna(overall_avg_return)
    enriched["calibrated_win_rate_pct"] = pd.to_numeric(enriched["calibrated_win_rate_pct"], errors="coerce").fillna(overall_win_rate)
    enriched["calibration_sample_size"] = enriched["bucket_sample_size"]
    enriched["calibration_source"] = np.where(
        enriched["bucket_sample_size"] > 0,
        "historical_bucket",
        "historical_global",
    )

    stats = CalibrationStats(
        overall_win_rate_pct=round(overall_win_rate, 2),
        overall_avg_return_pct=round(overall_avg_return, 4),
        sample_size=int(len(historical)),
        bucket_table=bucket_table,
    )
    return enriched, stats
