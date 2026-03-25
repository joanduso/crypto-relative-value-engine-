from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class RiskLimits:
    max_concurrent_positions: int
    max_daily_loss_pct: float
    max_weekly_drawdown_pct: float
    max_risk_per_trade_pct: float
    max_holding_hours: int
    stop_loss_pct: float
    take_profit_pct: float


@dataclass
class RiskState:
    equity: float = 100000.0
    realized_pnl_today: float = 0.0
    weekly_drawdown_pct: float = 0.0
    open_positions: int = 0


def direction_from_zscore(z_score: float) -> str:
    return "LONG" if z_score < 0 else "SHORT"


def compute_position_size(
    equity: float,
    max_risk_per_trade_pct: float,
    entry_price: float,
    stop_loss_price: float,
) -> float:
    risk_budget = equity * max_risk_per_trade_pct
    stop_distance = abs(entry_price - stop_loss_price)
    if stop_distance <= 0:
        return 0.0
    return max(risk_budget / stop_distance, 0.0)


def attach_trade_plan(
    opportunities: pd.DataFrame,
    limits: RiskLimits,
    state: RiskState,
) -> pd.DataFrame:
    if opportunities.empty:
        return opportunities.copy()

    out = opportunities.copy()
    stop_losses: list[float] = []
    take_profits: list[float] = []
    position_sizes: list[float] = []
    time_stops: list[int] = []
    risk_flags: list[bool] = []
    risk_reward_ratios: list[float] = []
    implied_win_rates: list[float] = []
    effective_win_rates: list[float] = []
    expected_value_pcts: list[float] = []
    stop_distance_pcts: list[float] = []
    target_distance_pcts: list[float] = []
    win_rate_sources: list[str] = []

    for row in out.itertuples(index=False):
        current_price = float(row.current_price)
        realized_volatility = float(getattr(row, "realized_volatility", 0.0) or 0.0)
        edge_after_fees_pct = float(getattr(row, "edge_after_fees_pct", 0.0) or 0.0)
        confidence_score = float(getattr(row, "confidence_score", 0.0) or 0.0)
        regime_multiplier = float(getattr(row, "regime_position_size_multiplier", 1.0) or 1.0)

        # Make the stop respond to volatility so the same fixed percentages are not
        # applied to very different market conditions.
        volatility_stop_pct = min(max(realized_volatility * 1.35, limits.stop_loss_pct * 0.75), limits.stop_loss_pct * 1.8)
        stop_pct = max(limits.stop_loss_pct * 0.75, volatility_stop_pct)
        gross_target_pct = max(min(edge_after_fees_pct * 0.80, limits.take_profit_pct * 1.35), stop_pct * 1.05)
        target_pct = min(gross_target_pct, max(limits.take_profit_pct * 0.60, stop_pct * 2.4))

        if row.suggested_direction == "LONG":
            stop_loss = current_price * (1.0 - stop_pct)
            take_profit = current_price * (1.0 + target_pct)
        else:
            stop_loss = current_price * (1.0 + stop_pct)
            take_profit = current_price * (1.0 - target_pct)
        size = compute_position_size(
            equity=state.equity,
            max_risk_per_trade_pct=limits.max_risk_per_trade_pct,
            entry_price=row.suggested_entry,
            stop_loss_price=stop_loss,
        )
        size *= max(regime_multiplier, 0.0)

        risk_distance_pct = abs(float(row.suggested_entry) - stop_loss) / max(float(row.suggested_entry), 1e-9) * 100.0
        reward_distance_pct = abs(take_profit - float(row.suggested_entry)) / max(float(row.suggested_entry), 1e-9) * 100.0
        risk_reward_ratio = reward_distance_pct / max(risk_distance_pct, 1e-9)

        # This is a conservative proxy from the internal score, not a calibrated probability.
        implied_win_rate = 0.35 + (min(max(confidence_score, 0.0), 100.0) / 100.0) * 0.30
        implied_win_rate *= min(max(regime_multiplier, 0.0), 1.0)
        implied_win_rate = min(max(implied_win_rate, 0.20), 0.68)
        calibrated_win_rate_pct = pd.to_numeric(getattr(row, "calibrated_win_rate_pct", np.nan), errors="coerce")
        if pd.notna(calibrated_win_rate_pct) and float(calibrated_win_rate_pct) > 0.0:
            effective_win_rate = min(max(float(calibrated_win_rate_pct) / 100.0, 0.20), 0.80)
            win_rate_source = str(getattr(row, "calibration_source", "historical"))
        else:
            effective_win_rate = implied_win_rate
            win_rate_source = "model_proxy"
        expected_value_pct = (
            effective_win_rate * reward_distance_pct
            - (1.0 - effective_win_rate) * risk_distance_pct
        )

        daily_ok = state.realized_pnl_today > -(state.equity * limits.max_daily_loss_pct)
        weekly_ok = state.weekly_drawdown_pct > -limits.max_weekly_drawdown_pct
        slots_ok = state.open_positions < limits.max_concurrent_positions
        liquidity_ok = bool(np.isfinite(row.quote_volume)) and row.quote_volume > 0
        expectancy_ok = expected_value_pct > 0.0 and risk_reward_ratio >= 1.25

        stop_losses.append(stop_loss)
        take_profits.append(take_profit)
        position_sizes.append(size)
        time_stops.append(limits.max_holding_hours)
        risk_flags.append(daily_ok and weekly_ok and slots_ok and liquidity_ok and expectancy_ok)
        stop_distance_pcts.append(risk_distance_pct)
        target_distance_pcts.append(reward_distance_pct)
        risk_reward_ratios.append(risk_reward_ratio)
        implied_win_rates.append(implied_win_rate * 100.0)
        effective_win_rates.append(effective_win_rate * 100.0)
        expected_value_pcts.append(expected_value_pct)
        win_rate_sources.append(win_rate_source)

    out["suggested_stop_loss"] = stop_losses
    out["suggested_take_profit"] = take_profits
    out["suggested_position_size"] = position_sizes
    out["time_stop_hours"] = time_stops
    out["risk_checks_passed"] = risk_flags
    out["stop_distance_pct"] = stop_distance_pcts
    out["target_distance_pct"] = target_distance_pcts
    out["risk_reward_ratio"] = risk_reward_ratios
    out["implied_win_rate_pct"] = implied_win_rates
    out["effective_win_rate_pct"] = effective_win_rates
    out["win_rate_source"] = win_rate_sources
    out["expected_value_pct"] = expected_value_pcts
    return out
