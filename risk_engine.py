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

    for row in out.itertuples(index=False):
        if row.suggested_direction == "LONG":
            stop_loss = row.current_price * (1.0 - limits.stop_loss_pct)
            take_profit = row.current_price * (1.0 + limits.take_profit_pct)
        else:
            stop_loss = row.current_price * (1.0 + limits.stop_loss_pct)
            take_profit = row.current_price * (1.0 - limits.take_profit_pct)
        size = compute_position_size(
            equity=state.equity,
            max_risk_per_trade_pct=limits.max_risk_per_trade_pct,
            entry_price=row.suggested_entry,
            stop_loss_price=stop_loss,
        )

        daily_ok = state.realized_pnl_today > -(state.equity * limits.max_daily_loss_pct)
        weekly_ok = state.weekly_drawdown_pct > -limits.max_weekly_drawdown_pct
        slots_ok = state.open_positions < limits.max_concurrent_positions
        liquidity_ok = bool(np.isfinite(row.quote_volume)) and row.quote_volume > 0

        stop_losses.append(stop_loss)
        take_profits.append(take_profit)
        position_sizes.append(size)
        time_stops.append(limits.max_holding_hours)
        risk_flags.append(daily_ok and weekly_ok and slots_ok and liquidity_ok)

    out["suggested_stop_loss"] = stop_losses
    out["suggested_take_profit"] = take_profits
    out["suggested_position_size"] = position_sizes
    out["time_stop_hours"] = time_stops
    out["risk_checks_passed"] = risk_flags
    return out
