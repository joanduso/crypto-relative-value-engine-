from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from risk_engine import RiskLimits, RiskState, attach_trade_plan


@dataclass
class Position:
    symbol: str
    direction: str
    entry_price: float
    quantity: float
    stop_loss: float
    take_profit: float
    opened_at: pd.Timestamp
    time_stop_hours: int
    mode: str


@dataclass
class PortfolioManager:
    limits: RiskLimits
    state: RiskState = field(default_factory=RiskState)
    positions: list[Position] = field(default_factory=list)

    def prepare_orders(self, opportunities: pd.DataFrame) -> pd.DataFrame:
        self.state.open_positions = len(self.positions)
        planned = attach_trade_plan(opportunities, self.limits, self.state)
        if planned.empty:
            return planned

        available_slots = max(self.limits.max_concurrent_positions - self.state.open_positions, 0)
        planned = planned.sort_values(
            by=["confidence_score", "edge_after_fees_pct", "spread_stability_score"],
            ascending=[False, False, False],
        ).reset_index(drop=True)
        planned["risk_checks_passed"] = planned["risk_checks_passed"] & (planned.index < available_slots)
        return planned

    def register_position(self, order_row: pd.Series, mode: str) -> None:
        position = Position(
            symbol=str(order_row["symbol"]),
            direction=str(order_row["suggested_direction"]),
            entry_price=float(order_row["suggested_entry"]),
            quantity=float(order_row["suggested_position_size"]),
            stop_loss=float(order_row["suggested_stop_loss"]),
            take_profit=float(order_row["suggested_take_profit"]),
            opened_at=pd.Timestamp.utcnow(),
            time_stop_hours=int(order_row["time_stop_hours"]),
            mode=mode,
        )
        self.positions.append(position)
        self.state.open_positions = len(self.positions)

    def mark_realized_pnl(self, pnl: float) -> None:
        self.state.realized_pnl_today += pnl
        self.state.weekly_drawdown_pct = min(self.state.weekly_drawdown_pct, pnl / max(self.state.equity, 1e-9))
