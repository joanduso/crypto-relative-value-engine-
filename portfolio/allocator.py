from __future__ import annotations

import logging
from collections import Counter
from typing import Iterable, Mapping


logger = logging.getLogger(__name__)

DEFAULT_WEIGHTS: dict[str, float] = {
    "funding_arbitrage": 0.3,
    "basis_trade": 0.25,
    "cross_exchange": 0.15,
    "btc_directional": 0.3,
}


def allocate_capital(
    total_capital: float,
    signals: Iterable[Mapping[str, object]],
    strategy_weights: Mapping[str, float] | None = None,
) -> list[dict[str, float | str]]:
    """Allocate capital across active strategy signals using configurable weights."""
    if total_capital < 0:
        raise ValueError("total_capital must be non-negative")

    weights = dict(strategy_weights or DEFAULT_WEIGHTS)
    active_signals = [
        signal for signal in signals
        if str(signal.get("signal", "NO_TRADE")) != "NO_TRADE"
    ]
    if not active_signals or total_capital == 0:
        return []

    strategy_counts = Counter(str(signal.get("strategy", "unknown")) for signal in active_signals)
    allocations: list[dict[str, float | str]] = []

    for signal in active_signals:
        strategy = str(signal.get("strategy", "unknown"))
        configured_weight = float(weights.get(strategy, 0.0))
        strategy_total = total_capital * configured_weight
        position_count = max(strategy_counts[strategy], 1)
        allocation = strategy_total / position_count
        allocation_row: dict[str, float | str] = {
            "strategy": strategy,
            "symbol": str(signal.get("symbol", "")),
            "signal": str(signal.get("signal", "")),
            "allocated_capital": round(allocation, 2),
            "strategy_weight": configured_weight,
        }
        allocations.append(allocation_row)

    logger.info("capital allocated", extra={"total_capital": total_capital, "allocations": len(allocations)})
    return allocations
