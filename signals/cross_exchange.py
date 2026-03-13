from __future__ import annotations

import logging
from dataclasses import dataclass


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ArbitrageOpportunity:
    """Describes a cross-exchange arbitrage opportunity."""

    price_exchange_a: float
    price_exchange_b: float
    difference_pct: float
    buy_exchange: str
    sell_exchange: str


def find_arbitrage(price_exchange_a: float, price_exchange_b: float) -> ArbitrageOpportunity | None:
    """Return an arbitrage opportunity if the price gap exceeds 0.2%."""
    if price_exchange_a <= 0 or price_exchange_b <= 0:
        raise ValueError("exchange prices must be greater than zero")

    difference_pct = abs(price_exchange_a - price_exchange_b) / min(price_exchange_a, price_exchange_b)
    if difference_pct <= 0.002:
        logger.info(
            "cross exchange no trade",
            extra={
                "price_exchange_a": price_exchange_a,
                "price_exchange_b": price_exchange_b,
                "difference_pct": difference_pct,
            },
        )
        return None

    buy_exchange = "EXCHANGE_A" if price_exchange_a < price_exchange_b else "EXCHANGE_B"
    sell_exchange = "EXCHANGE_B" if buy_exchange == "EXCHANGE_A" else "EXCHANGE_A"
    opportunity = ArbitrageOpportunity(
        price_exchange_a=price_exchange_a,
        price_exchange_b=price_exchange_b,
        difference_pct=difference_pct,
        buy_exchange=buy_exchange,
        sell_exchange=sell_exchange,
    )
    logger.info("cross exchange arbitrage found", extra=opportunity.__dict__)
    return opportunity
