from __future__ import annotations

import logging
from typing import Literal


logger = logging.getLogger(__name__)

FundingSignal = Literal["SHORT_PERP_LONG_SPOT", "LONG_PERP_SHORT_SPOT", "NO_TRADE"]


def get_funding_signal(symbol: str, funding_rate: float, threshold: float) -> FundingSignal:
    """Return the funding arbitrage action for a symbol."""
    if funding_rate > threshold:
        signal: FundingSignal = "SHORT_PERP_LONG_SPOT"
    elif funding_rate < -threshold:
        signal = "LONG_PERP_SHORT_SPOT"
    else:
        signal = "NO_TRADE"

    logger.info(
        "funding signal computed",
        extra={
            "symbol": symbol,
            "funding_rate": funding_rate,
            "threshold": threshold,
            "signal": signal,
        },
    )
    return signal
