from __future__ import annotations

import logging
from typing import Literal


logger = logging.getLogger(__name__)

BasisSignal = Literal["SHORT_PERP_LONG_SPOT", "LONG_PERP_SHORT_SPOT", "NO_TRADE"]


def calculate_basis(spot_price: float, perp_price: float) -> float:
    """Calculate futures basis relative to the spot price."""
    if spot_price <= 0:
        raise ValueError("spot_price must be greater than zero")
    return (perp_price - spot_price) / spot_price


def get_basis_signal(spot_price: float, perp_price: float) -> BasisSignal:
    """Map basis to a directional basis trade signal."""
    basis = calculate_basis(spot_price, perp_price)
    if basis > 0.003:
        signal: BasisSignal = "SHORT_PERP_LONG_SPOT"
    elif basis < -0.003:
        signal = "LONG_PERP_SHORT_SPOT"
    else:
        signal = "NO_TRADE"

    logger.info(
        "basis signal computed",
        extra={"spot_price": spot_price, "perp_price": perp_price, "basis": basis, "signal": signal},
    )
    return signal
