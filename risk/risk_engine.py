from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Literal


logger = logging.getLogger(__name__)

RiskDecision = Literal["ALLOW", "REDUCE_POSITION", "STOP_TRADING"]


@dataclass(frozen=True)
class RiskLimits:
    """Risk limits applied to modular strategy signals."""

    max_position_size: float
    max_daily_drawdown: float
    max_leverage: float


@dataclass(frozen=True)
class RiskEvaluation:
    """Risk evaluation output for a proposed trade."""

    decision: RiskDecision
    approved_position_size: float
    approved_leverage: float
    reason: str


def evaluate_risk(
    limits: RiskLimits,
    requested_position_size: float,
    current_drawdown: float,
    requested_leverage: float,
) -> RiskEvaluation:
    """Evaluate position size, drawdown and leverage against hard limits."""
    if current_drawdown >= limits.max_daily_drawdown:
        evaluation = RiskEvaluation(
            decision="STOP_TRADING",
            approved_position_size=0.0,
            approved_leverage=0.0,
            reason="max_daily_drawdown_breached",
        )
    elif requested_position_size > limits.max_position_size or requested_leverage > limits.max_leverage:
        evaluation = RiskEvaluation(
            decision="REDUCE_POSITION",
            approved_position_size=min(requested_position_size, limits.max_position_size),
            approved_leverage=min(requested_leverage, limits.max_leverage),
            reason="position_or_leverage_trimmed",
        )
    else:
        evaluation = RiskEvaluation(
            decision="ALLOW",
            approved_position_size=requested_position_size,
            approved_leverage=requested_leverage,
            reason="within_limits",
        )

    logger.info("risk evaluated", extra=evaluation.__dict__)
    return evaluation
