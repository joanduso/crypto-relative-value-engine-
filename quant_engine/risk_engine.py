from __future__ import annotations

import pandas as pd


def apply_risk_management(trades_df: pd.DataFrame) -> pd.DataFrame:
    if trades_df.empty:
        return trades_df.copy()

    managed = trades_df.copy()
    base_size = (pd.to_numeric(managed["opportunity_score"], errors="coerce").fillna(0.0) / 100.0) * 25.0
    volatility = pd.to_numeric(managed["realized_volatility"], errors="coerce").fillna(0.0)
    reduction = volatility.clip(lower=0.0, upper=5.0) / 10.0

    managed["position_size_pct"] = (base_size * (1.0 - reduction)).clip(lower=5.0, upper=25.0).round(2)
    return managed
