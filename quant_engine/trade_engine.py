from __future__ import annotations

import pandas as pd


def build_trade_setups(opportunities_df: pd.DataFrame) -> pd.DataFrame:
    if opportunities_df.empty:
        return opportunities_df.copy()

    trades = opportunities_df.copy()
    trades["entry"] = pd.to_numeric(trades["current_price"], errors="coerce").fillna(0.0)
    volatility_fraction = pd.to_numeric(trades["realized_volatility"], errors="coerce").fillna(0.0).clip(0.01, 5.0) / 100.0
    trades["risk_pct"] = volatility_fraction.clip(lower=0.01, upper=0.05)

    long_mask = trades["direction"].astype(str).str.upper() == "LONG"
    short_mask = trades["direction"].astype(str).str.upper() == "SHORT"

    trades["stop_loss"] = trades["entry"]
    trades.loc[long_mask, "stop_loss"] = trades.loc[long_mask, "entry"] * (1.0 - trades.loc[long_mask, "risk_pct"])
    trades.loc[short_mask, "stop_loss"] = trades.loc[short_mask, "entry"] * (1.0 + trades.loc[short_mask, "risk_pct"])

    trades["take_profit"] = trades["entry"]
    trades.loc[long_mask, "take_profit"] = trades.loc[long_mask, "entry"] * (1.0 + 2.0 * trades.loc[long_mask, "risk_pct"])
    trades.loc[short_mask, "take_profit"] = trades.loc[short_mask, "entry"] * (1.0 - 2.0 * trades.loc[short_mask, "risk_pct"])

    trades["risk_reward_ratio"] = 2.0
    return trades
