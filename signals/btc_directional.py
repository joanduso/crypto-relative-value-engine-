from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class BTCDirectionalSignal:
    timestamp: pd.Timestamp
    symbol: str
    signal: str
    score: float
    current_price: float
    ema_fast: float
    ema_slow: float
    ema_trend: float
    ema_gap_pct: float
    rsi_14: float
    macd_hist: float
    breakout_20_pct: float
    realized_volatility: float
    regime: str


def _rsi(series: pd.Series, window: int = 14) -> pd.Series:
    delta = series.diff()
    gains = delta.clip(lower=0.0)
    losses = -delta.clip(upper=0.0)
    avg_gain = gains.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    avg_loss = losses.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi.fillna(50.0)


def build_btc_directional_signal(
    market_df: pd.DataFrame,
    symbol: str = "BTCUSDT",
) -> BTCDirectionalSignal | None:
    if market_df.empty:
        return None

    frame = market_df.copy().sort_values("timestamp").reset_index(drop=True)
    if len(frame) < 210:
        return None

    close = pd.to_numeric(frame["close"], errors="coerce")
    if close.isna().all():
        return None

    returns = close.pct_change().fillna(0.0)
    ema_fast = close.ewm(span=20, adjust=False).mean()
    ema_slow = close.ewm(span=50, adjust=False).mean()
    ema_trend = close.ewm(span=200, adjust=False).mean()
    rsi_14 = _rsi(close, 14)

    ema_12 = close.ewm(span=12, adjust=False).mean()
    ema_26 = close.ewm(span=26, adjust=False).mean()
    macd_line = ema_12 - ema_26
    macd_signal = macd_line.ewm(span=9, adjust=False).mean()
    macd_hist = macd_line - macd_signal

    rolling_high = close.rolling(20, min_periods=20).max().shift(1)
    rolling_low = close.rolling(20, min_periods=20).min().shift(1)
    realized_vol = returns.rolling(20, min_periods=20).std(ddof=0) * np.sqrt(365 * 24)

    latest = frame.iloc[-1]
    price = float(close.iloc[-1])
    fast = float(ema_fast.iloc[-1])
    slow = float(ema_slow.iloc[-1])
    trend = float(ema_trend.iloc[-1])
    rsi_value = float(rsi_14.iloc[-1])
    macd_hist_value = float(macd_hist.iloc[-1])
    rv_value = float(realized_vol.iloc[-1]) if pd.notna(realized_vol.iloc[-1]) else 0.0

    breakout_up = 0.0
    breakout_down = 0.0
    if pd.notna(rolling_high.iloc[-1]) and float(rolling_high.iloc[-1]) != 0.0:
        breakout_up = (price / float(rolling_high.iloc[-1]) - 1.0) * 100.0
    if pd.notna(rolling_low.iloc[-1]) and float(rolling_low.iloc[-1]) != 0.0:
        breakout_down = (price / float(rolling_low.iloc[-1]) - 1.0) * 100.0

    ema_gap_pct = (fast / max(slow, 1e-9) - 1.0) * 100.0
    trend_gap_pct = (price / max(trend, 1e-9) - 1.0) * 100.0

    score = 0.0
    score += 30.0 if price > trend else -30.0
    score += 20.0 if fast > slow else -20.0
    score += 15.0 if macd_hist_value > 0 else -15.0
    score += 15.0 if rsi_value > 55.0 else -15.0 if rsi_value < 45.0 else 0.0
    score += 10.0 if breakout_up > 0 else -10.0 if breakout_down < 0 else 0.0
    score += 10.0 if trend_gap_pct > 1.0 else -10.0 if trend_gap_pct < -1.0 else 0.0

    if rv_value > 1.2:
        score *= 0.75
        regime = "high_volatility"
    elif abs(trend_gap_pct) < 0.75 and abs(ema_gap_pct) < 0.25:
        regime = "range"
    else:
        regime = "trend"

    signal = "NO_TRADE"
    if score >= 35.0:
        signal = "LONG"
    elif score <= -35.0:
        signal = "SHORT"

    return BTCDirectionalSignal(
        timestamp=pd.to_datetime(latest["timestamp"], utc=True, errors="coerce"),
        symbol=symbol,
        signal=signal,
        score=round(score, 2),
        current_price=round(price, 4),
        ema_fast=round(fast, 4),
        ema_slow=round(slow, 4),
        ema_trend=round(trend, 4),
        ema_gap_pct=round(ema_gap_pct, 4),
        rsi_14=round(rsi_value, 2),
        macd_hist=round(macd_hist_value, 6),
        breakout_20_pct=round(breakout_up if signal != "SHORT" else breakout_down, 4),
        realized_volatility=round(rv_value, 4),
        regime=regime,
    )
