from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import pandas as pd
import requests
from requests import Response


BINANCE_KLINES_URL = "https://api.binance.com/api/v3/klines"
BINANCE_FUNDING_URL = "https://fapi.binance.com/fapi/v1/fundingRate"


@dataclass(frozen=True)
class DataConfig:
    symbols: tuple[str, ...]
    interval: str = "1h"
    limit: int = 1000
    benchmark_symbols: tuple[str, str] = ("BTCUSDT", "ETHUSDT")


def _session() -> requests.Session:
    session = requests.Session()
    session.trust_env = False
    return session


def fetch_klines(symbol: str, interval: str = "1h", limit: int = 1000) -> list[list]:
    max_batch = 1000
    remaining = max(limit, 1)
    end_time: int | None = None
    batches: list[list[list]] = []

    while remaining > 0:
        batch_limit = min(remaining, max_batch)
        params: dict[str, int | str] = {"symbol": symbol, "interval": interval, "limit": batch_limit}
        if end_time is not None:
            params["endTime"] = end_time

        response: Response = _session().get(BINANCE_KLINES_URL, params=params, timeout=20)
        response.raise_for_status()
        payload = response.json()
        if not payload:
            break

        batches.append(payload)
        remaining -= len(payload)
        oldest_open_time = int(payload[0][0])
        end_time = oldest_open_time - 1

        if len(payload) < batch_limit:
            break

    combined: list[list] = []
    for batch in reversed(batches):
        combined.extend(batch)
    return combined[-limit:]


def fetch_latest_funding_rate(symbol: str) -> float | None:
    params = {"symbol": symbol, "limit": 1}
    try:
        response: Response = _session().get(BINANCE_FUNDING_URL, params=params, timeout=20)
        response.raise_for_status()
        payload = response.json()
        if not payload:
            return None
        return float(payload[-1]["fundingRate"])
    except Exception:
        return None


def klines_to_market_df(symbol: str, klines: list[list]) -> pd.DataFrame:
    raw = pd.DataFrame(klines)
    frame = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(raw[0], unit="ms", utc=True),
            "symbol": symbol,
            "close": pd.to_numeric(raw[4], errors="coerce"),
            "volume": pd.to_numeric(raw[5], errors="coerce"),
            "quote_volume": pd.to_numeric(raw[7], errors="coerce"),
            "trade_count": pd.to_numeric(raw[8], errors="coerce"),
        }
    )
    return frame.dropna(subset=["close"]).sort_values("timestamp").reset_index(drop=True)


def fetch_market_data(config: DataConfig) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for symbol in config.benchmark_symbols + tuple(config.symbols):
        klines = fetch_klines(symbol=symbol, interval=config.interval, limit=config.limit)
        market_df = klines_to_market_df(symbol, klines)
        market_df["funding_rate"] = fetch_latest_funding_rate(symbol)
        rows.append(market_df)
    return pd.concat(rows, ignore_index=True)


def symbols_from_string(raw_symbols: str | None, default_symbols: Iterable[str]) -> tuple[str, ...]:
    if not raw_symbols:
        return tuple(default_symbols)
    return tuple(symbol.strip().upper() for symbol in raw_symbols.split(",") if symbol.strip())
