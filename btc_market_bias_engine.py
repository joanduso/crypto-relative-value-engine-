from __future__ import annotations

from dataclasses import dataclass
from io import StringIO
import math
import os

import numpy as np
import pandas as pd
import requests

from news_engine import SOURCE_WEIGHT, load_news_events


BINANCE_FUNDING_URL = "https://fapi.binance.com/fapi/v1/fundingRate"
BINANCE_OPEN_INTEREST_HIST_URL = "https://fapi.binance.com/futures/data/openInterestHist"
BINANCE_GLOBAL_LONG_SHORT_URL = "https://fapi.binance.com/futures/data/globalLongShortAccountRatio"
FRED_CSV_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
FEAR_GREED_URL = "https://api.alternative.me/fng/"
FARSIDE_BTC_ETF_URL = "https://farside.co.uk/bitcoin-etf-flow-all-data/"
GLASSNODE_BASE_URL = "https://api.glassnode.com"


@dataclass(frozen=True)
class BTCMarketBiasState:
    bias: str
    regime: str
    composite_score: float
    bull_prob_4h: float
    bull_prob_24h: float
    technical_score: float
    derivatives_score: float
    macro_score: float
    sentiment_score: float
    news_score: float
    etf_score: float
    onchain_score: float
    stress_score: float
    funding_rate: float
    funding_zscore: float
    open_interest_change_pct: float
    long_short_ratio: float
    fear_greed_value: float
    vix_level: float
    vix_5d_change_pct: float
    us10y_level: float
    us10y_5d_change_bps: float
    dollar_index_level: float
    dollar_5d_change_pct: float
    etf_net_flow_usd_m: float
    etf_flow_5d_usd_m: float
    mvrv_value: float
    sopr_value: float


def _default_state() -> BTCMarketBiasState:
    return BTCMarketBiasState(
        bias="NEUTRAL",
        regime="range",
        composite_score=0.0,
        bull_prob_4h=0.5,
        bull_prob_24h=0.5,
        technical_score=0.0,
        derivatives_score=0.0,
        macro_score=0.0,
        sentiment_score=0.0,
        news_score=0.0,
        etf_score=0.0,
        onchain_score=0.0,
        stress_score=50.0,
        funding_rate=0.0,
        funding_zscore=0.0,
        open_interest_change_pct=0.0,
        long_short_ratio=1.0,
        fear_greed_value=50.0,
        vix_level=0.0,
        vix_5d_change_pct=0.0,
        us10y_level=0.0,
        us10y_5d_change_bps=0.0,
        dollar_index_level=0.0,
        dollar_5d_change_pct=0.0,
        etf_net_flow_usd_m=0.0,
        etf_flow_5d_usd_m=0.0,
        mvrv_value=1.0,
        sopr_value=1.0,
    )


def _session() -> requests.Session:
    session = requests.Session()
    session.trust_env = False
    return session


def _safe_float(value: object, default: float = 0.0) -> float:
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(numeric):
        return default
    return float(numeric)


def _clip(value: float, low: float, high: float) -> float:
    return min(max(value, low), high)


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def _rsi(series: pd.Series, window: int = 14) -> pd.Series:
    delta = series.diff()
    gains = delta.clip(lower=0.0)
    losses = -delta.clip(upper=0.0)
    avg_gain = gains.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    avg_loss = losses.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi.fillna(50.0)


def _allowed_binance_period(interval: str) -> str:
    return interval if interval in {"5m", "15m", "30m", "1h", "2h", "4h", "6h", "12h", "1d"} else "1h"


def _fetch_binance_frame(url: str, params: dict[str, object]) -> pd.DataFrame:
    try:
        response = _session().get(url, params=params, timeout=20)
        response.raise_for_status()
        payload = response.json()
        if not payload:
            return pd.DataFrame()
        return pd.DataFrame(payload)
    except Exception:
        return pd.DataFrame()


def _fetch_funding_history(symbol: str = "BTCUSDT", limit: int = 256) -> pd.DataFrame:
    frame = _fetch_binance_frame(BINANCE_FUNDING_URL, {"symbol": symbol, "limit": max(limit, 8)})
    if frame.empty:
        return frame
    frame["timestamp"] = pd.to_datetime(frame["fundingTime"], unit="ms", utc=True, errors="coerce")
    frame["funding_rate"] = pd.to_numeric(frame["fundingRate"], errors="coerce")
    return frame.dropna(subset=["timestamp", "funding_rate"]).sort_values("timestamp").reset_index(drop=True)


def _fetch_open_interest_history(symbol: str = "BTCUSDT", period: str = "1h", limit: int = 256) -> pd.DataFrame:
    frame = _fetch_binance_frame(
        BINANCE_OPEN_INTEREST_HIST_URL,
        {"symbol": symbol, "period": period, "limit": max(limit, 8)},
    )
    if frame.empty:
        return frame
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], unit="ms", utc=True, errors="coerce")
    if "sumOpenInterestValue" in frame.columns:
        frame["open_interest_value"] = pd.to_numeric(frame["sumOpenInterestValue"], errors="coerce")
    else:
        frame["open_interest_value"] = pd.to_numeric(frame.get("sumOpenInterest"), errors="coerce")
    return frame.dropna(subset=["timestamp", "open_interest_value"]).sort_values("timestamp").reset_index(drop=True)


def _fetch_global_long_short_ratio(symbol: str = "BTCUSDT", period: str = "1h", limit: int = 256) -> pd.DataFrame:
    frame = _fetch_binance_frame(
        BINANCE_GLOBAL_LONG_SHORT_URL,
        {"symbol": symbol, "period": period, "limit": max(limit, 8)},
    )
    if frame.empty:
        return frame
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], unit="ms", utc=True, errors="coerce")
    frame["long_short_ratio"] = pd.to_numeric(frame["longShortRatio"], errors="coerce")
    return frame.dropna(subset=["timestamp", "long_short_ratio"]).sort_values("timestamp").reset_index(drop=True)


def _fetch_fred_series(series_id: str) -> pd.DataFrame:
    try:
        response = _session().get(FRED_CSV_URL.format(series_id=series_id), timeout=20)
        response.raise_for_status()
        frame = pd.read_csv(StringIO(response.text))
    except Exception:
        return pd.DataFrame()

    if frame.empty or "DATE" not in frame.columns or series_id not in frame.columns:
        return pd.DataFrame()

    frame["timestamp"] = pd.to_datetime(frame["DATE"], errors="coerce", utc=True)
    frame["value"] = pd.to_numeric(frame[series_id], errors="coerce")
    return frame.dropna(subset=["timestamp", "value"]).sort_values("timestamp").reset_index(drop=True)


def _fetch_fear_greed_value() -> float:
    try:
        response = _session().get(FEAR_GREED_URL, params={"limit": 1}, timeout=20)
        response.raise_for_status()
        payload = response.json()
        data = payload.get("data") or []
        if not data:
            return 50.0
        return _safe_float(data[0].get("value"), 50.0)
    except Exception:
        return 50.0


def _normalize_external_series(frame: pd.DataFrame, value_column: str) -> pd.DataFrame:
    if frame.empty or value_column not in frame.columns:
        return pd.DataFrame(columns=["timestamp", "value"])
    out = frame.copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True, errors="coerce")
    out["value"] = pd.to_numeric(out[value_column], errors="coerce")
    out = out.dropna(subset=["timestamp", "value"]).sort_values("timestamp")
    return out[["timestamp", "value"]].reset_index(drop=True)


def _parse_parentheses_number(value: object) -> float:
    text = str(value or "").strip()
    if not text or text == "-":
        return 0.0
    negative = text.startswith("(") and text.endswith(")")
    text = text.replace("(", "").replace(")", "").replace(",", "")
    numeric = _safe_float(text, 0.0)
    return -numeric if negative else numeric


def _load_series_from_csv(path: str, value_candidates: tuple[str, ...]) -> pd.DataFrame:
    if not path or not os.path.exists(path):
        return pd.DataFrame()
    try:
        frame = pd.read_csv(path)
    except Exception:
        return pd.DataFrame()

    timestamp_col = None
    for candidate in ("timestamp", "date", "Date", "DATE"):
        if candidate in frame.columns:
            timestamp_col = candidate
            break
    value_col = None
    for candidate in value_candidates:
        if candidate in frame.columns:
            value_col = candidate
            break
    if timestamp_col is None or value_col is None:
        return pd.DataFrame()
    out = frame.rename(columns={timestamp_col: "timestamp", value_col: "value"})
    return _normalize_external_series(out, "value")


def _fetch_farside_btc_etf_flows() -> pd.DataFrame:
    try:
        tables = pd.read_html(FARSIDE_BTC_ETF_URL)
    except Exception:
        return pd.DataFrame()
    if not tables:
        return pd.DataFrame()

    table = tables[0].copy()
    if "Date" not in table.columns or "Total" not in table.columns:
        return pd.DataFrame()
    table = table.loc[~table["Date"].astype(str).str.upper().isin({"TOTAL", "AVERAGE", "MAXIMUM", "MINIMUM"})].copy()
    table["timestamp"] = pd.to_datetime(table["Date"], errors="coerce", utc=True)
    table["value"] = table["Total"].apply(_parse_parentheses_number)
    return table.dropna(subset=["timestamp"]).sort_values("timestamp")[["timestamp", "value"]].reset_index(drop=True)


def _fetch_glassnode_series(url_path: str, asset: str = "BTC", interval: str = "24h") -> pd.DataFrame:
    api_key = os.getenv("GLASSNODE_API_KEY", "").strip()
    if not api_key or not url_path:
        return pd.DataFrame()

    url = url_path if url_path.startswith("http") else f"{GLASSNODE_BASE_URL.rstrip('/')}/{url_path.lstrip('/')}"
    try:
        response = _session().get(
            url,
            params={"a": asset, "i": interval, "api_key": api_key},
            timeout=20,
        )
        response.raise_for_status()
        payload = response.json()
        frame = pd.DataFrame(payload)
    except Exception:
        return pd.DataFrame()

    if frame.empty:
        return pd.DataFrame()

    timestamp_col = "t" if "t" in frame.columns else "timestamp"
    value_col = "v" if "v" in frame.columns else "value"
    if timestamp_col not in frame.columns or value_col not in frame.columns:
        return pd.DataFrame()
    out = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(frame[timestamp_col], unit="s", utc=True, errors="coerce"),
            "value": pd.to_numeric(frame[value_col], errors="coerce"),
        }
    )
    return out.dropna(subset=["timestamp", "value"]).sort_values("timestamp").reset_index(drop=True)


def _load_etf_flow_series() -> pd.DataFrame:
    csv_path = os.getenv("BTC_ETF_FLOWS_PATH", "btc_etf_flows.csv").strip()
    csv_frame = _load_series_from_csv(csv_path, ("value", "flow", "net_flow", "total", "Total"))
    if not csv_frame.empty:
        return csv_frame
    if os.getenv("BTC_ETF_REMOTE_ENABLED", "true").strip().lower() not in {"1", "true", "yes", "on"}:
        return pd.DataFrame()
    return _fetch_farside_btc_etf_flows()


def _load_onchain_series(csv_env: str, path_env: str) -> pd.DataFrame:
    csv_path = os.getenv(csv_env, "").strip()
    csv_frame = _load_series_from_csv(csv_path, ("value", "mvrv", "sopr"))
    if not csv_frame.empty:
        return csv_frame
    remote_path = os.getenv(path_env, "").strip()
    if not remote_path:
        return pd.DataFrame()
    return _fetch_glassnode_series(remote_path)


def _weighted_news_frame(index: pd.Series) -> pd.DataFrame:
    events = load_news_events()
    out = pd.DataFrame({"timestamp": pd.to_datetime(index, utc=True, errors="coerce")})
    out["news_score"] = 0.0
    if events.empty or out.empty:
        return out

    relevant = events.loc[
        events["symbol"].astype(str).str.upper().isin({"BTCUSDT", "ALL"})
        | events["market_scope"].astype(str).str.upper().isin({"MACRO", "MARKET", "ALL"})
    ].copy()
    if relevant.empty:
        return out

    relevant["timestamp"] = pd.to_datetime(relevant["timestamp"], utc=True, errors="coerce")
    relevant = relevant.dropna(subset=["timestamp"])
    relevant["sentiment_sign"] = relevant["sentiment"].astype(str).str.upper().map(
        {
            "BULL": 1.0,
            "BULLISH": 1.0,
            "POSITIVE": 1.0,
            "BEAR": -1.0,
            "BEARISH": -1.0,
            "NEGATIVE": -1.0,
        }
    ).fillna(0.0)
    relevant = relevant.loc[relevant["sentiment_sign"] != 0.0].copy()
    if relevant.empty:
        return out

    relevant["severity"] = pd.to_numeric(relevant["severity"], errors="coerce").fillna(0.5).clip(lower=0.0, upper=1.0)
    relevant["confidence"] = pd.to_numeric(relevant["confidence"], errors="coerce").fillna(0.5).clip(lower=0.0, upper=1.0)
    relevant["source_weight"] = relevant["source_tier"].astype(str).str.upper().map(SOURCE_WEIGHT).fillna(0.6)

    half_life_hours = float(os.getenv("BTC_BIAS_NEWS_HALF_LIFE_HOURS", "24"))
    scores: list[float] = []
    for ts in out["timestamp"]:
        age_hours = (
            (ts - relevant["timestamp"]).dt.total_seconds().div(3600.0).clip(lower=0.0)
        )
        decay = 0.5 ** (age_hours / max(half_life_hours, 1.0))
        impact = relevant["sentiment_sign"] * relevant["severity"] * relevant["confidence"] * relevant["source_weight"] * decay
        scores.append(_clip(float(impact.sum()) * 12.0, -20.0, 20.0))
    out["news_score"] = scores
    return out


def _merge_series_asof(base: pd.DataFrame, series_df: pd.DataFrame, column_name: str) -> pd.DataFrame:
    out = base.copy()
    if series_df.empty:
        out[column_name] = np.nan
        return out
    merged = pd.merge_asof(
        out.sort_values("timestamp"),
        series_df.rename(columns={"value": column_name}).sort_values("timestamp"),
        on="timestamp",
        direction="backward",
    )
    return merged


def build_btc_market_bias_frame(
    market_df: pd.DataFrame,
    *,
    interval: str = "1h",
) -> pd.DataFrame:
    btc = market_df.loc[market_df["symbol"] == "BTCUSDT", ["timestamp", "close"]].copy()
    if btc.empty:
        return pd.DataFrame()

    btc["timestamp"] = pd.to_datetime(btc["timestamp"], utc=True, errors="coerce")
    btc["close"] = pd.to_numeric(btc["close"], errors="coerce")
    btc = btc.dropna(subset=["timestamp", "close"]).sort_values("timestamp").reset_index(drop=True)
    if btc.empty:
        return pd.DataFrame()

    out = btc.rename(columns={"close": "btc_close"}).copy()
    close = out["btc_close"]
    returns = close.pct_change().fillna(0.0)
    out["ema20"] = close.ewm(span=20, adjust=False).mean()
    out["ema50"] = close.ewm(span=50, adjust=False).mean()
    out["ema200"] = close.ewm(span=200, adjust=False).mean()
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    out["macd_hist"] = (ema12 - ema26) - (ema12 - ema26).ewm(span=9, adjust=False).mean()
    out["rsi14"] = _rsi(close, 14)
    out["realized_volatility"] = returns.rolling(24, min_periods=12).std(ddof=0) * np.sqrt(24 * 365)
    out["return_24h_pct"] = close.pct_change(24) * 100.0

    period = _allowed_binance_period(interval)
    funding_history = _fetch_funding_history("BTCUSDT", limit=256)
    if funding_history.empty:
        funding_history = pd.DataFrame({"timestamp": out["timestamp"], "funding_rate": np.zeros(len(out))})
    else:
        funding_history = funding_history[["timestamp", "funding_rate"]].copy()

    oi_history = _fetch_open_interest_history("BTCUSDT", period=period, limit=256)
    if not oi_history.empty:
        oi_history["open_interest_change_pct"] = oi_history["open_interest_value"].pct_change(4) * 100.0

    long_short_history = _fetch_global_long_short_ratio("BTCUSDT", period=period, limit=256)

    out = pd.merge_asof(out, funding_history.sort_values("timestamp"), on="timestamp", direction="backward")
    funding_roll_mean = out["funding_rate"].rolling(21, min_periods=4).mean()
    funding_roll_std = out["funding_rate"].rolling(21, min_periods=4).std(ddof=0)
    out["funding_zscore"] = ((out["funding_rate"] - funding_roll_mean) / funding_roll_std.replace(0.0, np.nan)).fillna(0.0)

    if oi_history.empty:
        out["open_interest_change_pct"] = 0.0
    else:
        out = pd.merge_asof(
            out.sort_values("timestamp"),
            oi_history[["timestamp", "open_interest_change_pct"]].sort_values("timestamp"),
            on="timestamp",
            direction="backward",
        )
        out["open_interest_change_pct"] = pd.to_numeric(out["open_interest_change_pct"], errors="coerce").fillna(0.0)

    if long_short_history.empty:
        out["long_short_ratio"] = 1.0
    else:
        out = pd.merge_asof(
            out.sort_values("timestamp"),
            long_short_history[["timestamp", "long_short_ratio"]].sort_values("timestamp"),
            on="timestamp",
            direction="backward",
        )
        out["long_short_ratio"] = pd.to_numeric(out["long_short_ratio"], errors="coerce").fillna(1.0)

    vix_frame = _fetch_fred_series("VIXCLS")
    if not vix_frame.empty:
        vix_frame["vix_level"] = vix_frame["value"]
        vix_frame["vix_5d_change_pct"] = vix_frame["value"].pct_change(5) * 100.0
        out = pd.merge_asof(out.sort_values("timestamp"), vix_frame[["timestamp", "vix_level", "vix_5d_change_pct"]].sort_values("timestamp"), on="timestamp", direction="backward")
    else:
        out["vix_level"] = 0.0
        out["vix_5d_change_pct"] = 0.0

    us10y_frame = _fetch_fred_series("DGS10")
    if not us10y_frame.empty:
        us10y_frame["us10y_level"] = us10y_frame["value"]
        us10y_frame["us10y_5d_change_bps"] = us10y_frame["value"].diff(5) * 100.0
        out = pd.merge_asof(out.sort_values("timestamp"), us10y_frame[["timestamp", "us10y_level", "us10y_5d_change_bps"]].sort_values("timestamp"), on="timestamp", direction="backward")
    else:
        out["us10y_level"] = 0.0
        out["us10y_5d_change_bps"] = 0.0

    dollar_frame = _fetch_fred_series("DTWEXBGS")
    if not dollar_frame.empty:
        dollar_frame["dollar_index_level"] = dollar_frame["value"]
        dollar_frame["dollar_5d_change_pct"] = dollar_frame["value"].pct_change(5) * 100.0
        out = pd.merge_asof(out.sort_values("timestamp"), dollar_frame[["timestamp", "dollar_index_level", "dollar_5d_change_pct"]].sort_values("timestamp"), on="timestamp", direction="backward")
    else:
        out["dollar_index_level"] = 0.0
        out["dollar_5d_change_pct"] = 0.0

    etf_flows = _load_etf_flow_series()
    if etf_flows.empty:
        out["etf_net_flow_usd_m"] = 0.0
        out["etf_flow_5d_usd_m"] = 0.0
    else:
        etf_flows["etf_net_flow_usd_m"] = etf_flows["value"]
        etf_flows["etf_flow_5d_usd_m"] = etf_flows["value"].rolling(5, min_periods=1).sum()
        out = pd.merge_asof(out.sort_values("timestamp"), etf_flows[["timestamp", "etf_net_flow_usd_m", "etf_flow_5d_usd_m"]].sort_values("timestamp"), on="timestamp", direction="backward")
        out["etf_net_flow_usd_m"] = pd.to_numeric(out["etf_net_flow_usd_m"], errors="coerce").fillna(0.0)
        out["etf_flow_5d_usd_m"] = pd.to_numeric(out["etf_flow_5d_usd_m"], errors="coerce").fillna(0.0)

    mvrv_series = _load_onchain_series("BTC_MVRV_PATH", "GLASSNODE_MVRV_PATH")
    if mvrv_series.empty:
        out["mvrv_value"] = 1.0
    else:
        out = pd.merge_asof(out.sort_values("timestamp"), mvrv_series.rename(columns={"value": "mvrv_value"}).sort_values("timestamp"), on="timestamp", direction="backward")
        out["mvrv_value"] = pd.to_numeric(out["mvrv_value"], errors="coerce").fillna(1.0)

    sopr_series = _load_onchain_series("BTC_SOPR_PATH", "GLASSNODE_SOPR_PATH")
    if sopr_series.empty:
        out["sopr_value"] = 1.0
    else:
        out = pd.merge_asof(out.sort_values("timestamp"), sopr_series.rename(columns={"value": "sopr_value"}).sort_values("timestamp"), on="timestamp", direction="backward")
        out["sopr_value"] = pd.to_numeric(out["sopr_value"], errors="coerce").fillna(1.0)

    news_frame = _weighted_news_frame(out["timestamp"])
    out = pd.merge_asof(out.sort_values("timestamp"), news_frame.sort_values("timestamp"), on="timestamp", direction="backward")
    out["news_score"] = pd.to_numeric(out["news_score"], errors="coerce").fillna(0.0)

    current_fng = _fetch_fear_greed_value()
    out["fear_greed_value"] = float(os.getenv("BTC_BIAS_FEAR_GREED_DEFAULT", str(current_fng)))

    out["technical_score"] = 0.0
    out["technical_score"] += np.where(out["btc_close"] > out["ema200"], 25.0, -25.0)
    out["technical_score"] += np.where(out["ema50"] > out["ema200"], 15.0, -15.0)
    out["technical_score"] += np.where(out["ema20"] > out["ema50"], 10.0, -10.0)
    out["technical_score"] += np.where(out["macd_hist"] > 0, 10.0, -10.0)
    out["technical_score"] += np.where(out["rsi14"] > 55.0, 10.0, np.where(out["rsi14"] < 45.0, -10.0, 0.0))
    out["technical_score"] += np.where(out["return_24h_pct"] > 1.0, 10.0, np.where(out["return_24h_pct"] < -1.0, -10.0, 0.0))
    out["technical_score"] += np.where(out["realized_volatility"] > 1.15, -10.0, 0.0)
    out["technical_score"] = out["technical_score"].clip(lower=-100.0, upper=100.0)

    out["derivatives_score"] = 0.0
    out["derivatives_score"] += np.where(
        (out["return_24h_pct"] > 0.75) & (out["open_interest_change_pct"] > 1.5) & (out["funding_rate"] >= -0.0001) & (out["funding_zscore"] < 1.25),
        18.0,
        0.0,
    )
    out["derivatives_score"] += np.where(
        (out["return_24h_pct"] < -0.75) & (out["open_interest_change_pct"] > 1.5) & (out["funding_rate"] <= 0.0001),
        -18.0,
        0.0,
    )
    out["derivatives_score"] += np.where((out["funding_zscore"] > 1.75) & (out["open_interest_change_pct"] > 2.5), -18.0, 0.0)
    out["derivatives_score"] += np.where((out["funding_zscore"] < -1.25) & (out["return_24h_pct"] > 0.5), 10.0, 0.0)
    out["derivatives_score"] += np.where((out["long_short_ratio"] > 1.2) & (out["funding_zscore"] > 1.0), -10.0, 0.0)
    out["derivatives_score"] += np.where((out["long_short_ratio"] < 0.9) & (out["return_24h_pct"] > 0.0), 8.0, 0.0)
    out["derivatives_score"] = out["derivatives_score"].clip(lower=-100.0, upper=100.0)

    out["macro_score"] = 0.0
    out["macro_score"] += np.where(out["vix_level"] > 30.0, -15.0, np.where(out["vix_level"] > 22.0, -8.0, np.where((out["vix_level"] > 0) & (out["vix_level"] < 16.0), 5.0, 0.0)))
    out["macro_score"] += np.where(out["vix_5d_change_pct"] < -5.0, 10.0, np.where(out["vix_5d_change_pct"] > 5.0, -10.0, 0.0))
    out["macro_score"] += np.where(out["dollar_5d_change_pct"] < -0.5, 8.0, np.where(out["dollar_5d_change_pct"] > 0.5, -8.0, 0.0))
    out["macro_score"] += np.where(out["us10y_5d_change_bps"] < -10.0, 5.0, np.where(out["us10y_5d_change_bps"] > 10.0, -5.0, 0.0))
    out["macro_score"] = out["macro_score"].clip(lower=-100.0, upper=100.0)

    out["etf_score"] = 0.0
    out["etf_score"] += np.where(out["etf_net_flow_usd_m"] > 150.0, 10.0, np.where(out["etf_net_flow_usd_m"] < -150.0, -10.0, 0.0))
    out["etf_score"] += np.where(out["etf_flow_5d_usd_m"] > 500.0, 12.0, np.where(out["etf_flow_5d_usd_m"] < -500.0, -12.0, 0.0))
    out["etf_score"] = out["etf_score"].clip(lower=-100.0, upper=100.0)

    out["onchain_score"] = 0.0
    out["onchain_score"] += np.where(out["mvrv_value"] < 0.95, 10.0, np.where(out["mvrv_value"] > 2.4, -10.0, 0.0))
    out["onchain_score"] += np.where(out["sopr_value"] > 1.01, 8.0, np.where(out["sopr_value"] < 0.99, -8.0, 0.0))
    out["onchain_score"] = out["onchain_score"].clip(lower=-100.0, upper=100.0)

    out["sentiment_score"] = out["news_score"]
    out["sentiment_score"] += np.where(out["fear_greed_value"] >= 62.0, 12.0, np.where(out["fear_greed_value"] <= 38.0, -12.0, 0.0))
    out["sentiment_score"] += np.where(out["fear_greed_value"] >= 85.0, -6.0, 0.0)
    out["sentiment_score"] = out["sentiment_score"].clip(lower=-100.0, upper=100.0)

    out["stress_score"] = 20.0
    out["stress_score"] += np.where(out["btc_close"] < out["ema200"], 20.0, 0.0)
    out["stress_score"] += np.where(out["realized_volatility"] > 1.25, 20.0, np.where(out["realized_volatility"] > 0.95, 8.0, 0.0))
    out["stress_score"] += np.where(out["vix_level"] > 28.0, 18.0, np.where(out["vix_level"] > 22.0, 10.0, 0.0))
    out["stress_score"] += np.where((out["funding_zscore"] > 1.75) & (out["open_interest_change_pct"] > 2.5), 15.0, 0.0)
    out["stress_score"] += np.where(out["news_score"] < -8.0, 10.0, 0.0)
    out["stress_score"] += np.where(out["etf_flow_5d_usd_m"] < -500.0, 8.0, 0.0)
    out["stress_score"] = out["stress_score"].clip(lower=0.0, upper=100.0)

    out["composite_score"] = (
        0.35 * out["technical_score"]
        + 0.22 * out["derivatives_score"]
        + 0.13 * out["macro_score"]
        + 0.12 * out["sentiment_score"]
        + 0.10 * out["etf_score"]
        + 0.08 * out["onchain_score"]
        - np.maximum(out["stress_score"] - 55.0, 0.0) * 0.2
    ).clip(lower=-100.0, upper=100.0)
    out["bull_prob_4h"] = out["composite_score"].apply(lambda value: _clip(_sigmoid(float(value) / 12.0), 0.0, 1.0))
    out["bull_prob_24h"] = (
        (0.7 * out["composite_score"] + 0.3 * out["technical_score"])
        .apply(lambda value: _clip(_sigmoid(float(value) / 12.0), 0.0, 1.0))
    )
    out["bias"] = np.where(
        (out["composite_score"] >= 15.0) & (out["bull_prob_24h"] >= 0.58),
        "BULLISH",
        np.where((out["composite_score"] <= -15.0) & (out["bull_prob_24h"] <= 0.42), "BEARISH", "NEUTRAL"),
    )
    out["regime"] = np.where(
        out["stress_score"] >= 70.0,
        "stress",
        np.where(
            (out["bias"] == "BULLISH") & (out["technical_score"] > 0),
            "bullish",
            np.where((out["bias"] == "BEARISH") & (out["technical_score"] < 0), "bearish", "range"),
        ),
    )

    numeric_columns = [
        "funding_rate",
        "funding_zscore",
        "open_interest_change_pct",
        "long_short_ratio",
        "fear_greed_value",
        "vix_level",
        "vix_5d_change_pct",
        "us10y_level",
        "us10y_5d_change_bps",
        "dollar_index_level",
        "dollar_5d_change_pct",
        "news_score",
        "technical_score",
        "derivatives_score",
        "macro_score",
        "sentiment_score",
        "etf_score",
        "onchain_score",
        "stress_score",
        "composite_score",
        "bull_prob_4h",
        "bull_prob_24h",
        "etf_net_flow_usd_m",
        "etf_flow_5d_usd_m",
        "mvrv_value",
        "sopr_value",
    ]
    for column in numeric_columns:
        out[column] = pd.to_numeric(out[column], errors="coerce").fillna(0.0)

    return out.reset_index(drop=True)


def build_btc_market_bias(
    market_df: pd.DataFrame,
    *,
    interval: str = "1h",
) -> BTCMarketBiasState:
    frame = build_btc_market_bias_frame(market_df, interval=interval)
    if frame.empty:
        return _default_state()

    latest = frame.iloc[-1]
    return BTCMarketBiasState(
        bias=str(latest.get("bias", "NEUTRAL")),
        regime=str(latest.get("regime", "range")),
        composite_score=round(_safe_float(latest.get("composite_score")), 4),
        bull_prob_4h=round(_safe_float(latest.get("bull_prob_4h"), 0.5), 4),
        bull_prob_24h=round(_safe_float(latest.get("bull_prob_24h"), 0.5), 4),
        technical_score=round(_safe_float(latest.get("technical_score")), 4),
        derivatives_score=round(_safe_float(latest.get("derivatives_score")), 4),
        macro_score=round(_safe_float(latest.get("macro_score")), 4),
        sentiment_score=round(_safe_float(latest.get("sentiment_score")), 4),
        news_score=round(_safe_float(latest.get("news_score")), 4),
        etf_score=round(_safe_float(latest.get("etf_score")), 4),
        onchain_score=round(_safe_float(latest.get("onchain_score")), 4),
        stress_score=round(_safe_float(latest.get("stress_score"), 50.0), 4),
        funding_rate=round(_safe_float(latest.get("funding_rate")), 8),
        funding_zscore=round(_safe_float(latest.get("funding_zscore")), 4),
        open_interest_change_pct=round(_safe_float(latest.get("open_interest_change_pct")), 4),
        long_short_ratio=round(_safe_float(latest.get("long_short_ratio"), 1.0), 4),
        fear_greed_value=round(_safe_float(latest.get("fear_greed_value"), 50.0), 4),
        vix_level=round(_safe_float(latest.get("vix_level")), 4),
        vix_5d_change_pct=round(_safe_float(latest.get("vix_5d_change_pct")), 4),
        us10y_level=round(_safe_float(latest.get("us10y_level")), 4),
        us10y_5d_change_bps=round(_safe_float(latest.get("us10y_5d_change_bps")), 4),
        dollar_index_level=round(_safe_float(latest.get("dollar_index_level")), 4),
        dollar_5d_change_pct=round(_safe_float(latest.get("dollar_5d_change_pct")), 4),
        etf_net_flow_usd_m=round(_safe_float(latest.get("etf_net_flow_usd_m")), 4),
        etf_flow_5d_usd_m=round(_safe_float(latest.get("etf_flow_5d_usd_m")), 4),
        mvrv_value=round(_safe_float(latest.get("mvrv_value"), 1.0), 4),
        sopr_value=round(_safe_float(latest.get("sopr_value"), 1.0), 4),
    )
