from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

from signal_engine import market_score_to_quality


DEFAULT_NEWS_EVENTS_PATH = Path(os.getenv("NEWS_EVENTS_PATH", "news_events.csv"))
SOURCE_WEIGHT = {
    "OFFICIAL": 1.0,
    "AGGREGATOR": 0.8,
    "MEDIA": 0.6,
    "SOCIAL": 0.35,
}
NEWS_IMPACT_CAP = 15.0


def _normalize_unit_interval(value: object, default: float = 0.5) -> float:
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(numeric):
        return default
    numeric = float(numeric)
    if numeric > 1.0:
        numeric /= 100.0
    return min(max(numeric, 0.0), 1.0)


def _recent_return_pct(market_df: pd.DataFrame, symbol: str, lookback_bars: int) -> float | None:
    subset = market_df.loc[market_df["symbol"] == symbol, ["timestamp", "close"]].copy()
    if subset.empty or len(subset) <= lookback_bars:
        return None
    subset = subset.sort_values("timestamp")
    latest_close = pd.to_numeric(subset.iloc[-1]["close"], errors="coerce")
    previous_close = pd.to_numeric(subset.iloc[-(lookback_bars + 1)]["close"], errors="coerce")
    if pd.isna(latest_close) or pd.isna(previous_close) or float(previous_close) <= 0.0:
        return None
    return (float(latest_close) / float(previous_close) - 1.0) * 100.0


def _recency_decay(event_timestamp: pd.Timestamp, reference_timestamp: pd.Timestamp, half_life_hours: float) -> float:
    if pd.isna(event_timestamp):
        return 0.0
    age_hours = max((reference_timestamp - event_timestamp).total_seconds() / 3600.0, 0.0)
    safe_half_life = max(half_life_hours, 1.0)
    return 0.5 ** (age_hours / safe_half_life)


def _source_weight(source_tier: object) -> float:
    tier = str(source_tier or "MEDIA").strip().upper()
    return SOURCE_WEIGHT.get(tier, SOURCE_WEIGHT["MEDIA"])


def _sentiment_sign(sentiment: object) -> int:
    value = str(sentiment or "").strip().upper()
    if value in {"BULL", "BULLISH", "POSITIVE", "GOOD"}:
        return 1
    if value in {"BEAR", "BEARISH", "NEGATIVE", "BAD"}:
        return -1
    return 0


def _market_confirmation(sentiment_sign: int, recent_return_pct: float | None) -> float:
    if sentiment_sign == 0:
        return 0.0
    if recent_return_pct is None:
        return 0.75
    if sentiment_sign > 0:
        if recent_return_pct >= 1.0:
            return 1.0
        if recent_return_pct >= 0.0:
            return 0.85
        if recent_return_pct >= -1.5:
            return 0.65
        return 0.45
    if recent_return_pct <= -1.0:
        return 1.0
    if recent_return_pct <= 0.0:
        return 0.85
    if recent_return_pct <= 1.5:
        return 0.65
    return 0.45


def _event_relevance(symbol: str, event_symbol: object, market_scope: object) -> float:
    normalized_symbol = symbol.strip().upper()
    normalized_event_symbol = str(event_symbol or "").strip().upper()
    normalized_scope = str(market_scope or "").strip().upper()

    if normalized_event_symbol == normalized_symbol:
        return 1.0
    if normalized_event_symbol in {"ALL", "MARKET", "TOTAL"} or normalized_scope in {"ALL", "MARKET", "MACRO"}:
        return 0.55
    if normalized_event_symbol == "BTCUSDT":
        return 0.7
    if normalized_event_symbol == "ETHUSDT":
        return 0.55
    if normalized_scope in {"L1", "ALT", "ALTS", "ALTCOINS"}:
        return 0.45
    return 0.0


def load_news_events(path: Path | None = None) -> pd.DataFrame:
    news_path = path or DEFAULT_NEWS_EVENTS_PATH
    if not news_path.exists():
        return pd.DataFrame()

    try:
        events = pd.read_csv(news_path)
    except Exception:
        return pd.DataFrame()

    if events.empty:
        return events

    out = events.copy()
    if "timestamp" in out.columns:
        out["timestamp"] = pd.to_datetime(out["timestamp"], errors="coerce", utc=True)
    else:
        out["timestamp"] = pd.NaT

    if "symbol" not in out.columns:
        out["symbol"] = "ALL"
    if "market_scope" not in out.columns:
        out["market_scope"] = ""
    if "sentiment" not in out.columns:
        out["sentiment"] = "NEUTRAL"
    if "source_tier" not in out.columns:
        out["source_tier"] = "MEDIA"
    if "severity" not in out.columns:
        out["severity"] = 0.5
    if "confidence" not in out.columns:
        out["confidence"] = 0.5
    if "event_type" not in out.columns:
        out["event_type"] = "headline"
    if "headline" not in out.columns:
        out["headline"] = ""

    return out.dropna(subset=["timestamp"]).reset_index(drop=True)


def news_comment(news_impact_score: object, news_event_count: object, news_bias: object) -> str:
    impact = pd.to_numeric(pd.Series([news_impact_score]), errors="coerce").iloc[0]
    events = pd.to_numeric(pd.Series([news_event_count]), errors="coerce").iloc[0]
    bias = str(news_bias or "NEUTRAL").strip().upper()

    impact_value = 0.0 if pd.isna(impact) else float(impact)
    event_count = 0 if pd.isna(events) else int(events)

    if event_count <= 0:
        return "Sin noticias recientes con impacto relevante."
    if bias == "BULLISH" and impact_value >= 6.0:
        return "Catalizador positivo fuerte por noticias recientes."
    if bias == "BULLISH" and impact_value >= 2.0:
        return "Sesgo positivo moderado por flujo de noticias."
    if bias == "BEARISH" and impact_value <= -6.0:
        return "Presion negativa fuerte por noticias recientes."
    if bias == "BEARISH" and impact_value <= -2.0:
        return "Sesgo negativo moderado por flujo de noticias."
    return "Impacto de noticias mixto o de baja intensidad."


def apply_news_overlay(
    ranked_universe: pd.DataFrame,
    market_df: pd.DataFrame,
    *,
    path: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if ranked_universe.empty:
        return ranked_universe.copy(), pd.DataFrame()

    out = ranked_universe.copy()
    out["pre_news_market_score"] = pd.to_numeric(out["market_opportunity_score"], errors="coerce").fillna(0.0)
    out["news_event_count"] = 0
    out["news_bullish_score"] = 0.0
    out["news_bearish_score"] = 0.0
    out["news_impact_score"] = 0.0
    out["news_bias"] = "NEUTRAL"
    out["news_comment"] = news_comment(0.0, 0, "NEUTRAL")

    events = load_news_events(path)
    if events.empty:
        out["market_opportunity_score"] = out["pre_news_market_score"]
        out["signal_quality"] = out["market_opportunity_score"].apply(market_score_to_quality)
        return out, events

    reference_timestamp = pd.Timestamp.utcnow()
    if reference_timestamp.tzinfo is None:
        reference_timestamp = reference_timestamp.tz_localize("UTC")

    lookback_bars = int(os.getenv("NEWS_CONFIRMATION_LOOKBACK_BARS", "6"))
    half_life_hours = float(os.getenv("NEWS_HALF_LIFE_HOURS", "18"))

    news_rows: list[dict[str, object]] = []
    for row in out.itertuples(index=False):
        symbol = str(row.symbol)
        recent_return_pct = _recent_return_pct(market_df, symbol, lookback_bars)
        relevant_events = events.copy()
        relevant_events["relevance"] = relevant_events.apply(
            lambda event: _event_relevance(symbol, event.get("symbol"), event.get("market_scope")),
            axis=1,
        )
        relevant_events = relevant_events.loc[relevant_events["relevance"] > 0].copy()
        if relevant_events.empty:
            continue

        relevant_events["sentiment_sign"] = relevant_events["sentiment"].apply(_sentiment_sign)
        relevant_events = relevant_events.loc[relevant_events["sentiment_sign"] != 0].copy()
        if relevant_events.empty:
            continue

        relevant_events["source_weight"] = relevant_events["source_tier"].apply(_source_weight)
        relevant_events["severity_norm"] = relevant_events["severity"].apply(_normalize_unit_interval)
        relevant_events["confidence_norm"] = relevant_events["confidence"].apply(_normalize_unit_interval)
        relevant_events["recency_decay"] = relevant_events["timestamp"].apply(
            lambda ts: _recency_decay(ts, reference_timestamp, half_life_hours)
        )
        relevant_events["market_confirmation"] = relevant_events["sentiment_sign"].apply(
            lambda sign: _market_confirmation(sign, recent_return_pct)
        )
        relevant_events["impact_points"] = (
            relevant_events["sentiment_sign"]
            * NEWS_IMPACT_CAP
            * relevant_events["source_weight"]
            * relevant_events["severity_norm"]
            * relevant_events["confidence_norm"]
            * relevant_events["relevance"]
            * relevant_events["recency_decay"]
            * relevant_events["market_confirmation"]
        )

        bullish_score = float(relevant_events.loc[relevant_events["impact_points"] > 0, "impact_points"].sum())
        bearish_score = float(-relevant_events.loc[relevant_events["impact_points"] < 0, "impact_points"].sum())
        impact_score = max(min(float(relevant_events["impact_points"].sum()), NEWS_IMPACT_CAP), -NEWS_IMPACT_CAP)
        news_rows.append(
            {
                "symbol": symbol,
                "news_event_count": int(len(relevant_events)),
                "news_bullish_score": round(bullish_score, 4),
                "news_bearish_score": round(bearish_score, 4),
                "news_impact_score": round(impact_score, 4),
                "news_bias": "BULLISH" if impact_score > 0.35 else "BEARISH" if impact_score < -0.35 else "NEUTRAL",
                "news_comment": news_comment(
                    round(impact_score, 4),
                    int(len(relevant_events)),
                    "BULLISH" if impact_score > 0.35 else "BEARISH" if impact_score < -0.35 else "NEUTRAL",
                ),
            }
        )

    if news_rows:
        out = out.drop(columns=["news_event_count", "news_bullish_score", "news_bearish_score", "news_impact_score", "news_bias", "news_comment"])
        out = out.merge(pd.DataFrame(news_rows), on="symbol", how="left")
        out["news_event_count"] = pd.to_numeric(out["news_event_count"], errors="coerce").fillna(0).astype(int)
        out["news_bullish_score"] = pd.to_numeric(out["news_bullish_score"], errors="coerce").fillna(0.0)
        out["news_bearish_score"] = pd.to_numeric(out["news_bearish_score"], errors="coerce").fillna(0.0)
        out["news_impact_score"] = pd.to_numeric(out["news_impact_score"], errors="coerce").fillna(0.0)
        out["news_bias"] = out["news_bias"].fillna("NEUTRAL")
        out["news_comment"] = out["news_comment"].fillna("Sin noticias recientes con impacto relevante.")

    out["market_opportunity_score"] = (
        out["pre_news_market_score"] + out["news_impact_score"]
    ).clip(lower=0.0, upper=100.0)
    out["signal_quality"] = out["market_opportunity_score"].apply(market_score_to_quality)
    out = out.sort_values(
        by=["market_opportunity_score", "confidence_score", "edge_after_fees_pct"],
        ascending=[False, False, False],
    ).reset_index(drop=True)
    return out, events
