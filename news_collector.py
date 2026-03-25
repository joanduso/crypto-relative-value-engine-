from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import timedelta
import os
from pathlib import Path
from typing import Iterable
from urllib.parse import urlparse
import xml.etree.ElementTree as ET

import pandas as pd
import requests


DEFAULT_OUTPUT_PATH = Path(os.getenv("NEWS_EVENTS_PATH", "news_events.csv"))
DEFAULT_RSS_FEEDS = (
    "https://www.coindesk.com/arc/outboundfeeds/rss/",
    "https://cointelegraph.com/rss",
)
ALTERNATIVE_ME_FNG_URL = "https://api.alternative.me/fng/"
DEFAULT_SYMBOLS = (
    "BTCUSDT",
    "ETHUSDT",
    "XRPUSDT",
    "SOLUSDT",
    "ADAUSDT",
    "DOGEUSDT",
    "BNBUSDT",
    "LTCUSDT",
    "LINKUSDT",
    "AVAXUSDT",
)

SOURCE_TIER_BY_DOMAIN = {
    "binance.com": "OFFICIAL",
    "www.binance.com": "OFFICIAL",
    "sec.gov": "OFFICIAL",
    "www.sec.gov": "OFFICIAL",
    "cftc.gov": "OFFICIAL",
    "www.cftc.gov": "OFFICIAL",
    "federalreserve.gov": "OFFICIAL",
    "www.federalreserve.gov": "OFFICIAL",
    "coindesk.com": "MEDIA",
    "www.coindesk.com": "MEDIA",
    "cointelegraph.com": "MEDIA",
    "www.cointelegraph.com": "MEDIA",
    "cryptopanic.com": "AGGREGATOR",
    "www.cryptopanic.com": "AGGREGATOR",
    "alternative.me": "AGGREGATOR",
    "www.alternative.me": "AGGREGATOR",
}

BULLISH_RULES = {
    "listing": ("listing", 0.95),
    "listed": ("listing", 0.95),
    "launch": ("launch", 0.70),
    "mainnet": ("upgrade", 0.75),
    "upgrade": ("upgrade", 0.65),
    "approval": ("regulation", 0.80),
    "approved": ("regulation", 0.80),
    "etf": ("etf", 0.85),
    "partnership": ("partnership", 0.60),
    "integrat": ("integration", 0.60),
    "institutional": ("adoption", 0.65),
}

BEARISH_RULES = {
    "delist": ("delisting", 0.95),
    "hack": ("hack", 0.98),
    "exploit": ("hack", 0.98),
    "breach": ("hack", 0.95),
    "lawsuit": ("regulation", 0.88),
    "sued": ("regulation", 0.88),
    "charges": ("regulation", 0.85),
    "settlement": ("regulation", 0.78),
    "outage": ("outage", 0.72),
    "halt": ("halt", 0.82),
    "bankruptcy": ("insolvency", 0.98),
    "withdrawal": ("liquidity", 0.85),
    "liquidat": ("liquidation", 0.80),
    "restrict": ("regulation", 0.72),
}


@dataclass(frozen=True)
class NewsItem:
    timestamp: pd.Timestamp
    symbol: str
    market_scope: str
    event_type: str
    sentiment: str
    source_tier: str
    severity: float
    confidence: float
    headline: str
    url: str


def _session() -> requests.Session:
    session = requests.Session()
    session.trust_env = False
    return session


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect market news and export normalized news events")
    parser.add_argument("--output-path", default=str(DEFAULT_OUTPUT_PATH))
    parser.add_argument("--feeds", default=os.getenv("NEWS_RSS_FEEDS", ",".join(DEFAULT_RSS_FEEDS)))
    parser.add_argument("--lookback-hours", type=int, default=int(os.getenv("NEWS_LOOKBACK_HOURS", "36")))
    parser.add_argument("--max-items-per-feed", type=int, default=int(os.getenv("NEWS_MAX_ITEMS_PER_FEED", "25")))
    parser.add_argument("--symbols", default=os.getenv("NEWS_SYMBOLS", ",".join(DEFAULT_SYMBOLS)))
    parser.add_argument("--include-fear-greed", dest="include_fear_greed", action="store_true")
    parser.add_argument("--skip-fear-greed", dest="include_fear_greed", action="store_false")
    parser.set_defaults(include_fear_greed=_env_bool("NEWS_INCLUDE_FEAR_GREED", True))
    return parser.parse_args()


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _normalize_symbols(raw: str | None) -> tuple[str, ...]:
    if not raw:
        return DEFAULT_SYMBOLS
    return tuple(part.strip().upper() for part in raw.replace(";", ",").split(",") if part.strip())


def _domain(url: str) -> str:
    return urlparse(url).netloc.lower()


def _source_tier_from_url(url: str) -> str:
    return SOURCE_TIER_BY_DOMAIN.get(_domain(url), "MEDIA")


def _symbol_aliases(symbols: Iterable[str]) -> dict[str, str]:
    aliases: dict[str, str] = {}
    for symbol in symbols:
        normalized = symbol.strip().upper()
        base = normalized.removesuffix("USDT")
        aliases[normalized] = normalized
        aliases[base] = normalized
    aliases["BITCOIN"] = "BTCUSDT"
    aliases["ETHER"] = "ETHUSDT"
    aliases["ETHEREUM"] = "ETHUSDT"
    aliases["SOLANA"] = "SOLUSDT"
    aliases["RIPPLE"] = "XRPUSDT"
    aliases["CARDANO"] = "ADAUSDT"
    aliases["DOGECOIN"] = "DOGEUSDT"
    aliases["BINANCE COIN"] = "BNBUSDT"
    aliases["CHAINLINK"] = "LINKUSDT"
    aliases["AVALANCHE"] = "AVAXUSDT"
    return aliases


def _extract_symbol(headline: str, aliases: dict[str, str]) -> str:
    normalized = headline.upper()
    for alias, symbol in aliases.items():
        if alias in normalized:
            return symbol
    return "ALL"


def _classify_headline(headline: str) -> tuple[str, str, float]:
    normalized = headline.lower()
    best_sentiment = "NEUTRAL"
    best_event_type = "headline"
    best_severity = 0.0

    for needle, (event_type, severity) in BULLISH_RULES.items():
        if needle in normalized and severity > best_severity:
            best_sentiment = "BULLISH"
            best_event_type = event_type
            best_severity = severity

    for needle, (event_type, severity) in BEARISH_RULES.items():
        if needle in normalized and severity > best_severity:
            best_sentiment = "BEARISH"
            best_event_type = event_type
            best_severity = severity

    if best_sentiment == "NEUTRAL":
        return "headline", "NEUTRAL", 0.35
    return best_event_type, best_sentiment, best_severity


def _confidence_from_source(source_tier: str) -> float:
    if source_tier == "OFFICIAL":
        return 0.95
    if source_tier == "AGGREGATOR":
        return 0.80
    if source_tier == "MEDIA":
        return 0.65
    return 0.45


def _safe_timestamp(value: str | None) -> pd.Timestamp | None:
    if not value:
        return None
    parsed = pd.to_datetime(value, errors="coerce", utc=True)
    if pd.isna(parsed):
        return None
    return parsed


def _rss_items(feed_url: str, *, lookback_hours: int, max_items: int, symbols: tuple[str, ...]) -> list[NewsItem]:
    response = _session().get(feed_url, timeout=20)
    response.raise_for_status()
    root = ET.fromstring(response.content)
    aliases = _symbol_aliases(symbols)
    cutoff = pd.Timestamp.utcnow()
    if cutoff.tzinfo is None:
        cutoff = cutoff.tz_localize("UTC")
    cutoff -= timedelta(hours=max(lookback_hours, 1))

    items: list[NewsItem] = []
    for item in root.findall(".//item"):
        title = (item.findtext("title") or "").strip()
        link = (item.findtext("link") or "").strip()
        pub_date = _safe_timestamp(item.findtext("pubDate")) or _safe_timestamp(item.findtext("{http://www.w3.org/2005/Atom}updated"))
        if not title or not link or pub_date is None or pub_date < cutoff:
            continue

        event_type, sentiment, severity = _classify_headline(title)
        if sentiment == "NEUTRAL":
            continue

        symbol = _extract_symbol(title, aliases)
        market_scope = "MACRO" if symbol == "ALL" else ""
        source_tier = _source_tier_from_url(link)
        confidence = _confidence_from_source(source_tier)
        items.append(
            NewsItem(
                timestamp=pub_date,
                symbol=symbol,
                market_scope=market_scope,
                event_type=event_type,
                sentiment=sentiment,
                source_tier=source_tier,
                severity=severity,
                confidence=confidence,
                headline=title,
                url=link,
            )
        )
        if len(items) >= max_items:
            break
    return items


def _fear_greed_item() -> NewsItem | None:
    response = _session().get(ALTERNATIVE_ME_FNG_URL, params={"limit": 1}, timeout=20)
    response.raise_for_status()
    payload = response.json()
    data = payload.get("data") or []
    if not data:
        return None

    latest = data[0]
    value = int(latest.get("value", 50))
    label = str(latest.get("value_classification", "Neutral")).strip()
    timestamp = pd.to_datetime(int(latest.get("timestamp")), unit="s", utc=True, errors="coerce")
    if pd.isna(timestamp):
        return None

    if value >= 60:
        sentiment = "BULLISH"
    elif value <= 40:
        sentiment = "BEARISH"
    else:
        sentiment = "NEUTRAL"

    if sentiment == "NEUTRAL":
        return None

    severity = min(abs(value - 50) / 50.0, 1.0)
    return NewsItem(
        timestamp=timestamp,
        symbol="ALL",
        market_scope="MACRO",
        event_type="sentiment_index",
        sentiment=sentiment,
        source_tier="AGGREGATOR",
        severity=round(severity, 4),
        confidence=0.8,
        headline=f"Alternative.me Fear and Greed Index at {value} ({label})",
        url="https://alternative.me/crypto/fear-and-greed-index/",
    )


def _items_to_frame(items: Iterable[NewsItem]) -> pd.DataFrame:
    rows = [
        {
            "timestamp": item.timestamp,
            "symbol": item.symbol,
            "market_scope": item.market_scope,
            "event_type": item.event_type,
            "sentiment": item.sentiment,
            "source_tier": item.source_tier,
            "severity": item.severity,
            "confidence": item.confidence,
            "headline": item.headline,
            "url": item.url,
        }
        for item in items
    ]
    return pd.DataFrame(rows)


def _load_existing(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        df = pd.read_csv(path)
    except Exception:
        return pd.DataFrame()
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
    return df


def _merge_events(existing: pd.DataFrame, fresh: pd.DataFrame, *, keep_hours: int) -> pd.DataFrame:
    frames = [frame for frame in (existing, fresh) if not frame.empty]
    if not frames:
        return pd.DataFrame(
            columns=["timestamp", "symbol", "market_scope", "event_type", "sentiment", "source_tier", "severity", "confidence", "headline", "url"]
        )

    merged = pd.concat(frames, ignore_index=True)
    merged["timestamp"] = pd.to_datetime(merged["timestamp"], errors="coerce", utc=True)
    merged = merged.dropna(subset=["timestamp", "headline"])
    if "url" not in merged.columns:
        merged["url"] = ""
    merged["dedupe_key"] = merged["url"].fillna("").astype(str).str.strip()
    merged["dedupe_key"] = merged["dedupe_key"].where(merged["dedupe_key"] != "", merged["headline"].astype(str).str.lower().str.strip())
    merged = merged.sort_values("timestamp", ascending=False).drop_duplicates(subset=["dedupe_key"], keep="first")
    cutoff = pd.Timestamp.utcnow()
    if cutoff.tzinfo is None:
        cutoff = cutoff.tz_localize("UTC")
    cutoff -= timedelta(hours=max(keep_hours, 1))
    merged = merged.loc[merged["timestamp"] >= cutoff].copy()
    return merged.drop(columns=["dedupe_key"]).sort_values("timestamp", ascending=False).reset_index(drop=True)


def collect_news_events(
    *,
    output_path: Path = DEFAULT_OUTPUT_PATH,
    feeds: Iterable[str] = DEFAULT_RSS_FEEDS,
    lookback_hours: int = 36,
    max_items_per_feed: int = 25,
    symbols: Iterable[str] = DEFAULT_SYMBOLS,
    include_fear_greed: bool = True,
) -> pd.DataFrame:
    normalized_symbols = tuple(str(symbol).strip().upper() for symbol in symbols if str(symbol).strip()) or DEFAULT_SYMBOLS
    items: list[NewsItem] = []
    for feed in feeds:
        if not str(feed).strip():
            continue
        try:
            items.extend(
                _rss_items(
                    str(feed).strip(),
                    lookback_hours=lookback_hours,
                    max_items=max_items_per_feed,
                    symbols=normalized_symbols,
                )
            )
        except Exception:
            continue

    if include_fear_greed:
        try:
            fear_greed_item = _fear_greed_item()
            if fear_greed_item is not None:
                items.append(fear_greed_item)
        except Exception:
            pass

    fresh = _items_to_frame(items)
    merged = _merge_events(_load_existing(output_path), fresh, keep_hours=max(lookback_hours * 3, 72))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output_path, index=False)
    return merged


def main() -> None:
    args = _parse_args()
    output_path = Path(args.output_path)
    feeds = [feed.strip() for feed in str(args.feeds).split(",") if feed.strip()]
    merged = collect_news_events(
        output_path=output_path,
        feeds=feeds,
        lookback_hours=args.lookback_hours,
        max_items_per_feed=args.max_items_per_feed,
        symbols=_normalize_symbols(args.symbols),
        include_fear_greed=args.include_fear_greed,
    )
    print(f"news_events_written={len(merged)} path={output_path}")


if __name__ == "__main__":
    main()
