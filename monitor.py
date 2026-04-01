from __future__ import annotations

import argparse
import logging
import os
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests
from requests import Response

from alerting import (
    append_daily_alert_snapshot,
    daily_best_alerts,
    email_config_from_env,
    filter_new_alerts,
    format_alert_email,
    format_alert_telegram,
    format_downside_shocks_telegram,
    format_market_summary_telegram,
    format_strategy_risk_telegram,
    format_top_opportunities_telegram,
    load_sent_alerts,
    parse_quality_list,
    save_sent_alerts,
    select_alert_candidates,
    send_email_alert,
    send_telegram_alert,
    telegram_config_from_env,
    update_sent_alerts,
)
from btc_bias_data_collector import collect_etf_flows, collect_glassnode_metric
from data_ingestion import fetch_klines, fetch_latest_funding_rate, klines_to_market_df, symbols_from_string
from engine import DEFAULT_ALTS, EngineRunConfig, run_engine
from interval_profiles import profile_for_interval
from news_collector import collect_news_events
from portfolio.allocator import allocate_capital
from presets import apply_preset, preset_names, restore_preset
from quant_engine import apply_risk_management, build_opportunity_table, build_trade_setups, load_monitor_latest_family, select_portfolio
from risk.risk_engine import RiskLimits, evaluate_risk
from signal_engine import EngineMode
from signals.basis_trade import calculate_basis, get_basis_signal
from signals.btc_directional import build_btc_directional_signal
from signals.cross_exchange import find_arbitrage
from signals.funding_arbitrage import get_funding_signal


logger = logging.getLogger(__name__)

BINANCE_SPOT_TICKER_URL = "https://api.binance.com/api/v3/ticker/price"
BINANCE_FUTURES_TICKER_URL = "https://fapi.binance.com/fapi/v1/ticker/price"
COINBASE_SPOT_URL_TEMPLATE = "https://api.coinbase.com/v2/prices/{product_id}/spot"

def telegram_min_opportunity_score() -> float:
    return float(os.getenv("TELEGRAM_MIN_OPPORTUNITY_SCORE", "45"))


def telegram_max_alerts() -> int:
    return int(os.getenv("TELEGRAM_MAX_ALERTS", "3"))


def telegram_send_on_every_cycle() -> bool:
    return os.getenv("TELEGRAM_SEND_ON_EVERY_CYCLE", "true").strip().lower() in {"1", "true", "yes", "on"}


def news_collection_enabled() -> bool:
    return os.getenv("NEWS_COLLECTION_ENABLED", "true").strip().lower() in {"1", "true", "yes", "on"}


def news_poll_minutes() -> int:
    return int(os.getenv("NEWS_POLL_MINUTES", "30"))


def news_lookback_hours() -> int:
    return int(os.getenv("NEWS_LOOKBACK_HOURS", "36"))


def news_max_items_per_feed() -> int:
    return int(os.getenv("NEWS_MAX_ITEMS_PER_FEED", "25"))


def news_include_fear_greed() -> bool:
    return os.getenv("NEWS_INCLUDE_FEAR_GREED", "true").strip().lower() in {"1", "true", "yes", "on"}


def news_feeds() -> tuple[str, ...]:
    raw = os.getenv(
        "NEWS_RSS_FEEDS",
        "https://www.coindesk.com/arc/outboundfeeds/rss/,https://cointelegraph.com/rss",
    )
    return tuple(part.strip() for part in raw.split(",") if part.strip())


def btc_bias_data_collection_enabled() -> bool:
    return os.getenv("BTC_BIAS_DATA_COLLECTION_ENABLED", "true").strip().lower() in {"1", "true", "yes", "on"}


def btc_bias_data_poll_minutes() -> int:
    return int(os.getenv("BTC_BIAS_DATA_POLL_MINUTES", "180"))


def btc_etf_flows_path() -> Path:
    return Path(os.getenv("BTC_ETF_FLOWS_PATH", "btc_etf_flows.csv"))


def btc_mvrv_path() -> Path:
    return Path(os.getenv("BTC_MVRV_PATH", "btc_mvrv.csv"))


def btc_sopr_path() -> Path:
    return Path(os.getenv("BTC_SOPR_PATH", "btc_sopr.csv"))


def glassnode_mvrv_path() -> str:
    return os.getenv("GLASSNODE_MVRV_PATH", "/v1/metrics/market/mvrv")


def glassnode_sopr_path() -> str:
    return os.getenv("GLASSNODE_SOPR_PATH", "/v1/metrics/indicators/sopr_adjusted")


def parse_interval_list(raw_value: str) -> tuple[str, ...]:
    normalized = raw_value.replace(";", ",").replace(" ", ",")
    intervals = tuple(part.strip() for part in normalized.split(",") if part.strip())
    return intervals or ("1h",)


def interval_slug(interval: str) -> str:
    return interval.replace("/", "_")


def configure_logging() -> None:
    """Configure monitor logging once per process."""
    logging.basicConfig(
        level=os.getenv("LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )


def _session() -> requests.Session:
    session = requests.Session()
    session.trust_env = False
    return session


def fetch_last_price(url: str, params: dict[str, str]) -> float | None:
    """Fetch a single last-traded price from an HTTP JSON endpoint."""
    try:
        response: Response = _session().get(url, params=params, timeout=20)
        response.raise_for_status()
        payload = response.json()
        return float(payload["price"])
    except Exception as exc:
        logger.warning("price fetch failed", extra={"url": url, "params": params, "error": str(exc)})
        return None


def fetch_coinbase_spot_price(symbol: str) -> float | None:
    """Fetch a USD reference price from Coinbase when the pair exists there."""
    base_asset = symbol.replace("USDT", "")
    if not base_asset:
        return None

    product_id = f"{base_asset}-USD"
    try:
        response: Response = _session().get(COINBASE_SPOT_URL_TEMPLATE.format(product_id=product_id), timeout=20)
        response.raise_for_status()
        payload = response.json()
        return float(payload["data"]["amount"])
    except Exception as exc:
        logger.warning("coinbase spot fetch failed", extra={"symbol": symbol, "error": str(exc)})
        return None


def merge_allocations(
    signals_df: pd.DataFrame,
    allocations: list[dict[str, float | str]],
) -> pd.DataFrame:
    """Attach capital allocations to strategy signal rows."""
    if signals_df.empty:
        return signals_df.copy()
    if not allocations:
        merged = signals_df.copy()
        merged["allocated_capital"] = 0.0
        merged["strategy_weight"] = 0.0
        return merged

    allocations_df = pd.DataFrame(allocations)
    return signals_df.merge(
        allocations_df,
        on=["strategy", "symbol", "signal"],
        how="left",
    ).fillna({"allocated_capital": 0.0, "strategy_weight": 0.0})


def build_strategy_signals(
    ranked_universe: pd.DataFrame,
    interval: str,
    limit: int,
    funding_threshold: float,
    risk_limits: RiskLimits,
    current_drawdown: float,
    requested_leverage: float,
    total_capital: float,
) -> pd.DataFrame:
    """Build modular strategy signals from ranked market data."""
    rows: list[dict[str, object]] = []
    requested_position_size = total_capital * float(os.getenv("SIGNAL_POSITION_FRACTION", "0.1"))

    try:
        btc_klines = fetch_klines(symbol="BTCUSDT", interval=interval, limit=max(limit, 250))
        btc_market_df = klines_to_market_df("BTCUSDT", btc_klines)
        btc_signal = build_btc_directional_signal(btc_market_df)
        btc_funding_rate = fetch_latest_funding_rate("BTCUSDT")
        if btc_signal is not None:
            rows.append(
                {
                    "timestamp": btc_signal.timestamp,
                    "symbol": btc_signal.symbol,
                    "strategy": "btc_directional",
                    "signal": btc_signal.signal,
                    "timeframe": interval,
                    "funding_rate": btc_funding_rate,
                    "spot_price": btc_signal.current_price,
                    "reference_price": btc_signal.current_price,
                    "directional_score": btc_signal.score,
                    "ema_fast": btc_signal.ema_fast,
                    "ema_slow": btc_signal.ema_slow,
                    "ema_trend": btc_signal.ema_trend,
                    "ema_gap_pct": btc_signal.ema_gap_pct,
                    "rsi_14": btc_signal.rsi_14,
                    "macd_hist": btc_signal.macd_hist,
                    "breakout_20_pct": btc_signal.breakout_20_pct,
                    "realized_volatility": btc_signal.realized_volatility,
                    "market_regime": btc_signal.regime,
                }
            )
    except Exception as exc:
        logger.warning("btc directional signal failed", extra={"interval": interval, "error": str(exc)})

    for row in ranked_universe.itertuples(index=False):
        symbol = str(row.symbol)
        spot_price = float(row.current_price)
        funding_rate = float(getattr(row, "funding_rate", 0.0) or 0.0)
        perp_price = fetch_last_price(BINANCE_FUTURES_TICKER_URL, {"symbol": symbol})
        exchange_b_price = fetch_coinbase_spot_price(symbol)

        funding_signal = get_funding_signal(symbol, funding_rate, funding_threshold)
        rows.append(
            {
                "timestamp": row.timestamp,
                "symbol": symbol,
                "strategy": "funding_arbitrage",
                "signal": funding_signal,
                "timeframe": interval,
                "funding_rate": funding_rate,
                "spot_price": spot_price,
                "perp_price": perp_price,
                "reference_price": exchange_b_price,
            }
        )

        if perp_price is not None:
            basis_value = calculate_basis(spot_price, perp_price)
            basis_signal = get_basis_signal(spot_price, perp_price)
            rows.append(
                {
                    "timestamp": row.timestamp,
                    "symbol": symbol,
                    "strategy": "basis_trade",
                    "signal": basis_signal,
                    "timeframe": interval,
                    "funding_rate": funding_rate,
                    "spot_price": spot_price,
                    "perp_price": perp_price,
                    "basis": basis_value,
                    "reference_price": exchange_b_price,
                }
            )

        if exchange_b_price is not None:
            arbitrage = find_arbitrage(spot_price, exchange_b_price)
            rows.append(
                {
                    "timestamp": row.timestamp,
                    "symbol": symbol,
                    "strategy": "cross_exchange",
                    "signal": "ARBITRAGE" if arbitrage is not None else "NO_TRADE",
                    "timeframe": interval,
                    "funding_rate": funding_rate,
                    "spot_price": spot_price,
                    "perp_price": perp_price,
                    "reference_price": exchange_b_price,
                    "cross_exchange_diff_pct": arbitrage.difference_pct if arbitrage is not None else 0.0,
                    "buy_exchange": arbitrage.buy_exchange if arbitrage is not None else "",
                    "sell_exchange": arbitrage.sell_exchange if arbitrage is not None else "",
                }
            )

    signals_df = pd.DataFrame(rows)
    if signals_df.empty:
        return signals_df

    risk_rows: list[dict[str, object]] = []
    for signal_row in signals_df.itertuples(index=False):
        risk_result = evaluate_risk(
            limits=risk_limits,
            requested_position_size=requested_position_size,
            current_drawdown=current_drawdown,
            requested_leverage=requested_leverage,
        )
        risk_rows.append(
            {
                "strategy": signal_row.strategy,
                "symbol": signal_row.symbol,
                "signal": signal_row.signal,
                "risk_decision": risk_result.decision,
                "approved_position_size": risk_result.approved_position_size,
                "approved_leverage": risk_result.approved_leverage,
                "risk_reason": risk_result.reason,
            }
        )

    signals_df = signals_df.merge(pd.DataFrame(risk_rows), on=["strategy", "symbol", "signal"], how="left")
    allocations = allocate_capital(total_capital=total_capital, signals=signals_df.to_dict("records"))
    signals_df = merge_allocations(signals_df, allocations)
    return signals_df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Continuous monitor for the crypto relative value engine")
    parser.add_argument(
        "--mode",
        choices=[mode.value for mode in EngineMode],
        default=os.getenv("ENGINE_MODE", EngineMode.COPILOT.value),
    )
    parser.add_argument("--preset", choices=preset_names(), default=os.getenv("ENGINE_PRESET", "DEFAULT"))
    parser.add_argument("--symbols", help="Comma separated altcoin symbols, e.g. XRPUSDT,SOLUSDT")
    parser.add_argument("--interval", default=os.getenv("ENGINE_INTERVAL", "1h"))
    parser.add_argument("--intervals", default=os.getenv("MONITOR_INTERVALS"))
    parser.add_argument("--limit", type=int, default=int(os.getenv("ENGINE_LIMIT", "1000")))
    parser.add_argument("--poll-minutes", type=int, default=int(os.getenv("POLL_MINUTES", "5")))
    parser.add_argument("--min-confidence", type=float, default=float(os.getenv("ALERT_MIN_CONFIDENCE", "75.0")))
    parser.add_argument("--qualities", default=os.getenv("ALERT_SIGNAL_QUALITIES", "A1,A2"))
    parser.add_argument("--alert-cooldown-minutes", type=int, default=int(os.getenv("ALERT_COOLDOWN_MINUTES", "180")))
    parser.add_argument("--state-path", default=os.getenv("ALERT_STATE_PATH", "output/sent_alerts.txt"))
    parser.add_argument("--csv-path", default=os.getenv("ENGINE_CSV_PATH", "output/monitor_latest.csv"))
    parser.add_argument("--history-path", default=os.getenv("ALERT_HISTORY_PATH", "output/daily_alert_history.csv"))
    parser.add_argument("--signals-path", default=os.getenv("STRATEGY_SIGNALS_PATH", "output/strategy_signals_latest.csv"))
    parser.add_argument("--funding-threshold", type=float, default=float(os.getenv("FUNDING_THRESHOLD", "0.0005")))
    parser.add_argument("--total-capital", type=float, default=float(os.getenv("TOTAL_CAPITAL", "10000")))
    parser.add_argument("--current-drawdown", type=float, default=float(os.getenv("CURRENT_DRAWDOWN", "0.0")))
    parser.add_argument("--max-position-size", type=float, default=float(os.getenv("MAX_POSITION_SIZE", "1500")))
    parser.add_argument("--max-daily-drawdown", type=float, default=float(os.getenv("MAX_DAILY_DRAWDOWN", "0.05")))
    parser.add_argument("--max-leverage", type=float, default=float(os.getenv("MAX_LEVERAGE", "3.0")))
    parser.add_argument("--requested-leverage", type=float, default=float(os.getenv("REQUESTED_LEVERAGE", "1.0")))
    return parser.parse_args()


def output_path_for_interval(base_path: str, interval: str) -> str:
    base = Path(base_path)
    return str(base.with_name(f"{base.stem}_{interval_slug(interval)}{base.suffix}"))


def downside_alert_threshold_pct() -> float:
    return float(os.getenv("DOWNSIDE_ALERT_THRESHOLD_PCT", "-2.5"))


def downside_alert_3bar_threshold_pct() -> float:
    return float(os.getenv("DOWNSIDE_ALERT_3BAR_THRESHOLD_PCT", "-4.0"))


def build_downside_shock_alerts(market_df: pd.DataFrame, interval: str) -> pd.DataFrame:
    if interval != "15m" or market_df.empty:
        return pd.DataFrame()

    threshold_1bar = downside_alert_threshold_pct()
    threshold_3bar = downside_alert_3bar_threshold_pct()
    rows: list[dict[str, object]] = []

    for symbol, subset in market_df.groupby("symbol"):
        ordered = subset.sort_values("timestamp").copy()
        if ordered.empty:
            continue
        latest = ordered.iloc[-1]
        open_price = float(latest.get("open", 0.0) or 0.0)
        low_price = float(latest.get("low", 0.0) or 0.0)
        close_price = float(latest.get("close", 0.0) or 0.0)
        volume = float(latest.get("quote_volume", 0.0) or 0.0)
        if open_price <= 0 or close_price <= 0:
            continue

        candle_return_pct = (close_price / open_price - 1.0) * 100.0
        low_from_open_pct = (low_price / open_price - 1.0) * 100.0 if low_price > 0 else candle_return_pct
        return_3bar_pct = 0.0
        if len(ordered) >= 4:
            ref_close = float(ordered.iloc[-4].get("close", 0.0) or 0.0)
            if ref_close > 0:
                return_3bar_pct = (close_price / ref_close - 1.0) * 100.0

        if candle_return_pct <= threshold_1bar or return_3bar_pct <= threshold_3bar:
            rows.append(
                {
                    "symbol": symbol,
                    "candle_return_pct": round(candle_return_pct, 2),
                    "return_3bar_pct": round(return_3bar_pct, 2),
                    "low_from_open_pct": round(low_from_open_pct, 2),
                    "quote_volume": volume,
                }
            )

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(["candle_return_pct", "return_3bar_pct"])


def filter_new_downside_shocks(
    shocks_df: pd.DataFrame,
    sent_alerts: dict[str, pd.Timestamp],
    cooldown_minutes: int,
) -> pd.DataFrame:
    if shocks_df.empty:
        return shocks_df.copy()

    now_utc = pd.Timestamp.utcnow()
    if now_utc.tzinfo is None:
        now_utc = now_utc.tz_localize("UTC")
    cooldown = pd.Timedelta(minutes=max(cooldown_minutes, 0))
    fresh_rows: list[pd.Series] = []

    for _, row in shocks_df.iterrows():
        key = f"shock|{row['symbol']}"
        last_sent = sent_alerts.get(key)
        if last_sent is None or (now_utc - last_sent) >= cooldown:
            fresh_rows.append(row)

    if not fresh_rows:
        return shocks_df.iloc[0:0].copy()
    return pd.DataFrame(fresh_rows).reset_index(drop=True)


def update_sent_downside_shocks(
    sent_alerts: dict[str, pd.Timestamp],
    shocks_df: pd.DataFrame,
) -> dict[str, pd.Timestamp]:
    if shocks_df.empty:
        return sent_alerts

    sent_at = pd.Timestamp.utcnow()
    if sent_at.tzinfo is None:
        sent_at = sent_at.tz_localize("UTC")

    for _, row in shocks_df.iterrows():
        sent_alerts[f"shock|{row['symbol']}"] = sent_at
    return sent_alerts


def run_interval_iteration(
    *,
    now_label: str,
    args: argparse.Namespace,
    interval: str,
    email_config: EmailConfig | None,
    telegram_config: TelegramConfig | None,
    sent_alerts: dict[str, pd.Timestamp],
    risk_limits: RiskLimits,
    signal_qualities: tuple[str, ...],
) -> dict[str, pd.Timestamp]:
    profile = profile_for_interval(interval)
    limit = args.limit if args.limit != 1000 or interval == args.interval else profile.limit
    csv_path = output_path_for_interval(args.csv_path, interval)
    state_path = output_path_for_interval(args.state_path, interval)
    history_path_value = output_path_for_interval(args.history_path, interval)
    signals_path_value = output_path_for_interval(args.signals_path, interval)

    previous_env = apply_preset(args.preset)
    try:
        result = run_engine(
            EngineRunConfig(
                mode=EngineMode(args.mode),
                symbols=symbols_from_string(args.symbols, DEFAULT_ALTS),
                interval=interval,
                limit=limit,
                feature_config=profile.feature_config,
                csv_path=csv_path,
                live_mode=False,
                paper_trading=True,
                dry_run=True,
                test_order_mode=True,
            )
        )
    finally:
        restore_preset(previous_env)
    ranked_universe = result.ranked_universe.assign(timeframe=interval)
    alerts = select_alert_candidates(
        ranked_universe,
        min_quality=signal_qualities,
        min_confidence=args.min_confidence,
    )
    history_path = append_daily_alert_snapshot(history_path_value, ranked_universe)
    new_alerts = filter_new_alerts(
        alerts_df=alerts,
        sent_alerts=sent_alerts,
        cooldown_minutes=args.alert_cooldown_minutes,
    )
    best_today = daily_best_alerts(
        ranked_universe.assign(snapshot_date=pd.Timestamp.utcnow().date().isoformat())
    )
    strategy_signals = build_strategy_signals(
        ranked_universe=result.ranked_universe,
        interval=interval,
        limit=limit,
        funding_threshold=args.funding_threshold,
        risk_limits=risk_limits,
        current_drawdown=args.current_drawdown,
        requested_leverage=args.requested_leverage,
        total_capital=args.total_capital,
    )
    signals_path = Path(signals_path_value)
    signals_path.parent.mkdir(parents=True, exist_ok=True)
    strategy_signals.to_csv(signals_path, index=False)

    logger.info(
        "[%s][%s] corridas=%s propuestas=%s alertas_nuevas=%s mejores_hoy=%s historial=%s strategy_signals=%s limit=%s",
        now_label,
        interval,
        len(result.ranked_universe),
        len(result.proposals),
        len(new_alerts),
        len(best_today),
        history_path,
        len(strategy_signals),
        limit,
    )

    telegram_candidates = alerts if not alerts.empty else ranked_universe.iloc[0:0].copy()
    if telegram_candidates.empty and telegram_send_on_every_cycle():
        telegram_candidates = (
            ranked_universe.sort_values(
                ["market_opportunity_score", "confidence_score"],
                ascending=[False, False],
                na_position="last",
            )
            .head(telegram_max_alerts())
            .copy()
        )

    if telegram_config is not None and not telegram_candidates.empty:
        telegram_current = telegram_candidates
        if not telegram_send_on_every_cycle():
            telegram_current = filter_new_alerts(
                alerts_df=telegram_candidates,
                sent_alerts=sent_alerts,
                cooldown_minutes=args.alert_cooldown_minutes,
            )
        opportunity_source = load_monitor_latest_family(args.csv_path)
        opportunity_table = build_opportunity_table(opportunity_source)
        opportunity_trades = build_trade_setups(opportunity_table)
        managed_trades = apply_risk_management(opportunity_trades)
        top_opportunities = select_portfolio(managed_trades)
        top_opportunities = top_opportunities.loc[
            top_opportunities["opportunity_score"] >= telegram_min_opportunity_score()
        ].head(telegram_max_alerts())
        market_summary = managed_trades.sort_values(
            ["opportunity_score", "alignment_score", "base_score"],
            ascending=[False, False, False],
            na_position="last",
        ).head(5)
        strategy_risk_summary = strategy_signals.sort_values(
            ["risk_decision", "allocated_capital", "symbol"],
            ascending=[True, False, True],
            na_position="last",
        ).head(5)
        downside_shocks = filter_new_downside_shocks(
            build_downside_shock_alerts(result.market_df, interval).head(5),
            sent_alerts=sent_alerts,
            cooldown_minutes=args.alert_cooldown_minutes,
        )

        if not top_opportunities.empty and not telegram_current.empty:
            message = format_top_opportunities_telegram(f"{result.mode.value} {interval}", top_opportunities)
            send_telegram_alert(telegram_config, message)
            if not market_summary.empty:
                summary_message = format_market_summary_telegram(f"{result.mode.value} {interval}", market_summary)
                send_telegram_alert(telegram_config, summary_message)
            if not strategy_risk_summary.empty:
                risk_message = format_strategy_risk_telegram(f"{result.mode.value} {interval}", strategy_risk_summary)
                send_telegram_alert(telegram_config, risk_message)
            if not downside_shocks.empty:
                shock_message = format_downside_shocks_telegram(f"{result.mode.value} {interval}", downside_shocks)
                send_telegram_alert(telegram_config, shock_message)
                sent_alerts = update_sent_downside_shocks(sent_alerts, downside_shocks)
            logger.info(
                "[%s][%s] telegram top opportunities enviado a chat %s",
                now_label,
                interval,
                telegram_config.chat_id,
            )
            sent_alerts = update_sent_alerts(sent_alerts, telegram_current)
            save_sent_alerts(state_path, sent_alerts)
        elif not telegram_current.empty:
            fallback_alerts = telegram_current.sort_values(
                ["market_opportunity_score", "confidence_score"],
                ascending=[False, False],
                na_position="last",
            ).head(telegram_max_alerts())
            message = format_alert_telegram(f"{result.mode.value} {interval}", fallback_alerts)
            send_telegram_alert(telegram_config, message)
            if not market_summary.empty:
                summary_message = format_market_summary_telegram(f"{result.mode.value} {interval}", market_summary)
                send_telegram_alert(telegram_config, summary_message)
            if not strategy_risk_summary.empty:
                risk_message = format_strategy_risk_telegram(f"{result.mode.value} {interval}", strategy_risk_summary)
                send_telegram_alert(telegram_config, risk_message)
            if not downside_shocks.empty:
                shock_message = format_downside_shocks_telegram(f"{result.mode.value} {interval}", downside_shocks)
                send_telegram_alert(telegram_config, shock_message)
                sent_alerts = update_sent_downside_shocks(sent_alerts, downside_shocks)
            logger.info(
                "[%s][%s] telegram fallback enviado a chat %s",
                now_label,
                interval,
                telegram_config.chat_id,
            )
            sent_alerts = update_sent_alerts(sent_alerts, telegram_current)
            save_sent_alerts(state_path, sent_alerts)
    elif email_config is not None and not new_alerts.empty:
        subject, body = format_alert_email(f"{result.mode.value} {interval}", new_alerts)
        send_email_alert(email_config, subject, body)
        sent_alerts = update_sent_alerts(sent_alerts, new_alerts)
        save_sent_alerts(state_path, sent_alerts)
        logger.info("[%s][%s] email enviado a %s", now_label, interval, email_config.to_email)

    return sent_alerts


def maybe_collect_news(args: argparse.Namespace, next_news_run_at: float) -> float:
    if not news_collection_enabled():
        return next_news_run_at

    now = time.time()
    if now < next_news_run_at:
        return next_news_run_at

    try:
        events = collect_news_events(
            output_path=Path(os.getenv("NEWS_EVENTS_PATH", "news_events.csv")),
            feeds=news_feeds(),
            lookback_hours=news_lookback_hours(),
            max_items_per_feed=news_max_items_per_feed(),
            symbols=symbols_from_string(args.symbols, DEFAULT_ALTS),
            include_fear_greed=news_include_fear_greed(),
        )
        logger.info("news collector updated events=%s", len(events))
    except Exception as exc:
        logger.warning("news collector failed", extra={"error": str(exc)})

    return time.time() + max(news_poll_minutes(), 5) * 60


def maybe_collect_btc_bias_data(next_run_at: float) -> float:
    if not btc_bias_data_collection_enabled():
        return next_run_at

    now = time.time()
    if now < next_run_at:
        return next_run_at

    etf_rows = 0
    mvrv_rows = 0
    sopr_rows = 0
    try:
        etf_rows = collect_etf_flows(btc_etf_flows_path())
    except Exception as exc:
        logger.warning("btc etf collector failed", extra={"error": str(exc)})

    try:
        if os.getenv("GLASSNODE_API_KEY", "").strip():
            mvrv_rows = collect_glassnode_metric(glassnode_mvrv_path(), btc_mvrv_path())
            sopr_rows = collect_glassnode_metric(glassnode_sopr_path(), btc_sopr_path())
    except Exception as exc:
        logger.warning("btc onchain collector failed", extra={"error": str(exc)})

    logger.info(
        "btc bias data collector updated etf_rows=%s mvrv_rows=%s sopr_rows=%s",
        etf_rows,
        mvrv_rows,
        sopr_rows,
    )
    return time.time() + max(btc_bias_data_poll_minutes(), 60) * 60


def main() -> None:
    configure_logging()
    args = parse_args()
    email_config = email_config_from_env()
    telegram_config = telegram_config_from_env()
    signal_qualities = parse_quality_list(args.qualities)
    intervals = parse_interval_list(args.intervals or args.interval)
    sent_alerts_by_interval = {
        interval: load_sent_alerts(output_path_for_interval(args.state_path, interval))
        for interval in intervals
    }
    risk_limits = RiskLimits(
        max_position_size=args.max_position_size,
        max_daily_drawdown=args.max_daily_drawdown,
        max_leverage=args.max_leverage,
    )

    if telegram_config is None and email_config is None:
        logger.info("Alertas desactivadas: Telegram y SMTP no estan configurados.")
    elif telegram_config is not None:
        logger.info("Telegram alerts enabled for chat %s", telegram_config.chat_id)
    else:
        logger.info("Telegram no configurado; se usa email.")

    next_run_by_interval = {interval: 0.0 for interval in intervals}
    next_news_run_at = 0.0
    next_btc_bias_run_at = 0.0

    while True:
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        current_time = time.time()
        ran_cycle = False
        next_news_run_at = maybe_collect_news(args, next_news_run_at)
        next_btc_bias_run_at = maybe_collect_btc_bias_data(next_btc_bias_run_at)
        for interval in intervals:
            profile = profile_for_interval(interval)
            if current_time < next_run_by_interval[interval]:
                continue
            ran_cycle = True
            try:
                sent_alerts_by_interval[interval] = run_interval_iteration(
                    now_label=now,
                    args=args,
                    interval=interval,
                    email_config=email_config,
                    telegram_config=telegram_config,
                    sent_alerts=sent_alerts_by_interval[interval],
                    risk_limits=risk_limits,
                    signal_qualities=signal_qualities,
                )
            except Exception as exc:
                logger.exception("[%s][%s] error en monitor: %s", now, interval, exc)
            next_run_by_interval[interval] = time.time() + max(profile.poll_minutes, 1) * 60

        if not ran_cycle:
            time.sleep(30)


if __name__ == "__main__":
    main()
