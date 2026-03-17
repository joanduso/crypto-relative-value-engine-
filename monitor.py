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
    load_sent_alerts,
    parse_quality_list,
    save_sent_alerts,
    select_alert_candidates,
    send_email_alert,
    update_sent_alerts,
)
from data_ingestion import symbols_from_string
from engine import DEFAULT_ALTS, EngineRunConfig, run_engine
from interval_profiles import profile_for_interval
from portfolio.allocator import allocate_capital
from risk.risk_engine import RiskLimits, evaluate_risk
from signal_engine import EngineMode
from signals.basis_trade import calculate_basis, get_basis_signal
from signals.cross_exchange import find_arbitrage
from signals.funding_arbitrage import get_funding_signal


logger = logging.getLogger(__name__)

BINANCE_SPOT_TICKER_URL = "https://api.binance.com/api/v3/ticker/price"
BINANCE_FUTURES_TICKER_URL = "https://fapi.binance.com/fapi/v1/ticker/price"
COINBASE_SPOT_URL_TEMPLATE = "https://api.coinbase.com/v2/prices/{product_id}/spot"


def parse_interval_list(raw_value: str) -> tuple[str, ...]:
    intervals = tuple(part.strip() for part in raw_value.split(",") if part.strip())
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
    funding_threshold: float,
    risk_limits: RiskLimits,
    current_drawdown: float,
    requested_leverage: float,
    total_capital: float,
) -> pd.DataFrame:
    """Build modular strategy signals from ranked market data."""
    rows: list[dict[str, object]] = []
    requested_position_size = total_capital * float(os.getenv("SIGNAL_POSITION_FRACTION", "0.1"))

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


def run_interval_iteration(
    *,
    now_label: str,
    args: argparse.Namespace,
    interval: str,
    email_config: EmailConfig | None,
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

    if email_config is not None and not new_alerts.empty:
        subject, body = format_alert_email(f"{result.mode.value} {interval}", new_alerts)
        send_email_alert(email_config, subject, body)
        sent_alerts = update_sent_alerts(sent_alerts, new_alerts)
        save_sent_alerts(state_path, sent_alerts)
        logger.info("[%s][%s] email enviado a %s", now_label, interval, email_config.to_email)

    return sent_alerts


def main() -> None:
    configure_logging()
    args = parse_args()
    email_config = email_config_from_env()
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

    if email_config is None:
        logger.info("Email alerts disabled: SMTP env vars are not configured.")

    next_run_by_interval = {interval: 0.0 for interval in intervals}

    while True:
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        current_time = time.time()
        ran_cycle = False
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
