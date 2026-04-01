from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from backtest import compare_bias_overlay_backtests
from calibration import calibrate_from_history
from dashboard import export_proposals_csv, render_terminal_dashboard
from data_ingestion import DataConfig, fetch_market_data
from execution_engine import BinanceExecutionEngine, execution_config_from_env
from feature_engine import FeatureConfig, build_feature_frame
from news_engine import apply_news_overlay
from portfolio_manager import PortfolioManager
from regime_engine import apply_regime_overlay
from risk_engine import RiskLimits, RiskState, attach_trade_plan
from signal_engine import EngineMode, build_opportunity_table_from_ranked, build_ranked_universe


DEFAULT_ALTS = ("XRPUSDT", "SOLUSDT", "ADAUSDT", "DOGEUSDT", "BNBUSDT", "LTCUSDT", "LINKUSDT", "AVAXUSDT")


@dataclass(frozen=True)
class EngineRunConfig:
    mode: EngineMode
    symbols: tuple[str, ...] = DEFAULT_ALTS
    limit: int = 1000
    interval: str = "1h"
    feature_config: FeatureConfig | None = None
    csv_path: str = "output/proposed_trades.csv"
    live_mode: bool = False
    paper_trading: bool = False
    dry_run: bool = False
    test_order_mode: bool = False


@dataclass
class EngineRunResult:
    mode: EngineMode
    live_mode: bool
    market_df: pd.DataFrame
    features_df: pd.DataFrame
    ranked_universe: pd.DataFrame
    daily_best: pd.DataFrame
    proposals: pd.DataFrame
    backtest_stats: pd.DataFrame
    dashboard_text: str
    csv_path: Path


def mode_risk_limits(mode: EngineMode) -> RiskLimits:
    if mode is EngineMode.AUTO_SAFE:
        base = RiskLimits(
            max_concurrent_positions=2,
            max_daily_loss_pct=0.02,
            max_weekly_drawdown_pct=0.05,
            max_risk_per_trade_pct=0.0075,
            max_holding_hours=36,
            stop_loss_pct=0.025,
            take_profit_pct=0.05,
        )
    elif mode is EngineMode.COPILOT_RELAXED:
        base = RiskLimits(
            max_concurrent_positions=2,
            max_daily_loss_pct=0.035,
            max_weekly_drawdown_pct=0.08,
            max_risk_per_trade_pct=0.01,
            max_holding_hours=48,
            stop_loss_pct=0.0325,
            take_profit_pct=0.065,
        )
    else:
        base = RiskLimits(
            max_concurrent_positions=1,
            max_daily_loss_pct=0.03,
            max_weekly_drawdown_pct=0.07,
            max_risk_per_trade_pct=0.01,
            max_holding_hours=48,
            stop_loss_pct=0.03,
            take_profit_pct=0.06,
        )

    prefix = f"RISK_{mode.value}_"
    return RiskLimits(
        max_concurrent_positions=int(os.getenv(f"{prefix}MAX_CONCURRENT_POSITIONS", os.getenv("RISK_MAX_CONCURRENT_POSITIONS", str(base.max_concurrent_positions)))),
        max_daily_loss_pct=float(os.getenv(f"{prefix}MAX_DAILY_LOSS_PCT", os.getenv("RISK_MAX_DAILY_LOSS_PCT", str(base.max_daily_loss_pct)))),
        max_weekly_drawdown_pct=float(os.getenv(f"{prefix}MAX_WEEKLY_DRAWDOWN_PCT", os.getenv("RISK_MAX_WEEKLY_DRAWDOWN_PCT", str(base.max_weekly_drawdown_pct)))),
        max_risk_per_trade_pct=float(os.getenv(f"{prefix}MAX_RISK_PER_TRADE_PCT", os.getenv("RISK_MAX_RISK_PER_TRADE_PCT", str(base.max_risk_per_trade_pct)))),
        max_holding_hours=int(os.getenv(f"{prefix}MAX_HOLDING_HOURS", os.getenv("RISK_MAX_HOLDING_HOURS", str(base.max_holding_hours)))),
        stop_loss_pct=float(os.getenv(f"{prefix}STOP_LOSS_PCT", os.getenv("RISK_STOP_LOSS_PCT", str(base.stop_loss_pct)))),
        take_profit_pct=float(os.getenv(f"{prefix}TAKE_PROFIT_PCT", os.getenv("RISK_TAKE_PROFIT_PCT", str(base.take_profit_pct)))),
    )


def run_engine(config: EngineRunConfig) -> EngineRunResult:
    active_risk_limits = mode_risk_limits(config.mode)
    market_df = fetch_market_data(
        DataConfig(
            symbols=config.symbols,
            interval=config.interval,
            limit=config.limit,
        )
    )
    features_df = build_feature_frame(market_df, config.symbols, config.feature_config or FeatureConfig())
    ranked_universe = build_ranked_universe(features_df, config.mode)
    ranked_universe, _ = apply_news_overlay(ranked_universe, market_df)
    ranked_universe, _ = apply_regime_overlay(ranked_universe, market_df, config.mode, interval=config.interval)
    ranked_universe, _ = calibrate_from_history(
        features_df,
        ranked_universe,
        mode=config.mode,
        interval=config.interval,
        risk_limits=active_risk_limits,
    )
    ranked_universe = attach_trade_plan(
        ranked_universe,
        limits=active_risk_limits,
        state=RiskState(),
    )
    daily_best = (
        ranked_universe.sort_values(["symbol", "confidence_score", "edge_after_fees_pct"], ascending=[True, False, False])
        .groupby("symbol", as_index=False)
        .head(1)
        .reset_index(drop=True)
        if not ranked_universe.empty
        else pd.DataFrame()
    )
    opportunities = build_opportunity_table_from_ranked(ranked_universe, config.mode)

    portfolio = PortfolioManager(limits=active_risk_limits)
    proposals = portfolio.prepare_orders(opportunities)
    proposals = proposals.loc[proposals["risk_checks_passed"]].copy() if not proposals.empty else proposals

    backtest_stats = compare_bias_overlay_backtests(market_df, features_df, interval=config.interval)
    csv_path = export_proposals_csv(proposals, config.csv_path)

    if config.mode is EngineMode.AUTO_SAFE and not proposals.empty:
        execution = BinanceExecutionEngine(
            execution_config_from_env(
                live_mode=config.live_mode,
                dry_run=(not config.live_mode) or config.dry_run,
                paper_trading=config.paper_trading,
                test_order_mode=config.test_order_mode or not config.live_mode,
            )
        )
        for row in proposals.itertuples(index=False):
            side = "BUY" if row.suggested_direction == "LONG" else "SELL"
            execution.create_market_order(
                symbol=row.symbol,
                side=side,
                quantity=row.suggested_position_size,
            )
            portfolio.register_position(proposals.loc[proposals["symbol"] == row.symbol].iloc[0], config.mode.value)

    dashboard_text = render_terminal_dashboard(
        mode_name=config.mode.value,
        live_mode=config.live_mode,
        proposals=proposals,
        backtest_stats=backtest_stats,
    )
    return EngineRunResult(
        mode=config.mode,
        live_mode=config.live_mode,
        market_df=market_df,
        features_df=features_df,
        ranked_universe=ranked_universe,
        daily_best=daily_best,
        proposals=proposals,
        backtest_stats=backtest_stats,
        dashboard_text=dashboard_text,
        csv_path=csv_path,
    )
