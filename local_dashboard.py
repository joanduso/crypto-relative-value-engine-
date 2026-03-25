from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from data_ingestion import symbols_from_string
from engine import DEFAULT_ALTS, EngineRunConfig, run_engine
from news_engine import load_news_events
from presets import apply_preset, preset_description, preset_names, restore_preset
from signal_engine import EngineMode


st.set_page_config(page_title="Crypto Relative Value Engine", layout="wide")
STRATEGY_SIGNALS_PATH = Path("output/strategy_signals_latest.csv")
NEWS_EVENTS_PATH = Path("news_events.csv")


def _interval_recommendation(interval: str) -> str:
    if interval == "15m":
        return "Recomendacion UX: 15m para entradas y 200 a 300 velas para intradia."
    if interval == "1h":
        return "Recomendacion UX: 1h para setup y 200 a 400 velas para swing corto."
    if interval == "4h":
        return "Recomendacion UX: 4h para tendencia y 150 a 300 velas para contexto."
    return "Recomendacion UX: intenta mostrar entre 150 y 300 velas para mantener contexto sin exceso de ruido."


def _style_direction(value: object) -> str:
    if value == "LONG":
        return "color: #0f9d58; font-weight: 700;"
    if value == "SHORT":
        return "color: #d93025; font-weight: 700;"
    return ""


def _style_boolean(value: object) -> str:
    if value is True:
        return "background-color: #e6f4ea;"
    if value is False:
        return "background-color: #fce8e6;"
    return ""


def _style_score(value: object) -> str:
    if not isinstance(value, (int, float)):
        return ""
    if value >= 75:
        return "background-color: #e6f4ea;"
    if value >= 55:
        return "background-color: #fff8e1;"
    return "background-color: #fce8e6;"


def _style_quality(value: object) -> str:
    if value == "A1":
        return "background-color: #c8e6c9; font-weight: 700;"
    if value == "A2":
        return "background-color: #dcedc8; font-weight: 700;"
    if value == "B1":
        return "background-color: #fff9c4;"
    if value == "B2":
        return "background-color: #fff3cd;"
    if value == "C1":
        return "background-color: #ffe0b2;"
    if value == "C2":
        return "background-color: #ffcdd2;"
    if value == "D1":
        return "background-color: #f8d7da;"
    if value == "D2":
        return "background-color: #f1b0b7; color: #5f2120;"
    return ""


def _display_table(df: pd.DataFrame, columns: list[str]) -> None:
    if df.empty:
        st.info("No hay datos para mostrar.")
        return
    available = [column for column in columns if column in df.columns]
    styled = (
        df[available]
        .style.map(_style_direction, subset=[col for col in ["suggested_direction"] if col in available])
        .map(_style_boolean, subset=[col for col in ["passes_filters", "risk_checks_passed"] if col in available])
        .map(_style_score, subset=[col for col in ["confidence_score"] if col in available])
        .map(_style_quality, subset=[col for col in ["signal_quality"] if col in available])
    )
    st.dataframe(styled, width="stretch")


def _load_strategy_signals() -> pd.DataFrame:
    if not STRATEGY_SIGNALS_PATH.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(STRATEGY_SIGNALS_PATH)
    except Exception:
        return pd.DataFrame()


def _load_news_events() -> pd.DataFrame:
    events = load_news_events(NEWS_EVENTS_PATH)
    if events.empty:
        return events
    return events.sort_values("timestamp", ascending=False).reset_index(drop=True)


def _news_for_symbols(events_df: pd.DataFrame, symbols: tuple[str, ...], limit: int = 6) -> pd.DataFrame:
    if events_df.empty:
        return events_df.copy()
    normalized = {symbol.upper() for symbol in symbols}
    filtered = events_df.loc[events_df["symbol"].astype(str).str.upper().isin(normalized)].copy()
    return filtered.head(limit).reset_index(drop=True)


def _benchmark_trend(market_df: pd.DataFrame, symbol: str) -> dict[str, object]:
    subset = market_df.loc[market_df["symbol"] == symbol, ["timestamp", "close"]].copy()
    if subset.empty or len(subset) < 200:
        return {
            "label": symbol.replace("USDT", ""),
            "trend": "SIN DATA",
            "arrow": "->",
            "delta_pct": 0.0,
        }

    subset = subset.sort_values("timestamp")
    close = pd.to_numeric(subset["close"], errors="coerce").dropna()
    if len(close) < 200:
        return {
            "label": symbol.replace("USDT", ""),
            "trend": "SIN DATA",
            "arrow": "->",
            "delta_pct": 0.0,
        }

    ema50 = close.ewm(span=50, adjust=False).mean()
    ema200 = close.ewm(span=200, adjust=False).mean()
    last_close = float(close.iloc[-1])
    last_ema50 = float(ema50.iloc[-1])
    last_ema200 = float(ema200.iloc[-1])
    price_vs_ema200_pct = (last_close / max(last_ema200, 1e-9) - 1.0) * 100.0

    bullish = last_close > last_ema200 and last_ema50 > last_ema200
    bearish = last_close < last_ema200 and last_ema50 < last_ema200
    if bullish:
        trend = "ALCISTA"
        arrow = "↑"
    elif bearish:
        trend = "BAJISTA"
        arrow = "↓"
    else:
        trend = "NEUTRAL"
        arrow = "→"

    return {
        "label": symbol.replace("USDT", ""),
        "trend": trend,
        "arrow": arrow,
        "delta_pct": round(price_vs_ema200_pct, 2),
    }


st.title("Crypto Relative Value Engine")
st.caption("Dashboard local con ejecucion manual, scoring y tabla ampliada por altcoin.")

strategy_signals_df = _load_strategy_signals()
news_events_df = _load_news_events()

with st.sidebar:
    st.header("Controles")
    mode_values = [mode.value for mode in EngineMode]
    mode = st.selectbox("Modo", mode_values, index=0)
    preset = st.selectbox("Preset", preset_names(), index=0)
    st.caption(preset_description(preset))
    symbols_raw = st.text_input("Symbols", ",".join(DEFAULT_ALTS))
    interval = st.selectbox("Intervalo", ["1h", "4h", "15m"], index=0)
    limit = st.slider("Velas", min_value=300, max_value=1500, value=1000, step=100)
    st.caption(_interval_recommendation(interval))
    st.caption("Base sugerida: 4h para tendencia, 1h para setup y 15m para entrada.")
    live_mode = st.checkbox("Live mode", value=False)
    paper_trading = st.checkbox("Paper trading", value=True)
    dry_run = st.checkbox("Dry run", value=True)
    test_order_mode = st.checkbox("Test order mode", value=True)
    run_clicked = st.button("Run Engine", type="primary", width="stretch")
    st.divider()
    st.subheader("Filtros de tabla")
    min_confidence = st.slider("Confidence min", min_value=0, max_value=100, value=0, step=5)
    min_abs_deviation = st.slider("Desvio abs % min", min_value=0.0, max_value=20.0, value=0.0, step=0.5)
    min_abs_zscore = st.slider("Abs z-score min", min_value=0.0, max_value=5.0, value=0.0, step=0.1)
    only_buy = st.checkbox("Solo compras", value=False)
    only_passed = st.checkbox("Solo las que pasan filtros", value=False)

if "last_result" not in st.session_state:
    st.session_state["last_result"] = None

if run_clicked:
    with st.spinner("Corriendo engine..."):
        previous_env = apply_preset(preset)
        try:
            result = run_engine(
                EngineRunConfig(
                    mode=EngineMode(mode),
                    symbols=symbols_from_string(symbols_raw, DEFAULT_ALTS),
                    interval=interval,
                    limit=limit,
                    csv_path=f"output/{mode.lower()}_{preset.lower()}_trades.csv",
                    live_mode=live_mode,
                    paper_trading=paper_trading,
                    dry_run=dry_run,
                    test_order_mode=test_order_mode,
                )
            )
        finally:
            restore_preset(previous_env)
        st.session_state["last_result"] = result

result = st.session_state["last_result"]

if result is None:
    st.info("Pulsa `Run Engine` para cargar oportunidades, score y backtest.")
else:
    metric_a, metric_b, metric_c, metric_d = st.columns(4)
    metric_a.metric("Modo", result.mode.value)
    metric_b.metric("Live mode", "ON" if result.live_mode else "OFF")
    metric_c.metric("Propuestas", len(result.proposals))
    metric_d.metric("CSV", str(Path(result.csv_path)))
    btc_trend = _benchmark_trend(result.market_df, "BTCUSDT")
    eth_trend = _benchmark_trend(result.market_df, "ETHUSDT")
    trend_a, trend_b = st.columns(2)
    trend_a.metric(
        f"{btc_trend['label']} tendencia",
        f"{btc_trend['trend']} {btc_trend['arrow']}",
        f"{btc_trend['delta_pct']}% vs EMA200",
    )
    trend_b.metric(
        f"{eth_trend['label']} tendencia",
        f"{eth_trend['trend']} {eth_trend['arrow']}",
        f"{eth_trend['delta_pct']}% vs EMA200",
    )
    st.subheader("Noticias BTC y ETH")
    btc_eth_news = _news_for_symbols(news_events_df, ("BTCUSDT", "ETHUSDT"), limit=6)
    if btc_eth_news.empty:
        st.caption("Sin noticias recientes de BTC o ETH en `news_events.csv`.")
    else:
        for item in btc_eth_news.itertuples(index=False):
            timestamp = pd.to_datetime(item.timestamp, errors="coerce", utc=True)
            ts_text = timestamp.strftime("%Y-%m-%d %H:%M UTC") if pd.notna(timestamp) else "-"
            headline = str(getattr(item, "headline", "") or "").strip()
            source = str(getattr(item, "source_tier", "") or "").strip()
            sentiment = str(getattr(item, "sentiment", "") or "").strip()
            url = str(getattr(item, "url", "") or "").strip()
            symbol = str(getattr(item, "symbol", "") or "").strip()
            if url:
                st.markdown(f"- `{ts_text}` [{headline}]({url})  \n  `{symbol} | {sentiment} | {source}`")
            else:
                st.markdown(f"- `{ts_text}` {headline}  \n  `{symbol} | {sentiment} | {source}`")
    st.caption(f"Preset activo: {preset}")
    st.info(
        "Calificacion relativa al mercado actual: A1/A2 son las mejores oportunidades de la corrida; "
        "B1/B2 son operables; C1/C2 son medias; D1/D2 son flojas. "
        "La nota sale del ranking relativo entre score, edge, z-score, estabilidad, liquidez y volatilidad."
    )
    st.info(
        "Velas e intervalo: un marco corto reacciona mas rapido pero mete mas ruido. "
        "Como base, usa 150 a 300 velas visibles para contexto. "
        "Referencia practica: 4h para tendencia, 1h para setup y 15m para entrada."
    )

    ranked_view = result.ranked_universe.copy()
    if not ranked_view.empty:
        ranked_view = ranked_view.loc[ranked_view["confidence_score"] >= min_confidence]
        ranked_view = ranked_view.loc[ranked_view["deviation_pct"].abs() >= min_abs_deviation]
        ranked_view = ranked_view.loc[ranked_view["z_score"].abs() >= min_abs_zscore]
        if only_buy:
            ranked_view = ranked_view.loc[ranked_view["suggested_direction"] == "LONG"]
        if only_passed:
            ranked_view = ranked_view.loc[ranked_view["passes_filters"]]

    if not ranked_view.empty and {"news_bias", "news_comment", "news_impact_score"}.issubset(ranked_view.columns):
        st.subheader("Impacto noticias")
        news_view = ranked_view.loc[
            pd.to_numeric(ranked_view["news_impact_score"], errors="coerce").abs() > 0
        , ["symbol", "news_bias", "news_impact_score", "news_event_count", "news_comment"]].copy()
        if news_view.empty:
            st.caption("Sin impacto de noticias relevante en esta corrida.")
        else:
            st.dataframe(news_view.head(8), width="stretch")

    st.subheader("Top propuestas")
    _display_table(
        result.proposals,
        [
            "timestamp",
            "symbol",
            "current_price",
            "expected_fair_value",
            "deviation_pct",
            "z_score",
            "spread_stability_score",
            "suggested_direction",
            "signal_quality",
            "suggested_entry",
            "suggested_stop_loss",
            "suggested_take_profit",
            "suggested_position_size",
            "confidence_score",
            "pre_news_market_score",
            "news_impact_score",
            "news_bias",
            "news_event_count",
            "news_comment",
            "effective_win_rate_pct",
            "win_rate_source",
            "risk_reward_ratio",
            "implied_win_rate_pct",
            "expected_value_pct",
        ],
    )

    st.subheader("Ranking completo del universo")
    _display_table(
        ranked_view,
        [
            "timestamp",
            "symbol",
            "current_price",
            "expected_fair_value",
            "deviation_pct",
            "z_score",
            "spread_stability_score",
            "realized_volatility",
            "quote_volume",
            "funding_rate",
            "edge_after_fees_pct",
            "confidence_score",
            "calibrated_win_rate_pct",
            "calibration_sample_size",
            "pre_news_market_score",
            "news_impact_score",
            "news_bullish_score",
            "news_bearish_score",
            "news_bias",
            "news_event_count",
            "news_comment",
            "market_opportunity_score",
            "signal_quality",
            "passes_filters",
            "filter_failure_reason",
            "passes_zscore_filter",
            "passes_volatility_filter",
            "passes_liquidity_filter",
            "passes_stability_filter",
            "passes_funding_filter",
            "execution_status",
            "suggested_direction",
            "suggested_entry",
        ],
    )

    st.subheader("Compras sugeridas")
    buy_view = ranked_view.loc[ranked_view["suggested_direction"] == "LONG"] if not ranked_view.empty else ranked_view
    _display_table(
        buy_view,
        [
            "timestamp",
            "symbol",
            "current_price",
            "expected_fair_value",
            "deviation_pct",
            "z_score",
            "spread_stability_score",
            "realized_volatility",
            "edge_after_fees_pct",
            "confidence_score",
            "pre_news_market_score",
            "news_impact_score",
            "news_bias",
            "news_comment",
            "market_opportunity_score",
            "signal_quality",
            "passes_filters",
            "suggested_entry",
        ],
    )

    st.subheader("Mejor alerta del dia por moneda")
    _display_table(
        result.daily_best,
        [
            "timestamp",
            "symbol",
            "current_price",
            "expected_fair_value",
            "deviation_pct",
            "z_score",
            "confidence_score",
            "pre_news_market_score",
            "news_impact_score",
            "news_bias",
            "news_comment",
            "market_opportunity_score",
            "signal_quality",
            "suggested_direction",
            "suggested_entry",
        ],
    )

    st.subheader("Backtest")
    st.dataframe(result.backtest_stats, width="stretch")

    st.subheader("Dashboard terminal")
    st.code(result.dashboard_text, language="text")

st.subheader("Strategy Signals Monitor")
if strategy_signals_df.empty:
    st.info("No hay strategy signals exportadas por el monitor todavia.")
else:
    _display_table(
        strategy_signals_df,
        [
            "timestamp",
            "symbol",
            "strategy",
            "signal",
            "funding_rate",
            "basis",
            "cross_exchange_diff_pct",
            "risk_decision",
            "approved_position_size",
            "approved_leverage",
            "allocated_capital",
            "strategy_weight",
            "buy_exchange",
            "sell_exchange",
        ],
    )
