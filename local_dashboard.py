from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from data_ingestion import symbols_from_string
from engine import DEFAULT_ALTS, EngineRunConfig, run_engine
from signal_engine import EngineMode


st.set_page_config(page_title="Crypto Relative Value Engine", layout="wide")
STRATEGY_SIGNALS_PATH = Path("output/strategy_signals_latest.csv")


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
    st.dataframe(styled, use_container_width=True)


def _load_strategy_signals() -> pd.DataFrame:
    if not STRATEGY_SIGNALS_PATH.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(STRATEGY_SIGNALS_PATH)
    except Exception:
        return pd.DataFrame()


st.title("Crypto Relative Value Engine")
st.caption("Dashboard local con ejecucion manual, scoring y tabla ampliada por altcoin.")

strategy_signals_df = _load_strategy_signals()

with st.sidebar:
    st.header("Controles")
    mode = st.selectbox("Modo", [EngineMode.COPILOT.value, EngineMode.AUTO_SAFE.value], index=0)
    symbols_raw = st.text_input("Symbols", ",".join(DEFAULT_ALTS))
    interval = st.selectbox("Intervalo", ["1h", "4h", "15m"], index=0)
    limit = st.slider("Velas", min_value=300, max_value=1500, value=1000, step=100)
    st.caption(_interval_recommendation(interval))
    st.caption("Base sugerida: 4h para tendencia, 1h para setup y 15m para entrada.")
    live_mode = st.checkbox("Live mode", value=False)
    paper_trading = st.checkbox("Paper trading", value=True)
    dry_run = st.checkbox("Dry run", value=True)
    test_order_mode = st.checkbox("Test order mode", value=True)
    run_clicked = st.button("Run Engine", type="primary", use_container_width=True)
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
        result = run_engine(
            EngineRunConfig(
                mode=EngineMode(mode),
                symbols=symbols_from_string(symbols_raw, DEFAULT_ALTS),
                interval=interval,
                limit=limit,
                csv_path="output/copilot_trades.csv" if mode == EngineMode.COPILOT.value else "output/auto_safe_trades.csv",
                live_mode=live_mode,
                paper_trading=paper_trading,
                dry_run=dry_run,
                test_order_mode=test_order_mode,
            )
        )
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
            "market_opportunity_score",
            "signal_quality",
            "passes_filters",
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
            "market_opportunity_score",
            "signal_quality",
            "suggested_direction",
            "suggested_entry",
        ],
    )

    st.subheader("Backtest")
    st.dataframe(result.backtest_stats, use_container_width=True)

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
