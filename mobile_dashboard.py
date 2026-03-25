from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pandas as pd
from flask import Flask, jsonify, render_template, request

from interval_profiles import profile_for_interval
from news_engine import news_comment
from quant_engine import (
    apply_risk_management,
    build_opportunity_table,
    build_trade_setups,
    load_monitor_latest_family,
    select_portfolio,
)


BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "output"

MONITOR_LATEST_PATH = OUTPUT_DIR / "monitor_latest.csv"
DAILY_HISTORY_PATH = OUTPUT_DIR / "daily_alert_history.csv"
STRATEGY_SIGNALS_PATH = OUTPUT_DIR / "strategy_signals_latest.csv"
EMBEDDED_MONITOR_PID_PATH = OUTPUT_DIR / "embedded_monitor.pid"

DEFAULT_LIMIT = 50


app = Flask(__name__)


def _load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def _timeframe_from_path(path: Path, stem_prefix: str) -> str:
    suffix = path.stem.removeprefix(stem_prefix)
    if suffix.startswith("_"):
        return suffix[1:]
    return "1h"


def _load_csv_family(base_path: Path) -> pd.DataFrame:
    pattern = f"{base_path.stem}*.csv"
    frames: list[pd.DataFrame] = []
    seen_paths: set[Path] = set()

    for path in sorted(base_path.parent.glob(pattern)):
        seen_paths.add(path)
        df = _load_csv(path)
        if df.empty:
            continue
        if "timeframe" not in df.columns:
            df["timeframe"] = _timeframe_from_path(path, base_path.stem)
        frames.append(df)

    if base_path not in seen_paths:
        df = _load_csv(base_path)
        if not df.empty:
            if "timeframe" not in df.columns:
                df["timeframe"] = "1h"
            frames.append(df)

    if not frames:
        return pd.DataFrame()

    return pd.concat(frames, ignore_index=True)


def _normalize_frame(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()

    out = df.copy()
    if "timestamp" in out.columns:
        out["timestamp"] = pd.to_datetime(out["timestamp"], errors="coerce", utc=True)
    return out


def _apply_filters(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()

    out = df.copy()

    direction = request.args.get("direction", "").strip().upper()
    if direction in {"LONG", "SHORT"} and "suggested_direction" in out.columns:
        out = out.loc[out["suggested_direction"].astype(str).str.upper() == direction]

    min_score = request.args.get("min_score", type=float)
    if min_score is not None and "market_opportunity_score" in out.columns:
        out = out.loc[pd.to_numeric(out["market_opportunity_score"], errors="coerce") >= min_score]

    min_confidence = request.args.get("min_confidence", type=float)
    if min_confidence is not None and "confidence_score" in out.columns:
        out = out.loc[pd.to_numeric(out["confidence_score"], errors="coerce") >= min_confidence]

    quality = request.args.get("quality", "").strip().upper()
    if quality and "signal_quality" in out.columns:
        out = out.loc[out["signal_quality"].astype(str).str.upper() == quality]

    execution_status = request.args.get("execution_status", "").strip().upper()
    if execution_status and "execution_status" in out.columns:
        out = out.loc[out["execution_status"].astype(str).str.upper() == execution_status]

    timeframe = request.args.get("timeframe", "").strip().lower()
    if timeframe and "timeframe" in out.columns:
        out = out.loc[out["timeframe"].astype(str).str.lower() == timeframe]

    passed_only = request.args.get("passed_only", "").strip().lower() in {"1", "true", "yes", "on"}
    if passed_only and "passes_filters" in out.columns:
        bool_map = {"true": True, "false": False}
        normalized = out["passes_filters"].astype(str).str.lower().map(bool_map)
        out = out.loc[normalized.fillna(False)]

    sort_column = "market_opportunity_score" if "market_opportunity_score" in out.columns else None
    if sort_column is not None:
        out = out.sort_values(by=sort_column, ascending=False, na_position="last")

    limit = request.args.get("limit", default=DEFAULT_LIMIT, type=int)
    if limit is not None and limit > 0:
        out = out.head(limit)

    return out


def _format_timestamp(value: object) -> str:
    if pd.isna(value):
        return "-"
    if isinstance(value, pd.Timestamp):
        return value.strftime("%Y-%m-%d %H:%M UTC")
    return str(value)


def _value_or_dash(row: pd.Series, column: str) -> object:
    value = row.get(column)
    if pd.isna(value):
        return "-"
    return value


def _opportunity_cards(df: pd.DataFrame) -> list[dict[str, object]]:
    cards: list[dict[str, object]] = []
    if df.empty:
        return cards

    for _, row in df.iterrows():
        cards.append(
            {
                "timestamp": _format_timestamp(row.get("timestamp")),
                "symbol": _value_or_dash(row, "symbol"),
                "direction": _value_or_dash(row, "suggested_direction"),
                "quality": _value_or_dash(row, "signal_quality"),
                "base_quality": _value_or_dash(row, "base_signal_quality"),
                "execution_status": _value_or_dash(row, "execution_status"),
                "market_regime": _value_or_dash(row, "btc_regime"),
                "timeframe": _value_or_dash(row, "timeframe"),
                "score": _value_or_dash(row, "market_opportunity_score"),
                "base_score": _value_or_dash(row, "base_market_opportunity_score"),
                "pre_news_score": _value_or_dash(row, "pre_news_market_score"),
                "news_impact_score": _value_or_dash(row, "news_impact_score"),
                "news_bias": _value_or_dash(row, "news_bias"),
                "news_event_count": _value_or_dash(row, "news_event_count"),
                "news_comment": _value_or_dash(row, "news_comment") if "news_comment" in row else news_comment(
                    row.get("news_impact_score"),
                    row.get("news_event_count"),
                    row.get("news_bias"),
                ),
                "confidence": _value_or_dash(row, "confidence_score"),
                "entry": _value_or_dash(row, "suggested_entry"),
                "price": _value_or_dash(row, "current_price"),
                "deviation_pct": _value_or_dash(row, "deviation_pct"),
                "z_score": _value_or_dash(row, "z_score"),
                "edge_after_fees_pct": _value_or_dash(row, "edge_after_fees_pct"),
                "stop_loss": _value_or_dash(row, "suggested_stop_loss"),
                "take_profit": _value_or_dash(row, "suggested_take_profit"),
            }
        )
    return cards


def _signal_cards(df: pd.DataFrame) -> list[dict[str, object]]:
    cards: list[dict[str, object]] = []
    if df.empty:
        return cards

    for _, row in df.iterrows():
        cards.append(
            {
                "timestamp": _format_timestamp(row.get("timestamp")),
                "symbol": _value_or_dash(row, "symbol"),
                "strategy": _value_or_dash(row, "strategy"),
                "signal": _value_or_dash(row, "signal"),
                "timeframe": _value_or_dash(row, "timeframe"),
                "risk_decision": _value_or_dash(row, "risk_decision"),
                "allocated_capital": _value_or_dash(row, "allocated_capital"),
                "approved_position_size": _value_or_dash(row, "approved_position_size"),
                "approved_leverage": _value_or_dash(row, "approved_leverage"),
                "directional_score": _value_or_dash(row, "directional_score"),
                "market_regime": _value_or_dash(row, "market_regime"),
            }
        )
    return cards


def _summary(latest_df: pd.DataFrame, history_df: pd.DataFrame, signals_df: pd.DataFrame) -> dict[str, object]:
    latest_count = int(len(latest_df))
    history_count = int(len(history_df))
    signals_count = int(len(signals_df))

    top_symbol = "-"
    top_score = "-"
    updated_at = "-"

    if not latest_df.empty:
        top_row = latest_df.sort_values(by="market_opportunity_score", ascending=False, na_position="last").iloc[0]
        top_symbol = str(top_row.get("symbol", "-"))
        top_score = _value_or_dash(top_row, "market_opportunity_score")
        updated_at = _format_timestamp(top_row.get("timestamp"))

    direction_mix = {"LONG": 0, "SHORT": 0}
    if not latest_df.empty and "suggested_direction" in latest_df.columns:
        counts = latest_df["suggested_direction"].astype(str).str.upper().value_counts()
        direction_mix["LONG"] = int(counts.get("LONG", 0))
        direction_mix["SHORT"] = int(counts.get("SHORT", 0))

    timeframe_coverage: list[str] = []
    timeframe_status: list[dict[str, object]] = []
    if not latest_df.empty and "timeframe" in latest_df.columns:
        timeframe_coverage = sorted({str(value) for value in latest_df["timeframe"].dropna().tolist()})
    for timeframe in ("15m", "1h", "4h"):
        frame = latest_df.loc[latest_df["timeframe"].astype(str) == timeframe] if not latest_df.empty and "timeframe" in latest_df.columns else pd.DataFrame()
        timeframe_status.append(
            {
                "timeframe": timeframe,
                "rows": int(len(frame)),
                "state": "active" if not frame.empty else "empty",
            }
        )

    return {
        "latest_count": latest_count,
        "history_count": history_count,
        "signals_count": signals_count,
        "top_symbol": top_symbol,
        "top_score": top_score,
        "updated_at": updated_at,
        "direction_mix": direction_mix,
        "timeframe_coverage": timeframe_coverage,
        "timeframe_status": timeframe_status,
    }


def _dashboard_guide() -> dict[str, object]:
    profiles = {}
    for timeframe in ("15m", "1h", "4h"):
        profile = profile_for_interval(timeframe)
        profiles[timeframe] = {
            "limit": profile.limit,
            "poll_minutes": profile.poll_minutes,
            "regression_window": profile.feature_config.regression_window,
            "zscore_window": profile.feature_config.zscore_window,
            "stability_window": profile.feature_config.stability_window,
            "volatility_window": profile.feature_config.volatility_window,
        }

    return {
        "opp_score_formula": "0.4 * base_score_normalized + 0.3 * alignment_score + 0.2 * volatility_score + 0.1 * liquidity_score",
        "alignment_notes": [
            "100 = 15m, 1h y 4h alineados en la misma direccion",
            "75 = 2 timeframes alineados",
            "55 = solo 1 timeframe claro",
            "45 = mezcla de direcciones con mayoria parcial",
            "20 = muy poca confirmacion",
        ],
        "rr_formula": "RR = 2.0 fijo en el quant trade setup actual",
        "entry_note": "Entry en Top oportunidades usa current_price del timeframe dominante seleccionado por el agregador.",
        "dev_note": "Dev % = (precio actual / fair value - 1) * 100. Negativo suele favorecer LONG; positivo suele favorecer SHORT.",
        "edge_note": "Edge fees = abs(dev %) - fee_bps. Mide cuanto edge queda despues de costos aproximados.",
        "news_note": "News impact suma o resta hasta 15 puntos sobre el market score base segun fuente, severidad, confianza, relevancia, recencia y confirmacion de mercado.",
        "regime_note": "Regime usa BTC como proxy macro: trending_bullish, trending_bearish o range segun EMA50/200 4h, pendiente y directional score.",
        "profiles": profiles,
        "timeframe_note": "15m se usa para timing, 1h para setup y 4h para contexto. Hoy el motor no corre 5m.",
    }


def _prepare_tables() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    latest_df = _normalize_frame(_load_csv_family(MONITOR_LATEST_PATH))
    history_df = _normalize_frame(_load_csv_family(DAILY_HISTORY_PATH))
    signals_df = _normalize_frame(_load_csv_family(STRATEGY_SIGNALS_PATH))
    return latest_df, history_df, signals_df


def _build_quant_opportunities() -> pd.DataFrame:
    latest_df = load_monitor_latest_family(MONITOR_LATEST_PATH)
    opportunities = build_opportunity_table(latest_df)
    trades = build_trade_setups(opportunities)
    managed = apply_risk_management(trades)
    portfolio = select_portfolio(managed)
    if portfolio.empty:
        return managed
    return portfolio


def _pid_is_running(pid_path: Path) -> bool:
    if not pid_path.exists():
        return False
    try:
        pid = int(pid_path.read_text(encoding="utf-8").strip())
    except Exception:
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def _start_embedded_monitor_if_enabled() -> None:
    enabled = os.getenv("EMBED_MONITOR_IN_WEB", "false").strip().lower() in {"1", "true", "yes", "on"}
    if not enabled:
        return
    if _pid_is_running(EMBEDDED_MONITOR_PID_PATH):
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    process = subprocess.Popen(
        [os.sys.executable, "monitor.py"],
        cwd=BASE_DIR,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    EMBEDDED_MONITOR_PID_PATH.write_text(str(process.pid), encoding="utf-8")


_start_embedded_monitor_if_enabled()


@app.get("/")
def index() -> str:
    latest_df, history_df, signals_df = _prepare_tables()
    quant_df = _build_quant_opportunities()
    filtered_latest = _apply_filters(latest_df)
    filtered_history = _apply_filters(history_df)
    classified_latest = filtered_latest.sort_values(
        ["market_opportunity_score", "confidence_score", "signal_quality"],
        ascending=[False, False, True],
        na_position="last",
    ) if not filtered_latest.empty else filtered_latest
    timeframe_views = {
        timeframe: _opportunity_cards(
            filtered_latest.loc[filtered_latest["timeframe"].astype(str) == timeframe].head(12)
        ) if not filtered_latest.empty and "timeframe" in filtered_latest.columns else []
        for timeframe in ("15m", "1h", "4h")
    }

    context = {
        "summary": _summary(latest_df, history_df, signals_df),
        "guide": _dashboard_guide(),
        "opportunities": _quant_cards(quant_df),
        "classified_opportunities": _opportunity_cards(classified_latest.head(24)),
        "timeframe_views": timeframe_views,
        "history": _opportunity_cards(filtered_history.head(20)),
        "signals": _signal_cards(signals_df.head(20)),
        "filters": {
            "direction": request.args.get("direction", ""),
            "min_score": request.args.get("min_score", ""),
            "min_confidence": request.args.get("min_confidence", ""),
            "quality": request.args.get("quality", ""),
            "execution_status": request.args.get("execution_status", ""),
            "timeframe": request.args.get("timeframe", ""),
            "passed_only": request.args.get("passed_only", ""),
            "limit": request.args.get("limit", str(DEFAULT_LIMIT)),
        },
    }
    return render_template("mobile_dashboard.html", **context)


@app.get("/health")
def health() -> tuple[dict[str, str], int]:
    return {"status": "ok"}, 200


@app.get("/api/opportunities")
def api_opportunities():
    opportunities_df = _build_quant_opportunities()
    limit = request.args.get("limit", default=DEFAULT_LIMIT, type=int)
    if limit is not None and limit > 0:
        opportunities_df = opportunities_df.head(limit)
    return jsonify(_quant_cards(opportunities_df))


@app.get("/api/history")
def api_history():
    _, history_df, _ = _prepare_tables()
    filtered_history = _apply_filters(history_df)
    return jsonify(_opportunity_cards(filtered_history))


@app.get("/api/signals")
def api_signals():
    _, _, signals_df = _prepare_tables()
    return jsonify(_signal_cards(signals_df.head(50)))


@app.get("/api/summary")
def api_summary():
    latest_df, history_df, signals_df = _prepare_tables()
    return jsonify(_summary(latest_df, history_df, signals_df))


@app.get("/api/classified-opportunities")
def api_classified_opportunities():
    latest_df, _, _ = _prepare_tables()
    filtered_latest = _apply_filters(latest_df)
    classified_latest = filtered_latest.sort_values(
        ["market_opportunity_score", "confidence_score", "signal_quality"],
        ascending=[False, False, True],
        na_position="last",
    ) if not filtered_latest.empty else filtered_latest
    return jsonify(_opportunity_cards(classified_latest.head(50)))


@app.get("/api/timeframe-universe")
def api_timeframe_universe():
    latest_df, _, _ = _prepare_tables()
    filtered_latest = _apply_filters(latest_df)
    payload = {
        timeframe: _opportunity_cards(
            filtered_latest.loc[filtered_latest["timeframe"].astype(str) == timeframe].head(50)
        ) if not filtered_latest.empty and "timeframe" in filtered_latest.columns else []
        for timeframe in ("15m", "1h", "4h")
    }
    return jsonify(payload)


def _quant_cards(df: pd.DataFrame) -> list[dict[str, object]]:
    cards: list[dict[str, object]] = []
    if df.empty:
        return cards

    for _, row in df.iterrows():
        cards.append(
            {
                "timestamp": _format_timestamp(row.get("timestamp")),
                "symbol": _value_or_dash(row, "symbol"),
                "direction": _value_or_dash(row, "direction"),
                "timeframe": _value_or_dash(row, "top_timeframe"),
                "score": _value_or_dash(row, "opportunity_score"),
                "alignment_score": _value_or_dash(row, "alignment_score"),
                "timeframes_confirmed": _value_or_dash(row, "timeframes_confirmed"),
                "entry": _value_or_dash(row, "entry"),
                "price": _value_or_dash(row, "current_price"),
                "stop_loss": _value_or_dash(row, "stop_loss"),
                "take_profit": _value_or_dash(row, "take_profit"),
                "risk_reward_ratio": _value_or_dash(row, "risk_reward_ratio"),
                "position_size_pct": _value_or_dash(row, "position_size_pct"),
            }
        )
    return cards


if __name__ == "__main__":
    port = int(os.getenv("PORT", "8080"))
    print("Starting server...")
    print(f"PORT: {port}")
    app.run(host="0.0.0.0", port=port, debug=False, use_reloader=False)
