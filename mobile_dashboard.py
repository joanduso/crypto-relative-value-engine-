from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pandas as pd
from flask import Flask, jsonify, render_template, request


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
                "timeframe": _value_or_dash(row, "timeframe"),
                "score": _value_or_dash(row, "market_opportunity_score"),
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

    return {
        "latest_count": latest_count,
        "history_count": history_count,
        "signals_count": signals_count,
        "top_symbol": top_symbol,
        "top_score": top_score,
        "updated_at": updated_at,
        "direction_mix": direction_mix,
    }


def _prepare_tables() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    latest_df = _normalize_frame(_load_csv_family(MONITOR_LATEST_PATH))
    history_df = _normalize_frame(_load_csv_family(DAILY_HISTORY_PATH))
    signals_df = _normalize_frame(_load_csv_family(STRATEGY_SIGNALS_PATH))
    return latest_df, history_df, signals_df


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
    filtered_latest = _apply_filters(latest_df)
    filtered_history = _apply_filters(history_df)

    context = {
        "summary": _summary(latest_df, history_df, signals_df),
        "opportunities": _opportunity_cards(filtered_latest),
        "history": _opportunity_cards(filtered_history.head(20)),
        "signals": _signal_cards(signals_df.head(20)),
        "filters": {
            "direction": request.args.get("direction", ""),
            "min_score": request.args.get("min_score", ""),
            "min_confidence": request.args.get("min_confidence", ""),
            "quality": request.args.get("quality", ""),
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
    latest_df, _, _ = _prepare_tables()
    filtered_latest = _apply_filters(latest_df)
    return jsonify(_opportunity_cards(filtered_latest))


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


if __name__ == "__main__":
    port = int(os.getenv("PORT", "8080"))
    print("Starting server...")
    print(f"PORT: {port}")
    app.run(host="0.0.0.0", port=port, debug=False, use_reloader=False)
