from __future__ import annotations

from pathlib import Path

import pandas as pd

from interval_profiles import profile_for_interval

from .multi_timeframe import build_multi_timeframe_alignment
from .scoring import build_opportunity_score_frame


def load_monitor_latest_family(base_path: str | Path) -> pd.DataFrame:
    path = Path(base_path)
    frames: list[pd.DataFrame] = []

    for csv_path in sorted(path.parent.glob(f"{path.stem}*.csv")):
        try:
            df = pd.read_csv(csv_path)
        except Exception:
            continue
        if df.empty:
            continue
        if "timeframe" not in df.columns:
            suffix = csv_path.stem.removeprefix(path.stem).lstrip("_")
            df["timeframe"] = suffix or "1h"
        df["snapshot_time_utc"] = pd.Timestamp(csv_path.stat().st_mtime, unit="s", tz="UTC")
        frames.append(df)

    if not frames and path.exists():
        try:
            df = pd.read_csv(path)
        except Exception:
            return pd.DataFrame()
        if not df.empty and "timeframe" not in df.columns:
            df["timeframe"] = "1h"
        if not df.empty:
            df["snapshot_time_utc"] = pd.Timestamp(path.stat().st_mtime, unit="s", tz="UTC")
        frames.append(df)

    if not frames:
        return pd.DataFrame()

    latest_df = pd.concat(frames, ignore_index=True)
    latest_df["timestamp"] = pd.to_datetime(latest_df.get("timestamp"), errors="coerce", utc=True)
    return latest_df


def _filter_fresh_snapshots(latest_df: pd.DataFrame) -> pd.DataFrame:
    if latest_df.empty or "snapshot_time_utc" not in latest_df.columns or "timeframe" not in latest_df.columns:
        return latest_df

    snapshot_times = pd.to_datetime(latest_df["snapshot_time_utc"], errors="coerce", utc=True)
    now_utc = pd.Timestamp.utcnow()
    if now_utc.tzinfo is None:
        now_utc = now_utc.tz_localize("UTC")

    keep_mask: list[bool] = []
    for timeframe, snapshot_time in zip(latest_df["timeframe"], snapshot_times):
        if pd.isna(snapshot_time):
            keep_mask.append(False)
            continue

        profile = profile_for_interval(str(timeframe))
        max_age = pd.Timedelta(minutes=max(profile.poll_minutes * 3, 15))
        keep_mask.append((now_utc - snapshot_time) <= max_age)

    return latest_df.loc[keep_mask].copy()


def build_opportunity_table(latest_df: pd.DataFrame) -> pd.DataFrame:
    if latest_df.empty:
        return pd.DataFrame(
            columns=[
                "symbol",
                "direction",
                "opportunity_score",
                "alignment_score",
                "timeframes_confirmed",
                "top_timeframe",
                "timestamp",
                "current_price",
                "realized_volatility",
                "quote_volume",
                "base_score",
            ]
        )

    latest_df = _filter_fresh_snapshots(latest_df)
    if latest_df.empty:
        return pd.DataFrame(
            columns=[
                "symbol",
                "direction",
                "opportunity_score",
                "alignment_score",
                "timeframes_confirmed",
                "top_timeframe",
                "timestamp",
                "current_price",
                "realized_volatility",
                "quote_volume",
                "base_score",
            ]
        )

    aligned = build_multi_timeframe_alignment(latest_df)
    scored = build_opportunity_score_frame(aligned)
    ordered = scored.sort_values(
        ["opportunity_score", "alignment_score", "base_score"],
        ascending=[False, False, False],
        na_position="last",
    ).reset_index(drop=True)
    return ordered
