from __future__ import annotations

import pandas as pd


TIMEFRAME_PRIORITY = {"15m": 1, "1h": 2, "4h": 3}


def _priority(timeframe: str) -> int:
    return TIMEFRAME_PRIORITY.get(str(timeframe), 0)


def build_multi_timeframe_alignment(latest_df: pd.DataFrame) -> pd.DataFrame:
    if latest_df.empty:
        return pd.DataFrame(
            columns=[
                "symbol",
                "direction",
                "alignment_score",
                "timeframes_confirmed",
                "top_timeframe",
                "timestamp",
                "base_score",
                "current_price",
                "realized_volatility",
                "quote_volume",
            ]
        )

    rows: list[dict[str, object]] = []
    ordered = latest_df.sort_values(["symbol", "timestamp"], ascending=[True, False])

    for symbol, group in ordered.groupby("symbol", sort=False):
        latest_per_tf = (
            group.sort_values("timestamp", ascending=False)
            .drop_duplicates(subset=["timeframe"], keep="first")
            .copy()
        )
        direction_counts = latest_per_tf["suggested_direction"].astype(str).value_counts()
        majority_direction = str(direction_counts.idxmax()) if not direction_counts.empty else "NEUTRAL"
        confirmed = latest_per_tf.loc[
            latest_per_tf["suggested_direction"].astype(str) == majority_direction, "timeframe"
        ].astype(str)
        confirmed_list = sorted(confirmed.tolist(), key=_priority, reverse=True)

        distinct_directions = latest_per_tf["suggested_direction"].astype(str).nunique()
        if distinct_directions == 1:
            if len(confirmed_list) >= 3:
                alignment_score = 100.0
            elif len(confirmed_list) == 2:
                alignment_score = 75.0
            else:
                alignment_score = 55.0
        elif len(confirmed_list) >= 2:
            alignment_score = 45.0
        else:
            alignment_score = 20.0

        majority_rows = latest_per_tf.loc[latest_per_tf["suggested_direction"].astype(str) == majority_direction].copy()
        if majority_rows.empty:
            majority_rows = latest_per_tf.copy()

        majority_rows["tf_priority"] = majority_rows["timeframe"].map(_priority)
        top_row = majority_rows.sort_values(
            ["tf_priority", "market_opportunity_score", "confidence_score"],
            ascending=[False, False, False],
            na_position="last",
        ).iloc[0]

        rows.append(
            {
                "symbol": symbol,
                "direction": majority_direction,
                "alignment_score": alignment_score,
                "timeframes_confirmed": ",".join(confirmed_list),
                "top_timeframe": top_row.get("timeframe", ""),
                "timestamp": top_row.get("timestamp"),
                "signal_quality": top_row.get("signal_quality", ""),
                "base_score": float(top_row.get("market_opportunity_score", 0.0) or 0.0),
                "current_price": float(top_row.get("current_price", 0.0) or 0.0),
                "realized_volatility": float(top_row.get("realized_volatility", 0.0) or 0.0),
                "quote_volume": float(top_row.get("quote_volume", 0.0) or 0.0),
            }
        )

    return pd.DataFrame(rows)
