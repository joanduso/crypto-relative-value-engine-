from __future__ import annotations

import pandas as pd


def select_portfolio(trades_df: pd.DataFrame) -> pd.DataFrame:
    if trades_df.empty:
        return trades_df.copy()

    ordered = trades_df.sort_values("opportunity_score", ascending=False, na_position="last").reset_index(drop=True)
    selected_rows: list[pd.Series] = []
    direction_allocations = {"LONG": 0.0, "SHORT": 0.0}

    for _, row in ordered.iterrows():
        if len(selected_rows) >= 5:
            break

        direction = str(row.get("direction", "")).upper()
        position_size = float(row.get("position_size_pct", 0.0) or 0.0)
        if position_size <= 0:
            continue
        if direction not in direction_allocations:
            continue
        if direction_allocations[direction] + position_size > 60.0:
            continue

        direction_allocations[direction] += position_size
        selected_rows.append(row)

    if not selected_rows:
        return ordered.iloc[0:0].copy()
    return pd.DataFrame(selected_rows).reset_index(drop=True)
