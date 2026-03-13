from __future__ import annotations

from pathlib import Path

import pandas as pd


def _format_float(value: float, digits: int = 4) -> str:
    if pd.isna(value):
        return "n/a"
    return f"{value:.{digits}f}"


def render_terminal_dashboard(
    mode_name: str,
    live_mode: bool,
    proposals: pd.DataFrame,
    backtest_stats: pd.DataFrame,
) -> str:
    lines = [
        "Crypto Relative Value Engine",
        f"Mode: {mode_name}",
        f"Live mode: {'ON' if live_mode else 'OFF'}",
        "",
        "Top Opportunities",
    ]
    if proposals.empty:
        lines.append("No opportunities passed the active filters.")
    else:
        lines.append("symbol | current | fair | dev% | z | stability | dir | quality | entry | stop | take | size | confidence")
        for row in proposals.itertuples(index=False):
            lines.append(
                " | ".join(
                    [
                        str(row.symbol),
                        _format_float(row.current_price, 4),
                        _format_float(row.expected_fair_value, 4),
                        _format_float(row.deviation_pct, 2),
                        _format_float(row.z_score, 2),
                        _format_float(row.spread_stability_score, 2),
                        str(row.suggested_direction),
                        str(row.signal_quality),
                        _format_float(row.suggested_entry, 4),
                        _format_float(row.suggested_stop_loss, 4),
                        _format_float(row.suggested_take_profit, 4),
                        _format_float(row.suggested_position_size, 4),
                        _format_float(row.confidence_score, 1),
                    ]
                )
            )

    lines.extend(["", "Backtest Comparison", "mode | total_return% | sharpe | max_drawdown% | win_rate% | avg_hold_hours | pnl_after_fees%"])
    for row in backtest_stats.itertuples(index=False):
        lines.append(
            " | ".join(
                [
                    str(row.mode),
                    _format_float(row.total_return_pct, 2),
                    _format_float(row.sharpe, 2),
                    _format_float(row.max_drawdown_pct, 2),
                    _format_float(row.win_rate_pct, 2),
                    _format_float(row.avg_holding_hours, 1),
                    _format_float(row.pnl_after_fees_pct, 2),
                ]
            )
        )
    return "\n".join(lines)


def export_proposals_csv(proposals: pd.DataFrame, path: str | Path) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    proposals.to_csv(output_path, index=False)
    return output_path
