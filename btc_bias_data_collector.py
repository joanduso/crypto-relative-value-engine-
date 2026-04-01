from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd

from btc_market_bias_engine import _fetch_farside_btc_etf_flows, _fetch_glassnode_series


DEFAULT_ETF_PATH = Path(os.getenv("BTC_ETF_FLOWS_PATH", "btc_etf_flows.csv"))
DEFAULT_MVRV_PATH = Path(os.getenv("BTC_MVRV_PATH", "btc_mvrv.csv"))
DEFAULT_SOPR_PATH = Path(os.getenv("BTC_SOPR_PATH", "btc_sopr.csv"))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect external BTC bias data into local CSV files")
    parser.add_argument("--skip-etf", action="store_true")
    parser.add_argument("--skip-mvrv", action="store_true")
    parser.add_argument("--skip-sopr", action="store_true")
    parser.add_argument("--etf-path", default=str(DEFAULT_ETF_PATH))
    parser.add_argument("--mvrv-path", default=str(DEFAULT_MVRV_PATH))
    parser.add_argument("--sopr-path", default=str(DEFAULT_SOPR_PATH))
    parser.add_argument(
        "--mvrv-api-path",
        default=os.getenv("GLASSNODE_MVRV_PATH", "/v1/metrics/market/mvrv"),
        help="Glassnode API path for MVRV",
    )
    parser.add_argument(
        "--sopr-api-path",
        default=os.getenv("GLASSNODE_SOPR_PATH", "/v1/metrics/indicators/sopr_adjusted"),
        help="Glassnode API path for SOPR",
    )
    return parser.parse_args()


def _write_series_csv(frame: pd.DataFrame, output_path: Path) -> int:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if frame.empty:
        starter = pd.DataFrame({"timestamp": [], "value": []})
        starter.to_csv(output_path, index=False)
        return 0

    out = frame.copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True, errors="coerce")
    out["value"] = pd.to_numeric(out["value"], errors="coerce")
    out = out.dropna(subset=["timestamp", "value"]).sort_values("timestamp").drop_duplicates(subset=["timestamp"], keep="last")
    out["timestamp"] = out["timestamp"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    out[["timestamp", "value"]].to_csv(output_path, index=False)
    return int(len(out))


def collect_etf_flows(output_path: Path) -> int:
    frame = _fetch_farside_btc_etf_flows()
    return _write_series_csv(frame, output_path)


def collect_glassnode_metric(api_path: str, output_path: Path) -> int:
    frame = _fetch_glassnode_series(api_path)
    return _write_series_csv(frame, output_path)


def main() -> None:
    args = _parse_args()
    rows_written: dict[str, int] = {}

    if not args.skip_etf:
        rows_written["etf"] = collect_etf_flows(Path(args.etf_path))

    if not args.skip_mvrv:
        rows_written["mvrv"] = collect_glassnode_metric(args.mvrv_api_path, Path(args.mvrv_path))

    if not args.skip_sopr:
        rows_written["sopr"] = collect_glassnode_metric(args.sopr_api_path, Path(args.sopr_path))

    for name, count in rows_written.items():
        print(f"{name}: {count} rows")


if __name__ == "__main__":
    main()
