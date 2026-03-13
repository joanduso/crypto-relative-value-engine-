from __future__ import annotations

import argparse

from data_ingestion import symbols_from_string
from engine import DEFAULT_ALTS, EngineRunConfig, run_engine
from signal_engine import EngineMode


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dual-mode crypto relative value engine")
    parser.add_argument("--mode", choices=[mode.value for mode in EngineMode], default=EngineMode.COPILOT.value)
    parser.add_argument("--symbols", help="Comma separated altcoin symbols, e.g. XRPUSDT,SOLUSDT")
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--interval", default="1h")
    parser.add_argument("--csv-path", default="output/proposed_trades.csv")
    parser.add_argument("--live-mode", action="store_true", help="Enable live execution. Off by default.")
    parser.add_argument("--paper-trading", action="store_true")
    parser.add_argument("--dry-run", action="store_true", default=False)
    parser.add_argument("--test-order-mode", action="store_true")
    return parser.parse_args()
def run() -> None:
    args = parse_args()
    result = run_engine(
        EngineRunConfig(
            mode=EngineMode(args.mode),
            symbols=symbols_from_string(args.symbols, DEFAULT_ALTS),
            interval=args.interval,
            limit=args.limit,
            csv_path=args.csv_path,
            live_mode=args.live_mode,
            paper_trading=args.paper_trading,
            dry_run=args.dry_run,
            test_order_mode=args.test_order_mode,
        )
    )
    print(result.dashboard_text)
    print("")
    print(f"CSV export: {result.csv_path}")


if __name__ == "__main__":
    run()
