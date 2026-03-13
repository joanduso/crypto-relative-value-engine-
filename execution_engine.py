from __future__ import annotations

import hashlib
import hmac
import json
import os
import time
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlencode

import requests

try:
    import websocket
except Exception:  # pragma: no cover
    websocket = None


BINANCE_API_BASE = "https://api.binance.com"
BINANCE_WS_BASE = "wss://stream.binance.com:9443/ws"


@dataclass(frozen=True)
class ExecutionConfig:
    api_key: str | None = None
    api_secret: str | None = None
    dry_run: bool = True
    paper_trading: bool = False
    test_order_mode: bool = True
    live_mode: bool = False


class BinanceExecutionEngine:
    def __init__(self, config: ExecutionConfig) -> None:
        self.config = config
        self.session = requests.Session()
        self.session.trust_env = False

    def _signed_request(self, method: str, path: str, params: dict[str, Any]) -> dict[str, Any]:
        if not self.config.api_key or not self.config.api_secret:
            raise RuntimeError("Binance API credentials are required for live execution.")
        params = {**params, "timestamp": int(time.time() * 1000)}
        query = urlencode(params)
        signature = hmac.new(
            self.config.api_secret.encode("utf-8"),
            query.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()
        response = self.session.request(
            method=method,
            url=f"{BINANCE_API_BASE}{path}?{query}&signature={signature}",
            headers={"X-MBX-APIKEY": self.config.api_key},
            timeout=20,
        )
        response.raise_for_status()
        return response.json()

    def place_order(self, order: dict[str, Any]) -> dict[str, Any]:
        if self.config.dry_run or self.config.paper_trading or not self.config.live_mode:
            return {"status": "SIMULATED", "details": order}

        endpoint = "/api/v3/order/test" if self.config.test_order_mode else "/api/v3/order"
        return self._signed_request("POST", endpoint, order)

    def create_market_order(self, symbol: str, side: str, quantity: float) -> dict[str, Any]:
        order = {
            "symbol": symbol,
            "side": side,
            "type": "MARKET",
            "quantity": f"{quantity:.6f}",
        }
        return self.place_order(order)

    def start_user_data_stream(self) -> str:
        if not self.config.api_key:
            raise RuntimeError("Binance API credentials are required for websocket monitoring.")
        response = self.session.post(
            f"{BINANCE_API_BASE}/api/v3/userDataStream",
            headers={"X-MBX-APIKEY": self.config.api_key},
            timeout=20,
        )
        response.raise_for_status()
        return response.json()["listenKey"]

    def monitor_fills(self, handler) -> None:
        if websocket is None:
            raise RuntimeError("websocket-client is required for user data stream monitoring.")
        listen_key = self.start_user_data_stream()
        ws = websocket.WebSocketApp(
            f"{BINANCE_WS_BASE}/{listen_key}",
            on_message=lambda _ws, message: handler(json.loads(message)),
        )
        ws.run_forever()


def execution_config_from_env(
    live_mode: bool,
    dry_run: bool,
    paper_trading: bool,
    test_order_mode: bool,
) -> ExecutionConfig:
    return ExecutionConfig(
        api_key=os.getenv("BINANCE_API_KEY"),
        api_secret=os.getenv("BINANCE_API_SECRET"),
        dry_run=dry_run,
        paper_trading=paper_trading,
        test_order_mode=test_order_mode,
        live_mode=live_mode,
    )
