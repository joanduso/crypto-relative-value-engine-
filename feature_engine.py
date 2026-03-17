from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


DEFAULT_WINDOW = 24 * 14
STABILITY_WINDOW = 24 * 3
VOL_WINDOW = 24


@dataclass(frozen=True)
class FeatureConfig:
    regression_window: int = DEFAULT_WINDOW
    zscore_window: int = DEFAULT_WINDOW
    stability_window: int = STABILITY_WINDOW
    volatility_window: int = VOL_WINDOW


def _rolling_regression(
    log_alt: pd.Series,
    log_btc: pd.Series,
    log_eth: pd.Series,
    window: int,
) -> pd.DataFrame:
    expected = np.full(len(log_alt), np.nan, dtype=float)
    beta_btc = np.full(len(log_alt), np.nan, dtype=float)
    beta_eth = np.full(len(log_alt), np.nan, dtype=float)
    intercepts = np.full(len(log_alt), np.nan, dtype=float)

    y = log_alt.to_numpy(dtype=float)
    x_btc = log_btc.to_numpy(dtype=float)
    x_eth = log_eth.to_numpy(dtype=float)

    for idx in range(window - 1, len(log_alt)):
        start = idx - window + 1
        design = np.column_stack(
            [
                np.ones(window, dtype=float),
                x_btc[start : idx + 1],
                x_eth[start : idx + 1],
            ]
        )
        target = y[start : idx + 1]
        coeffs, *_ = np.linalg.lstsq(design, target, rcond=None)
        intercepts[idx] = coeffs[0]
        beta_btc[idx] = coeffs[1]
        beta_eth[idx] = coeffs[2]
        expected[idx] = coeffs[0] + coeffs[1] * x_btc[idx] + coeffs[2] * x_eth[idx]

    return pd.DataFrame(
        {
            "predicted_log_price": expected,
            "beta_btc": beta_btc,
            "beta_eth": beta_eth,
            "intercept": intercepts,
        },
        index=log_alt.index,
    )


def build_feature_frame(
    market_df: pd.DataFrame,
    alt_symbols: tuple[str, ...],
    config: FeatureConfig | None = None,
) -> pd.DataFrame:
    cfg = config or FeatureConfig()
    pivot_close = market_df.pivot(index="timestamp", columns="symbol", values="close").sort_index()
    pivot_volume = market_df.pivot(index="timestamp", columns="symbol", values="quote_volume").sort_index()
    funding = (
        market_df.sort_values("timestamp")
        .groupby("symbol", as_index=False)
        .tail(1)[["symbol", "funding_rate"]]
        .set_index("symbol")["funding_rate"]
        .to_dict()
    )

    records: list[pd.DataFrame] = []
    for symbol in alt_symbols:
        if symbol not in pivot_close.columns or "BTCUSDT" not in pivot_close.columns or "ETHUSDT" not in pivot_close.columns:
            continue
        subset = pd.DataFrame(
            {
                "timestamp": pivot_close.index,
                "symbol": symbol,
                "alt_price": pivot_close[symbol],
                "btc_price": pivot_close["BTCUSDT"],
                "eth_price": pivot_close["ETHUSDT"],
                "quote_volume": pivot_volume[symbol],
            }
        ).dropna()
        if len(subset) < cfg.regression_window:
            continue

        subset["log_alt"] = np.log(subset["alt_price"])
        subset["log_btc"] = np.log(subset["btc_price"])
        subset["log_eth"] = np.log(subset["eth_price"])

        regression_df = _rolling_regression(
            subset["log_alt"],
            subset["log_btc"],
            subset["log_eth"],
            cfg.regression_window,
        )
        subset = pd.concat([subset, regression_df], axis=1)
        subset["fair_value"] = np.exp(subset["predicted_log_price"])
        subset["spread"] = subset["log_alt"] - subset["beta_btc"] * subset["log_btc"] - subset["beta_eth"] * subset["log_eth"]
        subset["residual_log"] = subset["log_alt"] - subset["predicted_log_price"]
        zscore_min_periods = min(cfg.zscore_window, max(24, cfg.zscore_window // 3))
        vol_min_periods = min(cfg.volatility_window, 12)
        stability_min_periods = min(cfg.stability_window, 12)
        subset["residual_mean"] = subset["residual_log"].rolling(cfg.zscore_window, min_periods=zscore_min_periods).mean()
        subset["residual_std"] = subset["residual_log"].rolling(cfg.zscore_window, min_periods=zscore_min_periods).std(ddof=0)
        subset["z_score"] = (subset["residual_log"] - subset["residual_mean"]) / subset["residual_std"]
        subset["deviation_pct"] = (subset["alt_price"] / subset["fair_value"] - 1.0) * 100.0
        subset["returns"] = subset["alt_price"].pct_change()
        subset["realized_volatility"] = subset["returns"].rolling(cfg.volatility_window, min_periods=vol_min_periods).std(ddof=0) * np.sqrt(24 * 365)
        subset["spread_abs_change"] = subset["residual_log"].diff().abs()
        stability_raw = subset["spread_abs_change"].rolling(cfg.stability_window, min_periods=stability_min_periods).mean()
        subset["spread_stability_score"] = (1.0 / (1.0 + stability_raw * 100.0)).clip(lower=0.0, upper=1.0)
        subset["funding_rate"] = funding.get(symbol)
        records.append(subset)

    if not records:
        return pd.DataFrame()
    return pd.concat(records, ignore_index=True)
