from __future__ import annotations

from dataclasses import dataclass

from feature_engine import FeatureConfig


@dataclass(frozen=True)
class IntervalProfile:
    interval: str
    limit: int
    poll_minutes: int
    feature_config: FeatureConfig


def interval_to_minutes(interval: str) -> int:
    normalized = interval.strip().lower()
    if normalized.endswith("m"):
        return int(normalized[:-1])
    if normalized.endswith("h"):
        return int(normalized[:-1]) * 60
    if normalized.endswith("d"):
        return int(normalized[:-1]) * 60 * 24
    raise ValueError(f"Unsupported interval: {interval}")


def profile_for_interval(interval: str) -> IntervalProfile:
    if interval == "15m":
        return IntervalProfile(
            interval=interval,
            limit=1500,
            poll_minutes=5,
            feature_config=FeatureConfig(
                regression_window=96 * 7,
                zscore_window=96 * 3,
                stability_window=96,
                volatility_window=96,
            ),
        )

    minutes = interval_to_minutes(interval)
    bars_per_day = max(int((24 * 60) / minutes), 1)

    # Keep the statistical horizons stable across timeframes:
    # regression/z-score ~= 14 days, stability ~= 3 days, volatility ~= 1 day.
    regression_window = max(bars_per_day * 14, 36)
    stability_window = max(bars_per_day * 3, 12)
    volatility_window = max(bars_per_day, 6)

    default_limit = {
        "15m": 1500,
        "1h": 1000,
        "4h": 360,
    }.get(interval, max(regression_window + bars_per_day * 3, int(regression_window * 1.2)))

    poll_minutes = {
        "15m": 5,
        "1h": 5,
        "4h": 15,
    }.get(interval, max(minutes // 3, 5))

    return IntervalProfile(
        interval=interval,
        limit=default_limit,
        poll_minutes=poll_minutes,
        feature_config=FeatureConfig(
            regression_window=regression_window,
            zscore_window=regression_window,
            stability_window=stability_window,
            volatility_window=volatility_window,
        ),
    )
