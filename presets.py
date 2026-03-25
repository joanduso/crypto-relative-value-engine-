from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class PresetDefinition:
    name: str
    description: str
    env: dict[str, str]


PRESETS: dict[str, PresetDefinition] = {
    "DEFAULT": PresetDefinition(
        name="DEFAULT",
        description="Usa los thresholds base del modo elegido.",
        env={},
    ),
    "INTRADAY_AGGRESSIVE": PresetDefinition(
        name="INTRADAY_AGGRESSIVE",
        description="Mas entradas, menor z-score y menor liquidez minima. Mejor para exploracion intradia.",
        env={
            "ENGINE_COPILOT_RELAXED_ZSCORE_ENTRY": "0.80",
            "ENGINE_COPILOT_RELAXED_MIN_QUOTE_VOLUME": "250000",
            "ENGINE_COPILOT_RELAXED_MIN_SPREAD_STABILITY": "0.30",
            "ENGINE_COPILOT_RELAXED_TOP_N": "10",
            "RISK_COPILOT_RELAXED_MAX_CONCURRENT_POSITIONS": "3",
        },
    ),
    "SWING_MODERATE": PresetDefinition(
        name="SWING_MODERATE",
        description="Filtro intermedio para 1h/4h con algo mas de paciencia y riesgo controlado.",
        env={
            "ENGINE_COPILOT_RELAXED_ZSCORE_ENTRY": "1.00",
            "ENGINE_COPILOT_RELAXED_MIN_QUOTE_VOLUME": "750000",
            "ENGINE_COPILOT_RELAXED_MIN_SPREAD_STABILITY": "0.40",
            "RISK_COPILOT_RELAXED_STOP_LOSS_PCT": "0.03",
            "RISK_COPILOT_RELAXED_TAKE_PROFIT_PCT": "0.06",
        },
    ),
    "HIGH_LIQUIDITY_ONLY": PresetDefinition(
        name="HIGH_LIQUIDITY_ONLY",
        description="Prioriza monedas liquidas y mantiene thresholds relativamente estrictos.",
        env={
            "ENGINE_COPILOT_RELAXED_ZSCORE_ENTRY": "1.10",
            "ENGINE_COPILOT_RELAXED_MIN_QUOTE_VOLUME": "5000000",
            "ENGINE_COPILOT_RELAXED_MIN_SPREAD_STABILITY": "0.40",
            "ENGINE_COPILOT_RELAXED_TOP_N": "6",
            "RISK_COPILOT_RELAXED_MAX_CONCURRENT_POSITIONS": "2",
        },
    ),
}


def preset_names() -> list[str]:
    return list(PRESETS.keys())


def preset_description(name: str) -> str:
    preset = PRESETS.get(name, PRESETS["DEFAULT"])
    return preset.description


def apply_preset(name: str) -> dict[str, str | None]:
    preset = PRESETS.get(name, PRESETS["DEFAULT"])
    previous: dict[str, str | None] = {}
    for key, value in preset.env.items():
        previous[key] = os.environ.get(key)
        os.environ[key] = value
    return previous


def restore_preset(previous: dict[str, str | None]) -> None:
    for key, old_value in previous.items():
        if old_value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = old_value
