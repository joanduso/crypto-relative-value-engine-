from __future__ import annotations

import os
import smtplib
from dataclasses import dataclass
from email.message import EmailMessage
from pathlib import Path
from typing import Iterable

import pandas as pd


@dataclass(frozen=True)
class EmailConfig:
    smtp_host: str
    smtp_port: int
    smtp_user: str
    smtp_password: str
    from_email: str
    to_email: str
    use_tls: bool = True


def email_config_from_env() -> EmailConfig | None:
    host = os.getenv("SMTP_HOST")
    port = os.getenv("SMTP_PORT")
    user = os.getenv("SMTP_USER")
    password = os.getenv("SMTP_PASSWORD")
    from_email = os.getenv("ALERT_FROM_EMAIL")
    to_email = os.getenv("ALERT_TO_EMAIL")
    if not all([host, port, user, password, from_email, to_email]):
        return None
    return EmailConfig(
        smtp_host=host,
        smtp_port=int(port),
        smtp_user=user,
        smtp_password=password,
        from_email=from_email,
        to_email=to_email,
        use_tls=os.getenv("SMTP_USE_TLS", "true").lower() != "false",
    )


def select_alert_candidates(
    ranked_universe: pd.DataFrame,
    min_quality: tuple[str, ...] = ("A1", "A2"),
    min_confidence: float = 75.0,
) -> pd.DataFrame:
    if ranked_universe.empty:
        return ranked_universe.copy()
    candidates = ranked_universe.copy()
    return candidates.loc[
        candidates["passes_filters"]
        & candidates["signal_quality"].isin(min_quality)
        & (
            (candidates["confidence_score"] >= min_confidence)
            | (candidates["market_opportunity_score"] >= 76.0)
        )
    ].copy()


def build_alert_key(row: pd.Series) -> str:
    return f"{row['symbol']}|{row['suggested_direction']}|{row['signal_quality']}"


def load_sent_alerts(path: str | Path) -> dict[str, pd.Timestamp]:
    state_path = Path(path)
    if not state_path.exists():
        return {}

    sent_alerts: dict[str, pd.Timestamp] = {}
    for raw_line in state_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if "\t" in line:
            key, sent_at = line.split("\t", 1)
        else:
            key, sent_at = line, ""
        timestamp = pd.to_datetime(sent_at, utc=True, errors="coerce")
        if pd.isna(timestamp):
            timestamp = pd.Timestamp.min.tz_localize("UTC")
        sent_alerts[key] = timestamp
    return sent_alerts


def save_sent_alerts(path: str | Path, sent_alerts: dict[str, pd.Timestamp]) -> None:
    state_path = Path(path)
    state_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"{key}\t{timestamp.isoformat()}"
        for key, timestamp in sorted(sent_alerts.items(), key=lambda item: item[0])
    ]
    state_path.write_text("\n".join(lines), encoding="utf-8")


def filter_new_alerts(
    alerts_df: pd.DataFrame,
    sent_alerts: dict[str, pd.Timestamp],
    cooldown_minutes: int,
    now_utc: pd.Timestamp | None = None,
) -> pd.DataFrame:
    if alerts_df.empty:
        return alerts_df.copy()

    reference_now = now_utc or pd.Timestamp.utcnow()
    if reference_now.tzinfo is None:
        reference_now = reference_now.tz_localize("UTC")

    cooldown = pd.Timedelta(minutes=max(cooldown_minutes, 0))
    fresh_rows: list[pd.Series] = []
    for _, row in alerts_df.iterrows():
        alert_key = build_alert_key(row)
        last_sent = sent_alerts.get(alert_key)
        if last_sent is None or (reference_now - last_sent) >= cooldown:
            fresh_rows.append(row)

    if not fresh_rows:
        return alerts_df.iloc[0:0].copy()
    return pd.DataFrame(fresh_rows).reset_index(drop=True)


def update_sent_alerts(
    sent_alerts: dict[str, pd.Timestamp],
    alerts_df: pd.DataFrame,
    sent_at: pd.Timestamp | None = None,
) -> dict[str, pd.Timestamp]:
    if alerts_df.empty:
        return sent_alerts

    timestamp = sent_at or pd.Timestamp.utcnow()
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("UTC")

    for _, row in alerts_df.iterrows():
        sent_alerts[build_alert_key(row)] = timestamp
    return sent_alerts


def parse_quality_list(raw_value: str | Iterable[str]) -> tuple[str, ...]:
    if isinstance(raw_value, str):
        parts = raw_value.split(",")
    else:
        parts = list(raw_value)
    cleaned = tuple(part.strip().upper() for part in parts if str(part).strip())
    return cleaned or ("A1", "A2")


def format_alert_email(mode_name: str, alerts_df: pd.DataFrame) -> tuple[str, str]:
    subject = f"[Crypto RV] {len(alerts_df)} senales nuevas en {mode_name}"
    lines = [f"Se detectaron {len(alerts_df)} senales nuevas:", ""]
    for row in alerts_df.itertuples(index=False):
        lines.append(
            (
                f"{row.symbol} | {row.suggested_direction} | {row.signal_quality} | "
                f"precio={row.current_price:.4f} | fair={row.expected_fair_value:.4f} | "
                f"entrada={row.suggested_entry:.4f} | stop={row.suggested_stop_loss:.4f} | take={row.suggested_take_profit:.4f} | "
                f"desvio={row.deviation_pct:.2f}% | z={row.z_score:.2f} | "
                f"score={row.confidence_score:.1f}"
            )
        )
    return subject, "\n".join(lines)


def append_daily_alert_snapshot(path: str | Path, ranked_universe: pd.DataFrame) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if ranked_universe.empty:
        if not output_path.exists():
            pd.DataFrame().to_csv(output_path, index=False)
        return output_path

    snapshot = ranked_universe.copy()
    snapshot["snapshot_date"] = pd.Timestamp.utcnow().date().isoformat()
    snapshot["snapshot_time_utc"] = pd.Timestamp.utcnow().isoformat()

    if output_path.exists():
        existing = pd.read_csv(output_path)
        combined = pd.concat([existing, snapshot], ignore_index=True)
        combined = combined.drop_duplicates(subset=["snapshot_time_utc", "symbol", "suggested_direction"], keep="last")
    else:
        combined = snapshot
    combined.to_csv(output_path, index=False)
    return output_path


def daily_best_alerts(history_df: pd.DataFrame) -> pd.DataFrame:
    if history_df.empty:
        return history_df
    ordered = history_df.sort_values(
        ["snapshot_date", "symbol", "confidence_score", "edge_after_fees_pct"],
        ascending=[True, True, False, False],
    )
    return ordered.groupby(["snapshot_date", "symbol"], as_index=False).head(1).reset_index(drop=True)


def send_email_alert(config: EmailConfig, subject: str, body: str) -> None:
    message = EmailMessage()
    message["Subject"] = subject
    message["From"] = config.from_email
    message["To"] = config.to_email
    message.set_content(body)

    with smtplib.SMTP(config.smtp_host, config.smtp_port, timeout=30) as smtp:
        if config.use_tls:
            smtp.starttls()
        smtp.login(config.smtp_user, config.smtp_password)
        smtp.send_message(message)
