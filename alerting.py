from __future__ import annotations

import logging
import os
import socket
import smtplib
import ssl
import time
from dataclasses import dataclass
from email.message import EmailMessage
from pathlib import Path
from typing import Iterable

import pandas as pd
import requests


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class EmailConfig:
    resend_api_key: str | None
    smtp_host: str
    smtp_port: int
    smtp_user: str
    smtp_password: str
    from_email: str
    to_email: str
    use_tls: bool = True


class IPv4FirstSMTP(smtplib.SMTP):
    def _get_socket(self, host: str, port: int, timeout: float):
        return _create_socket(host, port, timeout)


class IPv4FirstSMTP_SSL(smtplib.SMTP_SSL):
    def _get_socket(self, host: str, port: int, timeout: float):
        sock = _create_socket(host, port, timeout)
        return self.context.wrap_socket(sock, server_hostname=host)


def email_config_from_env() -> EmailConfig | None:
    resend_api_key = os.getenv("RESEND_API_KEY")
    host = os.getenv("SMTP_HOST")
    port = os.getenv("SMTP_PORT")
    user = os.getenv("SMTP_USER")
    password = os.getenv("SMTP_PASSWORD")
    from_email = os.getenv("ALERT_FROM_EMAIL")
    to_email = os.getenv("ALERT_TO_EMAIL")
    if resend_api_key and from_email and to_email:
        return EmailConfig(
            resend_api_key=resend_api_key,
            smtp_host=host or "",
            smtp_port=int(port or 587),
            smtp_user=user or "",
            smtp_password=password or "",
            from_email=from_email,
            to_email=to_email,
            use_tls=os.getenv("SMTP_USE_TLS", "true").lower() != "false",
        )

    if not all([host, port, user, password, from_email, to_email]):
        return None
    return EmailConfig(
        resend_api_key=None,
        smtp_host=host,
        smtp_port=int(port),
        smtp_user=user,
        smtp_password=password,
        from_email=from_email,
        to_email=to_email,
        use_tls=os.getenv("SMTP_USE_TLS", "true").lower() != "false",
    )


def _create_socket(host: str, port: int, timeout: float) -> socket.socket:
    last_error: OSError | None = None
    addresses = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
    addresses = sorted(addresses, key=lambda item: 0 if item[0] == socket.AF_INET else 1)

    for family, socktype, proto, _, sockaddr in addresses:
        sock = socket.socket(family, socktype, proto)
        sock.settimeout(timeout)
        try:
            sock.connect(sockaddr)
            return sock
        except OSError as exc:
            last_error = exc
            sock.close()

    if last_error is not None:
        raise last_error
    raise OSError(f"Unable to resolve SMTP host {host}:{port}")


def _smtp_attempts(config: EmailConfig) -> list[tuple[str, int, bool]]:
    attempts = [(config.smtp_host, config.smtp_port, config.use_tls)]
    if config.smtp_host == "smtp.gmail.com" and (config.smtp_port != 465 or config.use_tls):
        attempts.append((config.smtp_host, 465, False))
    return attempts


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
        timeframe = f" | tf={row.timeframe}" if hasattr(row, "timeframe") else ""
        stop_loss = f"{row.suggested_stop_loss:.4f}" if hasattr(row, "suggested_stop_loss") else "n/a"
        take_profit = f"{row.suggested_take_profit:.4f}" if hasattr(row, "suggested_take_profit") else "n/a"
        lines.append(
            (
                f"{row.symbol} | {row.suggested_direction} | {row.signal_quality}{timeframe} | "
                f"precio={row.current_price:.4f} | fair={row.expected_fair_value:.4f} | "
                f"entrada={row.suggested_entry:.4f} | stop={stop_loss} | take={take_profit} | "
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
    if config.resend_api_key:
        response = requests.post(
            "https://api.resend.com/emails",
            headers={
                "Authorization": f"Bearer {config.resend_api_key}",
                "Content-Type": "application/json",
            },
            json={
                "from": config.from_email,
                "to": [config.to_email],
                "subject": subject,
                "text": body,
            },
            timeout=30,
        )
        response.raise_for_status()
        return

    message = EmailMessage()
    message["Subject"] = subject
    message["From"] = config.from_email
    message["To"] = config.to_email
    message.set_content(body)

    last_error: Exception | None = None
    for attempt_number, (host, port, use_starttls) in enumerate(_smtp_attempts(config), start=1):
        try:
            context = ssl.create_default_context()
            if use_starttls:
                with IPv4FirstSMTP(host=host, port=port, timeout=30) as smtp:
                    smtp.starttls(context=context)
                    smtp.login(config.smtp_user, config.smtp_password)
                    smtp.send_message(message)
                    return
            with IPv4FirstSMTP_SSL(host=host, port=port, timeout=30, context=context) as smtp:
                smtp.login(config.smtp_user, config.smtp_password)
                smtp.send_message(message)
                return
        except Exception as exc:
            last_error = exc
            logger.warning(
                "smtp send attempt failed",
                extra={"host": host, "port": port, "attempt": attempt_number, "error": str(exc)},
            )
            time.sleep(min(attempt_number, 3))

    if last_error is not None:
        raise last_error
