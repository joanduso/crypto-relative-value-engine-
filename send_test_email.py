from __future__ import annotations

from datetime import datetime, timezone

from alerting import email_config_from_env, send_email_alert


def main() -> None:
    config = email_config_from_env()
    if config is None:
        raise SystemExit("SMTP env vars are not configured.")

    sent_at = datetime.now(timezone.utc).isoformat()
    subject = "[Crypto RV] Test email"
    body = (
        "Este es un correo de prueba del monitor de Railway.\n\n"
        f"Enviado en UTC: {sent_at}\n"
        f"Destino: {config.to_email}\n"
    )
    send_email_alert(config, subject, body)
    print(f"Test email sent to {config.to_email}")


if __name__ == "__main__":
    main()
