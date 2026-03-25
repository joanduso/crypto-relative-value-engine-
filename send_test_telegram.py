from __future__ import annotations

from datetime import datetime, timezone

from alerting import send_telegram_alert, telegram_config_from_env


def main() -> None:
    config = telegram_config_from_env()
    if config is None:
        raise SystemExit("TELEGRAM_BOT_TOKEN o TELEGRAM_CHAT_ID no estan configurados.")

    sent_at = datetime.now(timezone.utc).isoformat()
    message = (
        "[Crypto RV] Test Telegram\n\n"
        f"Enviado en UTC: {sent_at}\n"
        f"Chat ID: {config.chat_id}"
    )
    send_telegram_alert(config, message)
    print(f"Test telegram sent to chat {config.chat_id}")


if __name__ == "__main__":
    main()
