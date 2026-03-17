# Railway Web Dashboard

Dashboard web responsive para seguir oportunidades del engine desde el celular.

## Archivos

- `mobile_dashboard.py`
- `templates/mobile_dashboard.html`
- `static/mobile_dashboard.css`
- `Procfile.web`

## Que muestra

- Resumen del estado actual
- Top oportunidades
- Historico reciente
- Strategy signals
- Filtros por direccion, score, confidence, calidad y aprobadas

## Datos usados

Lee estos archivos de `output/`:

- `monitor_latest.csv`
- `daily_alert_history.csv`
- `strategy_signals_latest.csv`

## Ejecutar local

```powershell
cd "C:\Users\Jose.Duran\algoritmo- cripto\crypto_relative_value_engine"
python mobile_dashboard.py
```

Abre:

```text
http://localhost:8080
```

## Endpoints

- `/`
- `/health`
- `/api/summary`
- `/api/opportunities`
- `/api/history`
- `/api/signals`

## Railway

No necesitas dominio para empezar. Railway te da una URL publica gratis del servicio.

Arquitectura recomendada:

1. Servicio `web` unico:
   `python mobile_dashboard.py`
2. Variable:
   `EMBED_MONITOR_IN_WEB=true`

Con eso la web arranca y lanza `monitor.py` dentro del mismo servicio para que ambos compartan los CSV locales.

## Start Command para el servicio web

```text
python mobile_dashboard.py
```

## Variable recomendada

```text
EMBED_MONITOR_IN_WEB=true
```

## Dominio propio

Solo hace falta si quieres una URL personalizada como `app.tudominio.com`.

Si no, puedes lanzar primero con la URL nativa de Railway y conectar dominio despues.
