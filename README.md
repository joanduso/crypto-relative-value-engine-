# Crypto Relative Value Engine

Motor local para analizar oportunidades relative value en Binance y exponer un dashboard en `localhost`.

## Estado actual

El proyecto vive en [crypto_relative_value_engine](C:/Users/Jose.Duran/algoritmo-%20cripto/crypto_relative_value_engine).

Lo que quedó armado:

- Dashboard local con Streamlit en `http://localhost:8501`
- Monitor continuo en segundo plano con `monitor.py`
- Monitor multi-timeframe para `15m`, `1h` y `4h`
- Launcher único para ambos procesos: `start_services.ps1`
- Wrapper para Windows: `start_services.cmd`
- `run_monitor.ps1` y `run_monitor.cmd` redirigidos al launcher nuevo
- Logs y PID files en `output/`

## Archivos clave

```text
crypto_relative_value_engine/
    local_dashboard.py
    main.py
    monitor.py
    start_services.ps1
    start_services.cmd
    run_monitor.ps1
    run_monitor.cmd
    requirements.txt
    output/
```

## Modos

- `COPILOT`: solo análisis. No ejecuta órdenes.
- `AUTO_SAFE`: puede ejecutar con filtros y controles más estrictos.

## Requisitos

- Python 3.12 instalado en:
  `C:\Users\Jose.Duran\AppData\Local\Programs\Python\Python312\python.exe`
- Dependencias de [requirements.txt](C:/Users/Jose.Duran/algoritmo-%20cripto/crypto_relative_value_engine/requirements.txt)

## Cómo correrlo manualmente

Desde la carpeta del proyecto:

```powershell
cd "C:\Users\Jose.Duran\algoritmo- cripto\crypto_relative_value_engine"
python main.py --mode COPILOT
```

Para abrir el dashboard:

```powershell
cd "C:\Users\Jose.Duran\algoritmo- cripto\crypto_relative_value_engine"
python -m streamlit run local_dashboard.py --server.headless true --server.port 8501
```

URL local:

```text
http://localhost:8501
```

## Cómo arrancar dashboard + monitor juntos

Este es el entrypoint recomendado:

```powershell
powershell -ExecutionPolicy Bypass -File .\start_services.ps1
```

Qué hace:

- levanta Streamlit en el puerto `8501` si no está ya arriba
- levanta `monitor.py --mode COPILOT --poll-minutes 5` si no está corriendo
- guarda PIDs en `output\streamlit.pid` y `output\monitor.pid`
- escribe logs en `output\streamlit_stdout.log`, `output\streamlit_stderr.log`, `output\monitor_stdout.log`, `output\monitor_stderr.log`
- intenta abrir el navegador en `http://localhost:8501`

`localhost:8080` ya no forma parte del flujo local por defecto.
Si alguna vez quieres usar el dashboard web/mobile, puedes correrlo manualmente con:

```powershell
python mobile_dashboard.py
```

También puedes usar:

```powershell
.\start_services.cmd
```

## Autoarranque en Windows

### Opción que sí quedó funcionando

Se dejó la carpeta `Startup` como mecanismo de autoarranque del usuario.

Ruta:

```text
C:\Users\Jose.Duran\AppData\Roaming\Microsoft\Windows\Start Menu\Programs\Startup
```

Entrada usada:

```text
crypto_relative_value_monitor.cmd
```

Esa entrada llama a `start_services.ps1`.

### Opción Task Scheduler

Se preparó el archivo [CryptoRelativeValueAutoStart.xml](C:/Users/Jose.Duran/algoritmo-%20cripto/crypto_relative_value_engine/CryptoRelativeValueAutoStart.xml), pero no se pudo registrar desde esta sesión porque Windows devolvió `Acceso denegado`.

Si en algún momento puedes abrir una terminal con privilegios adecuados, puedes crear la tarea con:

```powershell
schtasks /Create /TN "CryptoRelativeValueAutoStart" /SC ONLOGON /TR "powershell -ExecutionPolicy Bypass -File \"C:\Users\Jose.Duran\algoritmo- cripto\crypto_relative_value_engine\start_services.ps1\"" /RL LIMITED /F
```

## Qué se corrigió

- Se detectó que el autoarranque original solo levantaba `monitor.py`
- Se detectó que Streamlit no estaba instalado en el entorno usado por el proyecto
- Se instalaron las dependencias para Python 3.12
- Se unificó el arranque en un solo script
- Se dejó verificado que el dashboard escucha en `localhost:8501`

## Comandos útiles

Ver si el dashboard está escuchando:

```powershell
netstat -ano | Select-String ":8501"
```

Ver logs de Streamlit:

```powershell
Get-Content .\output\streamlit_stdout.log
Get-Content .\output\streamlit_stderr.log
```

Ver logs del monitor:

```powershell
Get-Content .\output\monitor_stdout.log
Get-Content .\output\monitor_stderr.log
```

Ver PIDs guardados:

```powershell
Get-Content .\output\streamlit.pid
Get-Content .\output\monitor.pid
```

## Credenciales opcionales

Solo si quieres live trading:

- `BINANCE_API_KEY`
- `BINANCE_API_SECRET`

Solo si quieres alertas por email:

- `SMTP_HOST`
- `SMTP_PORT`
- `SMTP_USER`
- `SMTP_PASSWORD`
- `ALERT_FROM_EMAIL`
- `ALERT_TO_EMAIL`
- `SMTP_USE_TLS=true`

## Noticias y score

El motor puede ajustar el `market_opportunity_score` con eventos de noticias leidos desde `news_events.csv`.

Columnas esperadas:

```text
timestamp,symbol,market_scope,event_type,sentiment,source_tier,severity,confidence,headline,url
```

- `symbol`: por ejemplo `XRPUSDT`, `SOLUSDT` o `ALL`
- `market_scope`: por ejemplo `MACRO`
- `sentiment`: `BULLISH` o `BEARISH`
- `source_tier`: `OFFICIAL`, `AGGREGATOR`, `MEDIA` o `SOCIAL`
- `severity`: entre `0` y `1`
- `confidence`: entre `0` y `1`

El ajuste se calcula en `news_engine.py` y hoy se limita a `+/-15` puntos sobre el score base.

### Colector automatico

Se agrego `news_collector.py` para poblar `news_events.csv` desde RSS y un evento macro de Fear & Greed.

Ejemplo:

```powershell
cd "C:\Users\Jose.Duran\algoritmo- cripto\crypto_relative_value_engine"
python news_collector.py --include-fear-greed
```

Variables utiles:

- `NEWS_RSS_FEEDS`
- `NEWS_LOOKBACK_HOURS`
- `NEWS_MAX_ITEMS_PER_FEED`
- `NEWS_SYMBOLS`
- `NEWS_INCLUDE_FEAR_GREED`
- `NEWS_EVENTS_PATH`

Ejemplo de feeds:

```text
https://www.coindesk.com/arc/outboundfeeds/rss/,https://cointelegraph.com/rss
```

## BTC bias phase 2

Se agrego una capa mas completa de sesgo direccional de BTC con:

- tecnica
- derivados
- macro
- ETF flows
- on-chain
- noticias y sentimiento

Archivos clave:

- `btc_market_bias_engine.py`
- `regime_engine.py`
- `backtest.py`

Variables opcionales para enriquecer el bias:

- `BTC_ETF_FLOWS_PATH`
- `BTC_ETF_REMOTE_ENABLED=true`
- `BTC_MVRV_PATH`
- `BTC_SOPR_PATH`
- `GLASSNODE_API_KEY`
- `GLASSNODE_MVRV_PATH`
- `GLASSNODE_SOPR_PATH`

Formato esperado para CSVs externos:

```text
timestamp,value
2026-03-01T00:00:00Z,123.4
```

Ejemplos:

- `btc_etf_flows.csv`: flujo neto diario en millones de USD
- `btc_mvrv.csv`: serie diaria de MVRV
- `btc_sopr.csv`: serie diaria de SOPR

El backtest ahora compara:

- `base`
- `regime_overlay`
- `btc_bias_overlay`
- `combined_overlay`

### Collector rapido

Se agrego `btc_bias_data_collector.py` para poblar esos CSVs.

Ejemplo:

```powershell
cd "C:\Users\Jose.Duran\algoritmo- cripto\crypto_relative_value_engine"
py -3 btc_bias_data_collector.py
```

Si quieres solo ETF flows:

```powershell
py -3 btc_bias_data_collector.py --skip-mvrv --skip-sopr
```

Si tienes Glassnode:

```powershell
$env:GLASSNODE_API_KEY="tu_api_key"
py -3 btc_bias_data_collector.py
```

### Cadencia recomendada en monitor

Para no meter ruido ni gasto inutil de llamadas:

- alertas de trading: segun el intervalo del monitor (`15m` cada `5` min, `1h` cada `5` min, `4h` cada `15` min)
- noticias: cada `20` a `30` minutos
- ETF flows y on-chain: cada `180` minutos

Variables del monitor:

- `NEWS_COLLECTION_ENABLED=true`
- `NEWS_POLL_MINUTES=30`
- `BTC_BIAS_DATA_COLLECTION_ENABLED=true`
- `BTC_BIAS_DATA_POLL_MINUTES=180`

La razon es simple:

- las noticias pueden cambiar el sesgo intradia
- ETF flows cambian por dia y no necesitan polling agresivo
- MVRV y SOPR son datos lentos; consultarlos cada `3h` es suficiente para alerts operativas

## Metodologia de timeframes

El monitor de Railway puede correr varios intervalos en paralelo usando `MONITOR_INTERVALS`.

Configuracion recomendada:

- `15m`: `1500` velas, polling cada `5` minutos
- `1h`: `1000` velas, polling cada `5` minutos
- `4h`: `360` velas, polling cada `15` minutos

Criterio:

- la regresion y el z-score usan aproximadamente `14` dias de barras
- la estabilidad usa aproximadamente `3` dias de barras
- la volatilidad usa aproximadamente `1` dia de barras
- eso mantiene comparable la logica estadistica entre `15m`, `1h` y `4h`

## Recomendación de arquitectura

Si no puedes usar PowerShell como administrador, estas son las opciones sensatas:

### Mejor opción para este proyecto

Mantener el motor y el monitor corriendo en tu PC local con `Startup` o con un acceso directo al launcher.

Por qué:

- este proyecto depende de procesos persistentes
- consume APIs y puede escribir archivos locales
- Streamlit está pensado para correr como app viva, no como función serverless

### GitHub sí, pero para código y automatización limitada

GitHub sirve para:

- guardar versión del proyecto
- tener backup
- disparar tests o tareas batch con GitHub Actions

GitHub no sirve bien para:

- mantener abierto un `localhost`
- tener un monitor persistente 24/7 para uso personal local

### Vercel no es buena opción para este motor completo

Vercel puede servir si separas una parte web liviana, pero no para este motor tal como está.

Problemas:

- Vercel es serverless
- no está pensado para procesos persistentes
- `monitor.py` no encaja bien ahí
- Streamlit no es el deploy natural en Vercel
- el acceso a archivos y estado local es limitado

### Si quieres algo más estable sin admin

Alternativas reales:

- dejar `Startup` como está ahora
- usar un acceso directo al `.cmd` en el escritorio
- usar `pythonw` o un `.cmd` silencioso para que no moleste una consola
- subir el repo a GitHub solo para versionado
- si luego quieres nube real, mover esto a un VPS o Railway/Render, no a Vercel

## Siguiente paso recomendado

Si quieres seguir sin permisos de administrador:

1. deja `Startup` como mecanismo de autoarranque
2. sube el proyecto a GitHub para backup
3. si quieres acceso remoto o 24/7, migramos el monitor a un VPS o servicio tipo Render/Railway

Para uso local diario, hoy la ruta más pragmática es:

```text
PC local + Startup + Streamlit en localhost:8501
```
