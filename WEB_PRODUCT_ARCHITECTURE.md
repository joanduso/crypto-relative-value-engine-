# Web Product Architecture

## Goal

Separar la consola operativa local del producto comercial web.

- `localhost:8501` queda como cockpit interno para operador.
- la web comercial consume una API estable y no depende de Streamlit.

## Product split

### Internal operator console

Base:

- `local_dashboard.py`

Responsabilidad:

- ejecucion manual
- tuning
- debugging
- presets
- validacion de score
- revision profunda de tablas y backtests

### Commercial web app

Base recomendada:

- Flask/FastAPI API
- frontend web ligero primero

Responsabilidad:

- login
- watchlists
- top opportunities
- BTC/ETH trend
- symbol pages
- news by symbol
- recent alerts
- mobile-first UX

## API design

### Public/private boundary

No exponer directamente archivos CSV ni logica interna de Streamlit.
La web debe consumir endpoints definidos.

### Recommended endpoints

#### Health

```text
GET /api/health
```

Response:

```json
{
  "status": "ok",
  "engine_mode": "COPILOT",
  "latest_run_at": "2026-03-25T18:00:00Z"
}
```

#### Market regime

```text
GET /api/market/regime
```

Response:

```json
{
  "btc": {
    "symbol": "BTCUSDT",
    "trend": "ALCISTA",
    "price_vs_ema200_pct": 3.42
  },
  "eth": {
    "symbol": "ETHUSDT",
    "trend": "BAJISTA",
    "price_vs_ema200_pct": -1.18
  }
}
```

#### Opportunities

```text
GET /api/opportunities
```

Query params:

- `direction`
- `timeframe`
- `quality`
- `execution_status`
- `min_score`
- `min_confidence`
- `limit`

Response item:

```json
{
  "timestamp": "2026-03-25T18:00:00Z",
  "symbol": "SOLUSDT",
  "direction": "LONG",
  "timeframe": "1h",
  "market_opportunity_score": 84.6,
  "signal_quality": "A2",
  "confidence_score": 79.1,
  "news_impact_score": 4.2,
  "news_bias": "BULLISH",
  "news_comment": "Sesgo positivo moderado por flujo de noticias.",
  "execution_status": "OK"
}
```

#### Opportunity detail

```text
GET /api/opportunities/{symbol}
```

Response:

- ultima oportunidad del symbol
- breakdown de score
- filtros
- noticias relacionadas

#### News feed

```text
GET /api/news
```

Query params:

- `symbol`
- `sentiment`
- `source_tier`
- `limit`

Response item:

```json
{
  "timestamp": "2026-03-25T16:00:00Z",
  "symbol": "BNBUSDT",
  "market_scope": "",
  "sentiment": "BULLISH",
  "source_tier": "MEDIA",
  "severity": 0.7,
  "confidence": 0.65,
  "headline": "Example title",
  "url": "https://example.com/article"
}
```

#### Symbol news

```text
GET /api/symbols/{symbol}/news
```

Devuelve:

- noticias directas del symbol
- noticias macro `ALL/MACRO`
- conteo total

#### Alerts

```text
GET /api/alerts/recent
```

Devuelve:

- alertas recientes enviadas
- symbol
- quality
- execution status
- timestamp

#### History

```text
GET /api/history
```

Devuelve snapshots historicos filtrables por symbol/timeframe.

## Database design

### Core tables

#### engine_runs

Una corrida del motor.

Columns:

- `id`
- `created_at`
- `mode`
- `preset`
- `interval`
- `limit_bars`
- `status`
- `notes`

#### market_regimes

Columns:

- `id`
- `engine_run_id`
- `timestamp`
- `btc_trend`
- `btc_price_vs_ema200_pct`
- `btc_ema50_vs_ema200_pct`
- `btc_directional_score`
- `eth_trend`
- `eth_price_vs_ema200_pct`

#### opportunities

La tabla principal para web.

Columns:

- `id`
- `engine_run_id`
- `timestamp`
- `symbol`
- `timeframe`
- `direction`
- `current_price`
- `expected_fair_value`
- `deviation_pct`
- `z_score`
- `edge_after_fees_pct`
- `confidence_score`
- `pre_news_market_score`
- `news_impact_score`
- `market_opportunity_score`
- `signal_quality`
- `execution_status`
- `passes_filters`
- `news_bias`
- `news_comment`

#### opportunity_metrics

Opcional para desglose más granular.

Columns:

- `opportunity_id`
- `confidence_rank_pct`
- `edge_rank_pct`
- `z_rank_pct`
- `stability_rank_pct`
- `liquidity_rank_pct`
- `vol_rank_pct`
- `regime_score_penalty`

#### news_events

Columns:

- `id`
- `timestamp`
- `symbol`
- `market_scope`
- `event_type`
- `sentiment`
- `source_tier`
- `severity`
- `confidence`
- `headline`
- `url`
- `dedupe_key`

#### opportunity_news_links

Relaciona noticias con oportunidades mostradas.

Columns:

- `opportunity_id`
- `news_event_id`
- `relevance_score`
- `impact_points`

#### strategy_signals

Columns:

- `id`
- `engine_run_id`
- `timestamp`
- `symbol`
- `strategy`
- `signal`
- `risk_decision`
- `approved_position_size`
- `approved_leverage`
- `allocated_capital`

#### users

Columns:

- `id`
- `email`
- `password_hash`
- `plan`
- `created_at`
- `status`

#### watchlists

Columns:

- `id`
- `user_id`
- `name`

#### watchlist_symbols

Columns:

- `watchlist_id`
- `symbol`

## Migration from CSV

### Current CSV sources

- `output/monitor_latest*.csv`
- `output/daily_alert_history*.csv`
- `output/strategy_signals_latest*.csv`
- `news_events.csv`

### Migration phases

#### Phase 1

Mantener CSV y agregar escritura dual a DB.

- monitor sigue escribiendo CSV
- monitor tambien inserta en Postgres
- la web empieza a leer Postgres

#### Phase 2

La API deja de depender de CSV.

- API lee solo Postgres
- dashboards web consumen API

#### Phase 3

CSV solo para export/debug.

## Suggested first implementation

### Backend

- FastAPI
- SQLAlchemy
- Postgres
- Alembic

### Why

- contratos claros
- docs automaticas
- auth simple
- escalado limpio

## Minimum SaaS rollout

1. Crear Postgres.
2. Crear tablas `engine_runs`, `market_regimes`, `opportunities`, `news_events`.
3. Hacer que `monitor.py` escriba tambien en DB.
4. Crear `GET /api/opportunities`.
5. Crear `GET /api/symbols/{symbol}/news`.
6. Crear login simple.
7. Montar frontend comercial.

## What to keep out of v1

- live execution para clientes
- demasiadas configuraciones
- backtests pesados en la web
- multi-tenant complejo de entrada

## Best immediate path

- mantener `8501` para uso interno
- construir API + web por encima del motor actual
- usar Postgres antes de abrir el producto
