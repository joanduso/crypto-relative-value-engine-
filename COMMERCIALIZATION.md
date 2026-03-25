# Commercialization Roadmap

## Product thesis

El producto no debe venderse como "otro bot de crypto".
La mejor posicion es:

```text
Relative value intelligence engine for crypto:
multi-timeframe scoring, market regime filtering, news-aware ranking,
portfolio selection and operator-friendly dashboards.
```

## Product lines

### 1. Signals terminal

Ideal para traders discrecionales y semi-sistematicos.

Incluye:

- ranking por `market_opportunity_score`
- overlay de regimen
- `news_impact_score`
- alertas por Telegram
- dashboard web/mobile

Monetizacion:

- suscripcion mensual
- planes por cantidad de activos y alertas

### 2. Research Pro

Ideal para desks pequenos o power users.

Incluye:

- backtests comparativos
- calibracion de win rate
- export CSV/API
- reportes de drawdown, hit-rate y estabilidad

Monetizacion:

- plan premium
- licencias por asiento

### 3. Execution Assist

Ideal para usuarios avanzados.

Incluye:

- paper trading
- propuestas de posicion
- validaciones de riesgo
- modo live con aprobacion explicita

Monetizacion:

- add-on premium
- fee mensual alto, no fee por volumen al inicio

## Minimum viable SaaS architecture

### Phase 1

- collector de mercado
- collector de noticias
- engine de scoring
- API Flask/FastAPI
- dashboard web
- Postgres para snapshots y eventos

### Phase 2

- auth y cuentas
- workspaces por cliente
- colas para jobs
- scheduler
- auditoria y logs

### Phase 3

- billing
- API publica
- conectores de exchange
- reportes para clientes

## What must be true before selling

- secretos fuera del repo
- resultados historicos reproducibles
- metricas por estrategia
- disclaimers claros
- paper trading estable
- onboarding simple

## 30-day execution plan

1. Limpiar repo y dejar release interna estable.
2. Mover snapshots y eventos a base de datos.
3. Automatizar `news_collector.py`.
4. Crear endpoints API para oportunidades, señales y noticias.
5. Añadir auth basica.
6. Preparar landing + pricing + demo data.
7. Cerrar piloto con 3 a 5 usuarios.

## Technical blueprint

La separacion recomendada entre consola interna y producto web esta en:

- `WEB_PRODUCT_ARCHITECTURE.md`
