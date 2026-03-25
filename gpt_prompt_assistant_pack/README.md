# GPT Prompt Assistant Pack

Este paquete esta pensado para crear un GPT en ChatGPT que:

- te ayude a convertir ideas vagas en prompts claros
- te prepare prompts para Codex, ChatGPT, imagenes, automatizaciones y analisis
- mantenga contexto de trading/cripto cuando se lo pidas
- te devuelva respuestas en formatos utiles y consistentes

## Donde crear el GPT

Segun OpenAI, hoy se crea en la web:

- `https://chatgpt.com/gpts/editor`
- o `https://chatgpt.com/gpts` y luego `+ Create`

Referencias oficiales:

- `Creating a GPT`: https://help.openai.com/en/articles/8554397-create-a-gpt
- `Building and publishing a GPT`: https://help.openai.com/en/articles/8798878-building-and-publishing-a-gpt

## Que pegar en cada campo

- `Name`: usa el sugerido en `name.txt`
- `Description`: pega `description.txt`
- `Instructions`: pega `instructions.md`
- `Conversation starters`: usa `conversation_starters.txt`

## Que archivos cargar en Knowledge

Carga estos 3 archivos:

- `knowledge/prompt_framework.md`
- `knowledge/trading_context.md`
- `knowledge/response_formats.md`

No hace falta cargar `README.md`, `name.txt`, `description.txt` ni `conversation_starters.txt`.

## Capabilities recomendadas

Activa:

- `Web Search`
- `Code Interpreter & Data Analysis`

Activa `Image Generation` solo si quieres que tambien te ayude a crear prompts visuales.

No te recomiendo empezar con `Custom Actions` todavia. Primero haz que el GPT piense bien y te estructure prompts. Despues, si quieres, le conectamos APIs.

## Configuracion recomendada

- Visibilidad inicial: `Private`
- Usa una imagen simple y limpia
- Si luego lo publicas, revisa Builder Profile y verificacion

## Como usarlo bien

Ejemplos:

- `Quiero un prompt para que Codex revise por que una alerta de BNB salio con entry viejo`
- `Convierte esta idea en prompt para ChatGPT con contexto, restricciones y formato de salida`
- `Dame un prompt para analizar BTCUSDT en 4h, 1h y 15m con sesgo direccional`
- `Hazme 3 versiones del prompt: corta, robusta y experta`

## Siguiente mejora sensata

Despues de probarlo, la mejora natural es una version 2 con:

- memoria de tus preferencias de estilo
- plantillas por tipo de tarea
- actions para leer tu dashboard o tus CSV
