# Prompt Framework

Usa este marco para convertir una necesidad en un prompt ejecutable.

## Estructura base

### 1. Contexto

Que sistema, archivo, mercado, producto o situacion existe.

### 2. Objetivo

Que resultado debe producir el modelo.

### 3. Restricciones

Que no debe hacer, limites tecnicos, estilo, herramientas permitidas, formatos, plazos.

### 4. Inputs conocidos

Datos ya disponibles.

### 5. Inputs faltantes

Solo preguntar lo critico.

### 6. Formato de salida

Como debe responder.

### 7. Criterio de calidad

Como saber si la respuesta sirve.

## Plantilla generica

Contexto:
[describe el entorno]

Objetivo:
[describe el resultado exacto]

Restricciones:
[limites, estilo, herramientas, exclusiones]

Inputs disponibles:
[datos ya dados]

Formato de salida:
[lista, tabla, codigo, pasos, diagnostico, etc.]

Criterio de calidad:
[que debe quedar claro o resuelto]

## Plantilla para debugging

Contexto:
Estoy trabajando en [proyecto/sistema]. El problema ocurre en [archivo/modulo/flujo].

Objetivo:
Identificar la causa raiz y proponer el fix mas probable.

Restricciones:
No asumir informacion no dada. Priorizar causas verificables. No reescribir partes no relacionadas.

Datos disponibles:
- error observado:
- comportamiento esperado:
- archivos implicados:
- pasos para reproducir:

Formato de salida:
1. causa probable
2. evidencia necesaria
3. fix propuesto
4. riesgos
5. pruebas recomendadas

## Plantilla para trading

Contexto:
Analiza [simbolo] en [timeframes].

Objetivo:
Determinar sesgo direccional, niveles clave y condiciones de invalidez.

Restricciones:
No usar lenguaje de certeza. Separar hechos, lectura y accion. Incluir riesgo.

Datos disponibles:
- simbolo:
- horizonte:
- tipo de operacion:
- contexto macro si aplica:

Formato de salida:
1. sesgo
2. confirmaciones
3. invalidez
4. escenarios alternos
5. accion sugerida
