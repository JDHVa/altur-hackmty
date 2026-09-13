# Centinela Altur — Presentación de tecnologías (texto para Canva)

> 12 slides. Cada slide: título · 3–5 bullets cortos · una línea de "qué decir".
> Sugerencia visual: fondo `#0B0F14`, acento cian `#22D3EE`, semáforo verde `#34D399` / ámbar `#FBBF24` / rojo `#F43F5E`, tipografía Geist o Inter.

---

## Slide 1 — Portada
**Centinela Altur**
Detección de voz sintética en llamadas bancarias, en tiempo real.
HackMTY 2026 · Altur Challenge · Jesús · Emilio · Alonso

*Qué decir:* "Un bot con voz clonada marca al banco. Nosotros le decimos al operador si colgar, antes de que el fraude ocurra."

---

## Slide 2 — El problema
- Voz clonada + LLM = un atacante que suena humano y contesta preguntas de seguridad.
- El reto: audio de una llamada (8 kHz, estéreo) → `human` o `synthetic`.
- El set de evaluación es oculto y con **voces nunca vistas** → no podemos memorizar hablantes.
- Entregable: `POST /detect` → `{is_synthetic, confidence}`.

*Qué decir:* "No basta con reconocer una voz; hay que reconocer que **no es** una voz."

---

## Slide 3 — Arquitectura en una imagen
```
Caller (mic / dataset) ─▶ Consola web (Next.js) ─▶ API (FastAPI, WebSocket)
                                                     │
                                  ┌──────────────────┴──────────────────┐
                                  │ Ensemble A + B (+ C)                 │
                                  │ Conversacional · Acústico · Semántico│
                                  └──────────────────┬──────────────────┘
                                                     ▼
                                         Tiger Data (Postgres + TimescaleDB)
```
- 3 señales independientes, fusionadas con calibración.
- Un semáforo para el operador: **Continuar / Verificar / Colgar**.

*Qué decir:* "Tres detectores que miran cosas distintas; si uno falla, los otros siguen."

---

## Slide 4 — Señal A · Conversacional
- 45 features de la **dinámica de la llamada**: latencia de respuesta, solapamientos, silencios, ritmo de turnos.
- Turnos por canal con **webrtcvad** (o pyannote).
- Clasificador **LightGBM** → AUC 0.99 en validación (hablantes disjuntos).
- Corre en CPU en milisegundos; no codifica identidad → generaliza a voces nuevas.

*Qué decir:* "Un bot tarda lo mismo siempre en contestar; un humano duda, interrumpe, se ríe."

---

## Slide 5 — Señal B · Anti-spoofing acústico
- **XLS-R-SLS**: XLS-R 300M congelado, 25 capas con *Sensitive Layer Selection* + MLP → AUC **1.0** en motor TTS no visto (Piper).
- **WavLM head**: WavLM-base-plus congelado + cabeza ligera (contrato base).
- **Flow-LLR**: normalizing flows (zuko) → razón de verosimilitud humano/IA.
- Biomarcadores con **Praat**: jitter, shimmer, HNR (explicables en pantalla).
- Ventanas de 6 s promediadas; pesos afinados con barrido: XLS-R 0.5 · WavLM 0.4 · Flow 0.1.

*Qué decir:* "Modelos SSL **congelados** + cabezas chicas: aprenden artefactos de síntesis, no la voz de la persona."

---

## Slide 6 — Señal C · Semántica y lingüística
- STT: **Whisper large-v3** (GPU) / faster-whisper (CPU) / Parakeet opcional.
- Features lingüísticas: tasa de articulación, muletillas, pausas, riqueza léxica (AUC 0.86).
- LLM zero-shot local (**Ollama · qwen2.5**): cadencia robótica, *prompt leakage*, reacción a preguntas trampa.

*Qué decir:* "Si el que llama nunca dice 'este…' ni se equivoca, sospechamos."

---

## Slide 7 — Datos y canal telefónico
- Dataset Altur: 353 llamadas reales es-MX, 8 kHz, canal 0 = caller.
- Externos: FLEURS (humanos), ASVspoof 2019, MLAAD; sintéticos propios con Edge-TTS, gTTS, Piper, SAPI.
- **Clonación con XTTS-v2** de nuestras propias voces para el set de demo.
- Todo pasa por el **mismo canal telefónico** (ffmpeg: G.711 μ-law, Opus, pérdida de paquetes, ruido).

*Qué decir:* "Entrenar en estudio y evaluar por teléfono es hacer trampa; nosotros simulamos la línea."

---

## Slide 8 — Fusión y calibración
- Calibración **isotónica** out-of-fold + umbral por **EER**.
- Modelo final A+B: **HistGradientBoosting** sobre 42 features conversacionales + 3 señales de audio → 100 % train→val, 99.3 % en CV.
- Confianza calibrada (Platt) para desempates.
- Semáforo: `p < 0.35` continuar · `0.35–umbral` verificar · `≥ umbral` colgar.

*Qué decir:* "No entregamos solo un sí/no: entregamos una probabilidad en la que se puede confiar."

---

## Slide 9 — API en tiempo real
- **FastAPI + uvicorn**: `POST /detect` (contrato del reto), `/detect/detailed`, `/detect/audio`.
- **WebSocket `/ws/call`**: score por ventana cada ~1 s con suavizado EMA; modo mic (PCM 8 kHz) y modo dataset.
- Fallbacks en cascada: modelo pesado → ligero → stub. El endpoint nunca se cae.
- Docker; despliegue en **Modal (GPU)**, **HF Space** y **Vultr**.

*Qué decir:* "El operador ve el riesgo subir mientras la llamada sucede, no al final."

---

## Slide 10 — Consola del operador (web)
- **Next.js 16 · React 19 · Tailwind v4 · shadcn/ui · Motion · Recharts**.
- Sala de llamada: teléfono simulado (mic en vivo vía **AudioWorklet** o llamada real del dataset) + gauge de prob. IA, semáforo, timeline, señales A/B y biomarcadores.
- Panel con KPIs, historial y detalle de cada llamada (notas, export, audio).
- **Better Auth** (roles operador/admin) · **Drizzle ORM**.

*Qué decir:* "Diseñada como consola de seguridad: una sola pregunta en pantalla — ¿cuelgo o no?"

---

## Slide 11 — Datos en Tiger Data (TimescaleDB)
- **Tiger Cloud**: Postgres 17 + TimescaleDB, cero infra.
- `call_scores` es una **hypertable**: una fila por segundo de llamada (score, señales, biomarcadores).
- *Continuous aggregate* por hora + `time_bucket` para el dashboard.
- Auditoría completa: quién atendió, qué recomendó el sistema, si colgó, etiqueta real.

*Qué decir:* "Cada llamada deja una serie de tiempo consultable: así se audita y se re-entrena."

---

## Slide 12 — Hardware, equipo y siguientes pasos
- **Centinela físico**: Raspberry Pi 5 + ONNX Runtime + anillo LED; modo "reto adversarial"; offload a GPU por HTTP.
- 3 caminos en paralelo (A/B/C) con contratos de interfaz; Git con commits chicos; entrenos en RTX 4050/5050 y Modal.
- Siguiente: recalibrar umbral cross-corpus, más clones difíciles (voice conversion), señal C al ensemble.

*Qué decir:* "Un endpoint que puntúa, una consola que convence y un dispositivo que se puede tocar."

---

### Cierre (una frase)
**Centinela Altur: detecta la voz que no existe.**
