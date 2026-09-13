# Centinel Altur — Documentación técnica de tecnologías

> HackMTY 2026 · Altur Challenge. Detección de voz sintética en llamadas bancarias (es-MX).
> Este documento describe **qué tecnologías usamos, para qué y cómo se conectan**. Para el roadmap ver `PLAN.md`; para la división de trabajo `EQUIPO.md`.

---

## 1. Visión general

```
                 ┌──────────────────────────────────────────────────────────────┐
  Caller (mic /  │  web/  Next.js 16 · React 19 · Tailwind v4 · shadcn · Motion │
  dataset)  ───▶ │  Sala de llamada ─ WebSocket ─▶ consola del operador          │
                 └───────────────┬──────────────────────────────┬───────────────┘
                                 │ WS /ws/call  · REST /detect  │ Drizzle ORM · Better Auth
                                 ▼                              ▼
                 ┌──────────────────────────────┐   ┌────────────────────────────┐
                 │  api/  FastAPI + uvicorn      │   │  Tiger Data (Tiger Cloud)  │
                 │  predict_detailed / live.py   │   │  Postgres 17 + TimescaleDB │
                 └───────────────┬──────────────┘   │  calls · call_scores (hyper)│
                                 ▼                  └────────────────────────────┘
                 ┌──────────────────────────────────────────────────────────────┐
                 │  src/  Ensemble                                              │
                 │  A · Conversacional  LightGBM sobre 45 features de turnos    │
                 │  B · Audio           WavLM head · XLS-R-SLS · Flow-LLR (zuko)│
                 │      + bio           Praat (jitter/shimmer/HNR)              │
                 │  C · Semántico       Whisper large-v3 / Parakeet · lingüísticas · LLM │
                 │  Fusión              calibración isotónica + umbral EER      │
                 └──────────────────────────────────────────────────────────────┘
```

Tres caminos independientes (A/B/C) unidos por **contratos de interfaz** (`src/features/audio.py::audio_score`, `src/ensemble.py`, `POST /detect`). El endpoint `POST /detect` con solo la señal A ya es entregable válido; todo lo demás suma sin bloquear.

---

## 2. Datos y procesamiento de audio

| Tecnología | Uso | Dónde |
|---|---|---|
| **WAV estéreo 8 kHz 16-bit** | Formato del reto: canal 0 = caller (lo que se clasifica), canal 1 = agente | `hackmty26/audio/` |
| **soundfile / numpy** | Lectura/escritura de WAV, decodificación base64 en el endpoint | `api/inference.py::decode_wav` |
| **torchaudio** | Resampleo 8 kHz → 16 kHz para los modelos SSL | `src/features/audio.py`, `heavy_audio.py` |
| **webrtcvad** (fallback energía) | VAD para derivar turnos por canal cuando no hay `turns/*.json` (paridad con los turnos oficiales) | `src/features/conversational.py::turns_from_audio` |
| **pyannote.audio 4.x** (opt-in, `HF_TOKEN`) | Diarización/VAD neural para turnos más precisos | `src/features/diarization.py` |
| **ffmpeg** (vía `imageio-ffmpeg`) | Simulación de canal telefónico: códecs G.711 μ-law/a-law, Opus, pérdida de paquetes, ruido, ganancia | `src/features/telephony_aug.py::augment_chain` |
| **praat-parselmouth** | Biomarcadores de voz: jitter, shimmer, HNR, F0, fracción sonora | `src/features/prosody.py` |
| **Edge-TTS, gTTS, Piper, SAPI** | Generación de sintéticos es-MX para entrenar/evaluar (Piper se reserva como motor *held-out*) | `scripts/generate_synthetic_*.py`, `scripts/make_synthetic.py` |
| **Coqui XTTS-v2** | Clonación de voz (ataque realista) para el set de demos | `scripts/clone_voice.py`, `wav_demos/` |
| **HF datasets (FLEURS es)**, ASVspoof 2019 LA, MLAAD | Humanos reales y sintéticos externos para evaluación cross-corpus | `scripts/download_*.py`, `datasets_externos/` |
| **sounddevice** | Grabar humanos propios ya pasados por canal telefónico | `scripts/record_human.py` |

Principio: **todo lo que entra al modelo pasa por el mismo canal telefónico** (8 kHz + μ-law) para no comparar dominios distintos.

---

## 3. Señal A — Conversacional (tabular)

| Tecnología | Detalle |
|---|---|
| **pandas / numpy** | 45 features por llamada: latencias de respuesta, solapamientos, silencios, ratios de habla, estadísticas de duración de turnos. Paridad exacta con `dataset_turns_features.csv`. |
| **LightGBM** | Clasificador backbone. VAL AUC ≈ 0.99 (splits speaker-disjoint). `src/models/saved/lgbm_tabular.pkl` |
| **XGBoost** | Disponible como alternativa (plan original), no es el modelo activo. |
| **scikit-learn** (QDA, LDA, GaussianNB, GMM-LLR, Bayes-LogReg) | Modelos probabilísticos sobre las 42 features conductuales, calibrados; sirven para `ensemble_full`. `src/models/saved/prob_*.joblib` |

Por qué tabular primero: es robusto a **hablantes no vistos** (no codifica identidad de voz) y corre en CPU en milisegundos.

---

## 4. Señal B — Anti-spoofing acústico

| Modelo | Tecnología | Rol | Resultado clave |
|---|---|---|---|
| **WavLM head** | `microsoft/wavlm-base-plus` **congelado** (transformers) + cabeza sklearn (scaler + clasificador) sobre mean/std del último estado oculto | Contrato base `audio_score`. Fallback: ResNet mel-spec (PyTorch) → stub 0.5 | Piper held-out AUC 0.979; humanos 100 % |
| **XLS-R-SLS** | `facebook/wav2vec2-xls-r-300m` congelado, **25 capas** con pesos sigmoides aprendidos (*Sensitive Layer Selection*) + MLP (PyTorch) | Señal principal cross-motor | Piper AUC **1.000** |
| **Flow-LLR** | *Normalizing flows* con **zuko** por clase sobre embeddings WavLM → razón de verosimilitud | Señal generativa fuerte on-domain | Piper AUC 0.892 |
| **ResNet mel-spec** | CNN propia sobre espectrograma mel (`src/models/audio_model.py`), export **ONNX** | Fallback y versión para edge (Pi 5) | VAL AUC 0.9976 |
| **Prosody clf** | sklearn sobre jitter/shimmer/HNR | Interpretable para demo; **peso 0 en producción** (subía falsos positivos) | — |

Detalles de inferencia:
- Todos los scorers promedian **ventanas de 6 s (máx. 8)** para ser robustos a la duración.
- `src/features/wavlm_embed.py`: una sola pasada de WavLM compartida entre `audio_score` y `flow_llr`.
- Fusión de audio (`api/inference.py::combine_audio`): promedio ponderado de las señales disponibles. Config ganadora del barrido (`plan_parametros.md`): **wavlm 0.40 · xlsr 0.50 · flow 0.10 · prosody 0** → ACC 0.917, FN 2, FP 0 en 24 demos.
- Entrenamiento pesado en **Modal** (`scripts/modal_app.py`, GPU A10G, volúmenes `altur-data`/`altur-models`) o en las RTX 4050/5050 del equipo.

Descartado: modelos HF zero-shot de deepfake-audio (AUC 0.41 en val, domain gap a 8 kHz/español); augmentation multi-códec y "más humanos" (no mejoran).

---

## 5. Señal C — Semántica / lingüística

| Tecnología | Uso |
|---|---|
| **faster-whisper** (CTranslate2) | Transcripción es en CPU (`src/features/semantic.py`) |
| **Whisper large-v3** (GPU) / **NVIDIA Parakeet-v3** (NeMo, opcional `STT_BACKEND=parakeet`) | STT SOTA con timestamps por palabra (`src/features/stt.py`) |
| `src/features/linguistic.py` | Tasa de habla, articulación, muletillas, pausas, TTR. Top separadores: `articulation_rate` AUC 0.86 |
| **Ollama + qwen2.5:3b** (local) / Gemini (sponsor, opcional) | LLM zero-shot: cadencia robótica, *prompt leakage*, reacción a preguntas trampa. Pendiente de integrar al ensemble |

---

## 6. Ensemble y calibración

- `src/ensemble.py`: **calibración isotónica** out-of-fold (`IsotonicRegression`, `StratifiedKFold`) y **umbral por EER** (`roc_curve`). Métricas: AUC, EER, Brier, accuracy.
- Fusión A+B: `final = w_tab · cal(p_tabular) + (1 − w_tab) · p_audio`, persistida en `ensemble.pkl` (`signals`, `fusion`, `calibrator`, `threshold`).
- `src/ensemble_full.py` / `scripts/train_ensemble_full.py`: variante con probabilísticos conductuales (QDA) + XLS-R-SLS.
- Semáforo del operador sobre `p` (prob. sintético): `< 0.35` continuar · `[0.35, umbral)` verificar · `≥ umbral` colgar.
- Pendiente crítico: umbral 0.5 alto para motores no vistos → recalibrar en dev cross-corpus (~0.09–0.30) y stacking logístico de `[p_tabular, p_wavlm, p_xlsr, p_flow, lingüísticas]`.

---

## 7. API (`api/`)

| Tecnología | Detalle |
|---|---|
| **FastAPI + uvicorn + pydantic** | `POST /detect` (contrato del reto), `POST /detect/detailed`, `GET /health` (`heavy: bool`) |
| **WebSocket nativo (FastAPI/websockets)** | `WS /ws/call?source=mic\|dataset&anon_id&speed`. Modo `mic`: recibe PCM int16 8 kHz; modo `dataset`: el servidor reproduce el WAV estéreo en tiempo real. Emite `frame` cada ~1 s (WavLM cada tick, pesadas cada `ALTUR_HEAVY_EVERY_S`=4 s, **EMA** α=0.45 por señal, tabular cada 6 s) y `final` al colgar |
| **asyncio.to_thread** | Inferencia fuera del event loop; flag `busy` evita solapar ticks |
| **CORSMiddleware** | `ALTUR_CORS_ORIGINS` (default `http://localhost:3000`) |
| Flags | `ALTUR_HEAVY`, `ALTUR_HEAVY_XLSR`, `ALTUR_HEAVY_FLOW`, `ALTUR_HEAVY_EVERY_S`, `ALTUR_WARMUP` (warm-up en hilo al arrancar) |
| **Docker** | `api/Dockerfile` para despliegue (Vultr/otro) |
| `api/dataset.py` | Sirve `manifest.csv` sin etiqueta, WAV y `label` bajo demanda ("Revelar") |

Todas las señales pesadas están **guardadas con try/except**: si falta `zuko`, pesos o GPU, esa señal se omite y el endpoint sigue respondiendo.

---

## 8. Consola web "Centinel" (`web/`)

| Capa | Tecnología | Por qué |
|---|---|---|
| Framework | **Next.js 16** (App Router, `proxy.ts`, `params` async) + **React 19** + TypeScript | Server Components, Server Actions, streaming |
| UI | **Tailwind v4** + **shadcn/ui v4** (Radix, preset Nova) + **lucide-react** + **sonner** | Componentes accesibles, tokens CSS |
| Animación | **Motion** (`motion/react`): springs en gauge, `AnimatePresence` en el semáforo, barras | Feedback continuo del score |
| Gráficas | **Recharts 3** (`AreaChart` timeline con zonas, `BarChart` por hora) | Serie temporal del score |
| Audio en navegador | **Web Audio API**: `AudioWorklet` (`public/pcm-worklet.js`) que baja a 8 kHz int16 y manda chunks de 250 ms por WS; `AnalyserNode` para la onda; `decodeAudioData` + `AudioBufferSourceNode` para reproducir el dataset con `playbackRate` | Sin dependencias, sin problemas de CORS en `<audio>` |
| Tiempo real | `WebSocket` nativo + hook `useLiveCall` | Frames → estado React |
| Auth | **Better Auth** (email/password, Google opcional, plugin `admin` con roles `operator`/`admin`, `nextCookies`) + `proxy.ts` con `getSessionCookie` | Sesiones en la misma DB |
| ORM / migraciones | **Drizzle ORM** (`node-postgres`) + **drizzle-kit** (`0000_init`, `0001_timescale`, `0002_heavy_signals`) | Tipado end-to-end |
| DB | **Tiger Data / Tiger Cloud** = Postgres 17 + **TimescaleDB**: `call_scores` es **hypertable** (`create_hypertable`), *continuous aggregate* `call_scores_hourly` con política de refresco; `time_bucket('1 hour')` en el dashboard | Series de tiempo de scores por llamada |
| Persistencia | Server Actions (`saveCall`, `updateCallNotes`, `setRevealedLabel`); audio del mic en `bytea` | Sin capa REST extra |
| Tema | Paleta "Centinel": fondo `#0B0F14`, acento cian `#22D3EE`, eje verde `#34D399` → ámbar `#FBBF24` → rojo `#F43F5E`; Geist Sans/Mono; dark-first con toggle | Consola de seguridad |
| Tooling | **pnpm**, ESLint 9, TypeScript 5.9, Turbopack | — |

Pantallas: `/login` `/register` · `/` panel (KPIs 24 h, llamadas por hora, recientes) · `/call/new` sala de llamada (caller simulado + consola del operador con gauge, semáforo, timeline, señales A/B/XLS-R/WavLM/Flow y biomarcadores) · `/calls`, `/calls/[id]` (detalle, notas, export JSON, audio) · `/settings`.

---

## 9. Evaluación y pruebas

| Herramienta | Uso |
|---|---|
| `scripts/eval_crosscorpus.py` | AUC/EER por corpus externo (FLEURS, Edge/gTTS/Piper, grabaciones propias) — el test que predice el set oculto |
| `scripts/eval_endpoint_val.py` | `/detect` end-to-end sobre Altur val con scorers reales |
| `scripts/eval_config.py` + `demo_scores.csv` | Barrido de pesos/umbral sin GPU sobre las 4 señales precalculadas |
| `scripts/analyze_audio.py` | CLI: cualquier wav/mp3/m4a → HUMANO vs IA con desglose |
| `scripts/score_demos.py`, `wav_demos/` | Pares real/clon del equipo para el pitch |
| Consola web + "Revelar" | KPI "Acierto vs etiqueta" en vivo |
| **Claude in Chrome** | QA visual del flujo caller → operador durante el desarrollo |

---

## 10. Infraestructura, colaboración y hardware

- **Git + GitHub** (`JDHVa/altur-hackmty`), commits chicos por camino (`A:`, `B:`, `web:`), `git pull` antes de editar, push tras cada cambio funcional. Rama `hardware` para el Camino C.
- **Git LFS** para el audio del reto; `.gitignore` para credenciales (`*credentials.env`, `.env`) y audio personal.
- **Modal** (GPU A10G) para entrenos pesados; **2 GPUs locales** (RTX 4050 / RTX 5050) para fine-tune e inferencia.
- **Tiger Cloud** (sponsor) como DB gestionada; Gemini/Vultr/Snowflake/TigerGraph solo como bonus.
- **Centinel Altur hardware** (Camino C, Raspberry Pi 5): ONNX Runtime arm64 + webrtcvad + anillo LED/pantalla, modo reto adversarial, fallback *offload* a la RTX 5050 por HTTP.
- Agentes de código (**Claude Code**, Gemini) guiados por `CLAUDE.md`/`GEMINI.md`: sin comentarios en código, planes marcados como definitivos no se re-analizan.

---

## 11. Decisiones de diseño (resumen)

1. **SSL congelado + cabeza chica** (WavLM, XLS-R) en vez de fine-tune completo → evita sobreajuste a hablantes del dataset chico.
2. **Nada de voiceprints (ECAPA-TDNN) ni FHE** en el clasificador: codifican identidad y requieren enrolamiento que el reto no tiene.
3. **Mismo canal telefónico** en entreno, evaluación y demo (μ-law 8 kHz).
4. **Fallbacks en cascada** en cada señal (modelo pesado → ligero → stub 0.5) para que el endpoint nunca se caiga.
5. **API stateless**; la persistencia y la sesión del operador viven en Next.js + Tiger Data.
6. **Umbral/pesos como dato, no como código**: `ensemble.pkl`, flags de entorno y barridos reproducibles (`demo_scores.csv`).
