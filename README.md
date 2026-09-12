# Centinela Altur — HackMTY 2026

Detección de voz sintética en llamadas bancarias (es-MX): ensemble conversacional (A) + anti-spoofing acústico (B),
expuesto como `POST /detect` y como una consola web en tiempo real para el operador.

## Correr todo en local

```bash
# 1. API (FastAPI) — modelos en src/models/saved, audio en hackmty26/audio
pip install -r requirements.txt
python -m uvicorn api.main:app --port 8000

# 2. Web (Next 16 + Tiger Data)
cd web
cp .env.example .env        # DATABASE_URL de Tiger Cloud + BETTER_AUTH_SECRET
pnpm install
pnpm db:migrate             # crea tablas + hypertable call_scores
pnpm dev                    # http://localhost:3000
```

## Web (`web/`)

| Ruta | Qué hace |
|---|---|
| `/login`, `/register` | Better Auth (email + password; Google opcional). Roles `operator` / `admin`. |
| `/` | Panel: KPIs de 24 h, llamadas por hora (`time_bucket` de TimescaleDB), recientes. |
| `/call/new` | **Sala de llamada.** Izquierda: simulación del que marca (micrófono en vivo o llamada real del dataset). Derecha: consola del operador con gauge de prob. IA, semáforo *Continuar / Verificar / Colgar*, timeline del score, señales A/B y biomarcadores. |
| `/calls`, `/calls/[id]` | Historial y detalle (score en el tiempo, notas, export JSON, audio si fue mic). |
| `/settings` | Cuenta, estado de la API, tema. |

Stack: Next.js 16 (App Router), React 19, Tailwind v4, shadcn/ui, Motion, Recharts, Drizzle ORM, Better Auth,
**Tiger Data** (Postgres + TimescaleDB: `call_scores` es hypertable con continuous aggregate por hora).

## API (`api/`)

| Endpoint | Descripción |
|---|---|
| `POST /detect` | Contrato oficial del reto: `{audio_base64}` → `{is_synthetic, confidence}`. |
| `POST /detect/detailed` | Igual pero con `p_final`, `p_tabular`, `p_audio`, `threshold`, `bio`, `recommendation`. |
| `WS /ws/call?source=mic\|dataset&anon_id=&speed=` | Scoring por ventana en vivo. `mic`: el cliente manda PCM int16 8 kHz; `dataset`: el servidor reproduce el WAV estéreo. Emite `frame` cada ~1 s y `final` al colgar. |
| `GET /dataset/calls`, `/dataset/calls/{id}/audio`, `/dataset/calls/{id}/label` | Llamadas del dataset para el modo simulación (label oculta hasta "Revelar"). |

### Señales de audio (Camino B) y flags

`p_audio` es el promedio ponderado de las señales disponibles: **XLS-R-SLS 0.5 · WavLM 0.3 · Flow-LLR 0.2**
(las que fallen o no estén instaladas se omiten). Cada respuesta trae `signals: {wavlm, xlsr, flow}`.

| Variable | Default | Efecto |
|---|---|---|
| `ALTUR_HEAVY` | `1` | `0` desactiva XLS-R y Flow-LLR (solo WavLM). Útil en CPU sin los pesos. |
| `ALTUR_HEAVY_XLSR` / `ALTUR_HEAVY_FLOW` | `1` | Apagar una señal pesada en particular. |
| `ALTUR_HEAVY_EVERY_S` | `4` | En `/ws/call`, cada cuántos segundos de llamada se recalculan las pesadas (WavLM va en cada tick). |
| `ALTUR_WARMUP` | `1` | Carga modelos en background al arrancar (XLS-R frío ~30–60 s). |
| `ALTUR_CORS_ORIGINS` | `http://localhost:3000` | Orígenes permitidos (separados por coma). |

Requiere `zuko` y los pesos en `src/models/saved/` (`xlsr_sls.pt`, `flow_llr.joblib`); XLS-R 300M se descarga de HF la primera vez.
En GPU (Emilio/Alonso) correr con todo activo; en CPU: `ALTUR_HEAVY=0 python -m uvicorn api.main:app --port 8000`.

Semáforo del operador sobre `p` (prob. de sintético): `p < 0.35` continuar · `0.35 ≤ p < umbral` verificar · `p ≥ umbral` colgar.
El umbral es el calibrado por EER del ensemble (`src/models/saved/ensemble.pkl`).

Ver [`CLAUDE.md`](CLAUDE.md), [`PLAN.md`](PLAN.md) y [`EQUIPO.md`](EQUIPO.md) para el contexto del reto y la división de trabajo.
