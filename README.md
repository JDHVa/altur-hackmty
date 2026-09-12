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

Semáforo del operador sobre `p` (prob. de sintético): `p < 0.35` continuar · `0.35 ≤ p < umbral` verificar · `p ≥ umbral` colgar.
El umbral es el calibrado por EER del ensemble (`src/models/saved/ensemble.pkl`).

Ver [`CLAUDE.md`](CLAUDE.md), [`PLAN.md`](PLAN.md) y [`EQUIPO.md`](EQUIPO.md) para el contexto del reto y la división de trabajo.
