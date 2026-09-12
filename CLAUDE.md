# CLAUDE.md

Guía para agentes (Claude Code, etc.) que trabajen en este repositorio.

## Qué es este proyecto

Solución para el **Altur Challenge — HackMTY 2026**: dado el audio de una llamada
telefónica a un banco (en español mexicano), decidir si quien llama es una
**persona real (`human`)** o una **IA autónoma (`synthetic`)** — reconocimiento de
voz + LLM + voz sintética marcando el mismo número.

Es un problema de **detección de voz sintética / anti-spoofing** sobre
conversaciones reales de atención a clientes.

## Estructura

```
Altur/
├── CLAUDE.md                 ← este archivo
├── GEMINI.md                 ← misma guía para Gemini
├── Reconocimiento de Voz Bancario.docx   ← enunciado / notas del reto
└── hackmty26/                ← datos del reto (repo de la organización: alturio/hackmty26)
    ├── README.md             ← especificación oficial del challenge (LEER)
    ├── manifest.csv          ← 1 fila por llamada: anon_id, label, split, duration_s
    ├── dataset_turns_features.csv  ← features ya extraídas por turno/llamada
    ├── turns/<anon_id>.json  ← segmentos de habla por canal
    ├── audio/<anon_id>.wav   ← audio (se obtiene descomprimiendo el zip, ver abajo)
    └── altur-challenge-audio.zip   ← ~671 MB, audio comprimido
```

## Setup — LO QUE SE OCUPA AHORITA

El audio se distribuye comprimido. **Antes de entrenar o inferir hay que
descomprimir** `altur-challenge-audio.zip` para que los `.wav` queden en
`hackmty26/audio/`:

```bash
cd hackmty26
unzip -o altur-challenge-audio.zip     # deja los .wav en audio/
ls audio | wc -l                        # deben ser 353 archivos
```

> Nota: en `hackmty26/.gitignore` se ignoran `audio/` y `*.zip` porque el repo
> original de la organización distribuye el audio por Releases. En **este** repo
> queremos versionar el audio, así que se maneja con **Git LFS** (ver más abajo).

## Datos

- **`audio/<anon_id>.wav`**: estéreo, 8 kHz, 16-bit PCM.
  - Canal 0 = quien llama (**lo que hay que clasificar**).
  - Canal 1 = el agente del banco.
- **`turns/<anon_id>.json`**: `{"turns": [{"channel": 0, "start": 12.4, "end": 15.1}, ...]}`
  (segundos desde el inicio). Derivados automáticamente; usar como punto de partida.
- **`manifest.csv`**: `anon_id, label (human|synthetic), split (train|val), duration_s`.
- **`dataset_turns_features.csv`**: features ya calculadas (latencias, solapamientos,
  silencios, ratios de habla, etc.) útiles para un baseline tabular.
- `train` y `val` son **speaker-disjoint**. El set de evaluación es oculto, con
  llamadas de voces que no aparecen en ningún split → **cuidado con el overfitting
  a hablantes**.

## Objetivo de entrega

Exponer un endpoint `POST /detect` que recibe un WAV estéreo (8 kHz, base64,
canal 0 = caller, canal 1 = agent) y responde:

```json
{"is_synthetic": true, "confidence": 0.87}
```

`is_synthetic` es obligatorio; `confidence` es opcional (rompe empates y premia
calibración).

## Plan y arquitectura (DECIDIDO) — leer `PLAN.md` y `EQUIPO.md`

El roadmap oficial está en **[`PLAN.md`](PLAN.md)** y la asignación por persona en
**[`EQUIPO.md`](EQUIPO.md)**. Resumen de decisiones para agentes:

- **Ensemble de 3 señales**, calibrado (Platt/isotónica) para `confidence`:
  - **A. Conversacional** — 45 features de `dataset_turns_features.csv` → **XGBoost** (backbone).
  - **B. Audio anti-spoofing** — **Wav2Vec2-AASIST** con SSL **frozen** + prosodia sobre canal 0
    (SOTA 2025; frozen evita sobreajuste a hablantes). Modelos HF listos para score zero-shot.
  - **C. Semántico** — transcripción (`faster-whisper`) + LLM zero-shot (cadencia robótica,
    prompt leakage, reacción a preguntas trampa del agente).
- **Fuera del clasificador:** voiceprints ECAPA-TDNN (codifican identidad → overfit a voces vistas)
  y FHE (verificación con enrolamiento, que este reto no tiene). FHE = demo opcional.
- **Infra híbrida:** core local que siempre funciona + sponsors oportunistas (Gemini/Vultr;
  Snowflake/TigerGraph solo bonus). Hay **2 GPUs locales** → la GPU de Vultr no hace falta.
- **Regla de oro:** el endpoint `POST /detect` con **solo la señal A** ya es entregable válido;
  todo lo demás suma pero **nunca bloquea** ese mínimo.
- **3 caminos paralelos** (Jesús = A/API en CPU, Emilio = B/audio en RTX 4050,
  Alonso = hardware "Centinela Altur" Pi 5 + GPU worker en RTX 5050). Contratos de interfaz en `EQUIPO.md`.

## Flujo de trabajo Git / colaboración (OBLIGATORIO)

Somos **3 personas trabajando en paralelo**. Para no chocar y poder testear juntos:

- **Cambios chicos e incrementales.** Nada de PRs/commits gigantes. Cada tarea se parte en
  pasos pequeños y **testeables por separado**. Si un cambio toca muchos archivos a la vez, pararse
  y dividirlo.
- **`git pull` ANTES de empezar cualquier cambio** (y otra vez antes de commitear) para integrar lo
  de los demás y evitar conflictos grandes.
- **`git push` en cuanto haya un cambio considerable** que compile/pase pruebas — no acumular trabajo local.
  Regla práctica: si llevas más de ~30–60 min sin pushear algo funcional, es señal de que el cambio es muy grande.
- **Cada commit debe dejar el repo en estado funcional** (que corra / pase el test mínimo). No commitear código roto a `main`.
- Mensajes de commit claros y por camino, p.ej. `A: baseline tabular`, `B: score audio zero-shot`, `C: captura Pi`.
- Respetar los **contratos de interfaz** de `EQUIPO.md` para que los 3 caminos integren sin fricción.

> Para agentes (Claude/Gemini): **haz `git pull` antes de editar y `git push` tras cada cambio
> considerable y funcional.** Mantén los diffs pequeños; nunca dejes cambios grandes sin pushear.

## Términos

Los humanos usaron datos personales inventados. **No intentar identificar a
nadie.** Dataset solo para HackMTY 2026, **no redistribuir**.
