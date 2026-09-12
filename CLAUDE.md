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

## Ideas de enfoque

1. **Baseline tabular** con `dataset_turns_features.csv` (las señales
   conversacionales — latencias, solapamientos, silencios — ya distinguen bien
   humano vs IA). Rápido de montar.
2. **Modelo de audio** sobre el canal 0: mel-espectrograma / wav2vec2 y un
   clasificador; combinar con las features conversacionales.
3. El canal 1 (agente) da contexto: a qué reaccionaba el caller (pausas,
   interrupciones, preguntas trampa).

## Términos

Los humanos usaron datos personales inventados. **No intentar identificar a
nadie.** Dataset solo para HackMTY 2026, **no redistribuir**.
