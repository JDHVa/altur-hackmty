# Setup del entorno — qué está en .gitignore y cómo obtenerlo

> Qué necesita un compañero al clonar el repo, separando lo que **SÍ viaja en git**
> de lo **gitignored** (hay que generarlo/obtenerlo).

## Para el barrido de parámetros (Persona A/B) — NO necesitas nada gitignored
Solo Python estándar. Todo lo necesario ya está versionado:
- `scripts/eval_config.py` + `demo_scores.csv` (las señales ya calculadas).
```
git clone <repo> && cd altur-hackmty
python scripts/eval_config.py --wavlm 0.2 --xlsr 0.6 --prosody 0.1 --flow 0.1 --threshold 0.30
```
No hace falta venv, GPU, ni modelos. (Ver `planes/plan_parametros.md`.)

## Para correr el detector completo (analyze_audio / regenerar demo_scores)
Esto SÍ usa cosas pesadas. Necesitas:

### 1. venv + dependencias (gitignored: `.venv/`)
```
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt
```

### 2. Modelos entrenados (en `src/models/saved/`)
- **SÍ viajan en git** (excepciones en `.gitignore`): `wavlm_head.joblib`, `xlsr_sls.pt`,
  `flow_llr.joblib`, `prosody_clf.joblib`, `prob_*.joblib`, `ensemble_full.joblib`,
  y por **Git LFS**: `best_audio_resnet.pth`, `audio_resnet.onnx`.
  → tras clonar: `git lfs install && git lfs pull`.
- Los grandes SSL (WavLM, XLS-R) **se descargan solos** de Hugging Face la 1a vez (~1-3GB c/u, cache local).

### 3. Datos (gitignored)
- `hackmty26/` (dataset del reto) — obtener del repo de la organización + `unzip audio`.
- `datasets_externos/` (FLEURS, sintéticos) — regenerar con `scripts/download_human_es.py`,
  `generate_synthetic_mx.py`, `generate_synthetic_engines.py`.
- `models/ssl_cache/` — caches de features; se regeneran solos al correr los scripts.
- `wav_demos/` — **SÍ viaja en git** (voces reales + clones del equipo).

### 4. Clonación de voz (XTTS) — venv APARTE (gitignored: `.venv_tts/`)
```
python -m venv .venv_tts
.venv_tts\Scripts\pip install torch torchaudio coqui-tts==0.24.2 click
.venv_tts\Scripts\python scripts/clone_voice.py --ref voz.wav --out clon.wav
```
OJO: no instalar `coqui-tts`/`pyannote`/`piper-tts` en el venv principal — rompen el torch CUDA.

## Resumen: lo gitignored que hay que obtener
| Qué | Cómo |
|---|---|
| `.venv/` | `python -m venv` + `pip install -r requirements.txt` |
| SSL WavLM/XLS-R | descarga automática de HF (1a corrida) |
| `hackmty26/` | repo de la organización + unzip |
| `datasets_externos/` | scripts de descarga/generación |
| `.venv_tts/` | venv aparte para XTTS |
| `HF_TOKEN` | solo para pyannote (opcional) |
