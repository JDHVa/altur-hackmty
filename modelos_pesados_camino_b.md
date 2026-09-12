# Modelos pesados Camino B — STT SOTA + anti-spoofing + probabilísticos

> Resultado de `plan_v4_deteccion.md` + el plan de modelos pesados. Todo entrenado y con
> scorers de inferencia listos para el ensemble. Held-out real = FLEURS-test + emilio (humanos)
> y **Piper (motor neural NO visto)**.

## Resumen de señales nuevas (held-out)

| Señal | Módulo / scorer | Piper (no visto) | Humanos | Nota |
|---|---|---|---|---|
| **XLS-R-SLS** (SOTA) | `src/features/heavy_audio.py:xlsr_sls_score` | **AUC 1.000, @0.5 100%** | 98% ok | **el mejor cross-motor** |
| Flow-LLR (zuko) | `heavy_audio.py:flow_llr_score` | AUC 0.892, @0.5 5% | 100% | fuerte on-domain |
| WavLM head | `src/features/audio.py:audio_score` | AUC 0.979, @0.5 15% | 100% | contrato base |
| Probabilísticos tabular | `src/models/saved/prob_*.joblib` | — | val AUC 0.99 | QDA/Bayes-LogReg calibrados |
| Lingüísticas | `src/features/linguistic.py` | — | train AUC 0.86 | articulation_rate/n_words/rate |

**Recomendación:** `xlsr_sls_score` como señal de audio principal (o miembro de mayor peso);
generaliza a motores TTS nuevos (el riesgo del set oculto) donde WavLM/flow fallan a umbral 0.5.

## Componentes

- **STT** (`src/features/stt.py`): `transcribe(wave, sr)` → Whisper large-v3 en GPU (~2s/20s),
  Parakeet-v3 opcional si se instala `nemo_toolkit[asr]` (`STT_BACKEND=parakeet`). Alimenta señal C + lingüísticas.
- **Lingüísticas** (`src/features/linguistic.py`): `linguistic_features(stt_result)` → tasa de habla,
  articulación, muletillas, pausas, TTR. Top separadores: `articulation_rate` 0.856, `n_words` 0.781.
- **XLS-R-SLS** (`src/models/xlsr_sls.py` + `src/models/saved/xlsr_sls.pt`): XLS-R 300M frozen, 25 capas con
  pesos sigmoides aprendidos (Sensitive Layer Selection) + MLP. Reentrenar: `scripts/train_xlsr_sls.py`.
- **Flow-LLR** (`src/models/flow_llr.py` + `flow_llr.joblib`): normalizing flows (zuko) por clase → LLR.
  Reentrenar: `scripts/train_flow_llr.py`.
- **Probabilísticos** (`scripts/train_probabilistic.py` → `prob_*.joblib`): QDA/LDA/NB/GMM/Bayes-LogReg
  sobre las 42 features conductuales (ver `investigacion_modelos_probabilisticos.md`).
- **Diarización pyannote** (`src/features/diarization.py`, opt-in): mejora los turnos (señal más fuerte).
  Requiere `pip install pyannote.audio` + aceptar términos + `HF_TOKEN`. `available()` degrada a falso sin token.

## Integración en el ensemble (para Jesús)

En `api/inference.py` / `src/ensemble.py`, añadir señales como miembros calibrados:
```python
from src.features.heavy_audio import xlsr_sls_score, flow_llr_score
p_xlsr = xlsr_sls_score(caller, sr)   # prob sintetico, cross-motor robusto
p_flow = flow_llr_score(caller, sr)   # generativo, fuerte on-domain
# stackear [p_tabular, p_wavlm, p_xlsr, p_flow, p_semantico] con logistica + calibracion
```
- Umbral por EER/minDCF sobre dev cross-corpus (no solo Altur val).
- Warm-up de XLS-R/WavLM al arrancar la API (carga fría ~30-60s; warm ~0.2-0.3s/llamada en GPU).
- Marcar GPU-only o usar modo "GPU worker" (Alonso) si el `/detect` en CPU excede 5s.

## Pesos y activación de turnos pyannote
```
pip install pyannote.audio
export HF_TOKEN=hf_xxx   # tras aceptar términos de pyannote/voice-activity-detection
```
Luego en `conversational.py` usar `diarization.turns_from_audio_pyannote(...)` cuando `available()`.

## Compute
- Entrenamiento pesado (XLS-R all-layers, flows): RTX 5050 (8GB) preferida; extracción/inferencia frozen cabe en RTX 4050.
- Descargas: XLS-R-300m ~1.2GB, WavLM-base-plus ~380MB, Whisper large-v3 ~3GB, pyannote ~30MB.
