# altur_100 — sistema A+B aislado

Detector humano/IA **autocontenido** enfocado 100% en la distribución de Altur.
No depende del resto del repo (salvo el dataset `hackmty26/` para reentrenar).
El sistema anterior (`api/`, `src/`) queda intacto.

## Qué es

Modelo de decisión **HistGradientBoostingClassifier** (`altur_ab.joblib`) que fusiona:

- **A — conversacional (42 features):** `turns_from_audio` (VAD) → `extract_features_from_turns`
  (latencias, solapes, silencios, `time_to_first_caller_s`, ratios de habla, nº de turnos…).
- **B — audio (3 señales):**
  - `xlsr` → XLS-R `facebook/wav2vec2-xls-r-300m` (SSL congelado) + cabeza XLSRSLS (`xlsr_sls.pt`)
  - `wavlm` → WavLM-base-plus + cabeza (`wavlm_head.joblib`)
  - `flow` → `flow_llr.joblib`

Umbral de decisión: **0.5**.

## Rendimiento (sobre Altur)

- CV 5-fold × 5 semillas: **acc 99.26% (±0.23)**, AUC 0.9992
- Split oficial train→val: **100%** (0 FP, 0 FN), AUC 1.0

## Estructura

```
altur_100/
├── pipeline.py   extraccion de features + predict(data, sr)
├── train.py      reentrena altur_ab.joblib desde hackmty26/
├── infer.py      CLI: python infer.py archivo.wav
├── serve.py      FastAPI: POST /detect  -> {is_synthetic, confidence}
└── src/          codigo y pesos copiados (features/, models/, models/saved/)
```

## Uso

Inferencia de un archivo:

```bash
python infer.py ruta/al/audio.wav
```

Servidor (contrato del reto):

```bash
python -m uvicorn serve:app --host 127.0.0.1 --port 8010
# POST /detect  body: {"audio_base64": "..."}  -> {"is_synthetic": bool, "confidence": float}
```

Reentrenar (desde `hackmty26/` en la raíz del repo):

```bash
python train.py
```

## Notas

- Entrada esperada: WAV estéreo (canal 0 = quien llama, canal 1 = agente). Mono también funciona.
- El modelo desplegado se entrenó con los 353 calls (train+val) para máxima data;
  la cifra realista para voces nuevas del juez es la de CV (~99.3%).
