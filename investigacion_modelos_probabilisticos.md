# Investigación — Modelos probabilísticos sobre el dataset Altur

> Análisis exhaustivo de las 353 llamadas de Altur con enfoque **probabilístico**:
> qué separa `human` de `synthetic`, qué modelos generativos/bayesianos lo capturan,
> y cómo usarlos. Reproducible con `scripts/eda_altur.py` y `scripts/train_probabilistic.py`.

## 1. Datos analizados

- **353 llamadas** (train 282: 113 human / 169 synthetic · val 71, **speaker-disjoint**).
- **42 features conversacionales** derivadas de `turns/<id>.json` con `src/features/conversational.py`
  (duraciones, latencias de turno, silencios, solapamientos, ratios de habla).
- Sin audio ni identidad de hablante → **señal conductual**, robusta a voces nuevas.
- Matriz reproducible en `models/altur_features.csv`.

## 2. Hallazgo específico: la firma de turn-taking

AUC univariada (train), ordenada:

| Feature | AUC | Humano | Sintético | Lectura |
|---|---|---|---|---|
| `time_to_first_caller_s` | **0.948** | 3.7 s | **10.2 s** | El bot tarda ~10 s en hablar (STT→LLM→TTS); el humano ~4 s |
| `caller_latency_mean` | 0.873 | **−0.40 s** | +1.64 s | El humano responde tan rápido que **interrumpe** (latencia negativa); el bot es lento |
| `caller_latency_median` | 0.860 | 0.30 s | 1.94 s | idem |
| `agent_speech_ratio` | 0.845 | 0.54 | 0.45 | — |
| `caller_latency_min` | 0.808 | −5.7 s | −1.6 s | El humano solapa fuerte al agente |
| `overlap_total_s` | 0.768 | 19.7 s | 6.5 s | El humano **barge-in**; el bot casi no |
| `n_caller_responses` | 0.780 | 12.3 | 9.1 | El humano interactúa más |

**Conclusión:** la señal más fuerte es la **latencia / toma de turnos**. Coincide con la literatura
(bots con latencia >500 ms que no hacen *barge-in*). Es **conductual → generaliza a hablantes no vistos**,
que es justo el enemigo del set oculto.

## 3. Modelos probabilísticos (por qué cada uno)

Todos devuelven una **probabilidad calibrada** (útil para el `confidence` del endpoint):

- **Gaussian Naive Bayes** — clase-condicional gaussiana con independencia de features. Baseline generativo.
- **LDA** — gaussiano con covarianza compartida (frontera lineal). Estable con pocos datos.
- **QDA** — gaussiano con **covarianza por clase** (frontera cuadrática). Captura que la dispersión
  de latencias difiere entre humano y bot. `reg_param=0.5` para no singularizar con 42 dims.
- **Bayes LogReg** — logística con regularización L2 (= **prior gaussiano / MAP bayesiano**) +
  calibración isotónica OOF.
- **GMM likelihood-ratio** — GMM por clase (2 componentes, sobre PCA-10) → razón de verosimilitud
  `log p(x|synth) − log p(x|human)` → calibrada a probabilidad. Modelo **generativo** puro.

## 4. Resultados (val speaker-disjoint)

| Modelo | AUC | EER | Brier | ACC |
|---|---|---|---|---|
| **QDA** | **0.991** | 0.085 | **0.035** | 0.944 |
| **Bayes LogReg** | 0.986 | **0.055** | 0.051 | 0.915 |
| LDA | 0.967 | 0.128 | 0.062 | 0.944 |
| GMM-LLR | 0.963 | 0.141 | 0.081 | 0.859 |
| Gaussian NB | 0.942 | 0.128 | 0.208 | 0.789 |

**QDA** es el mejor y el mejor calibrado; **Bayes-LogReg** tiene el mejor EER. Ambos rivalizan con
el LightGBM tabular (val AUC ~0.99) pero son **generativos/calibrados e interpretables**.

## 5. Cómo usarlos

```python
import joblib, pandas as pd
from src.features.conversational import extract_features_from_turns, FEATURE_ORDER

m = joblib.load('src/models/saved/prob_qda.joblib')       # {model, feature_cols, kind}
feats = extract_features_from_turns(turns, duration_s)      # dict de 42 features
X = pd.DataFrame([[feats[c] for c in m['feature_cols']]], columns=m['feature_cols'])
p_synth = m['model'].predict_proba(X.values)[0, 1]         # prob calibrada de sintético
```

## 6. Uso recomendado en el ensemble

- Añadir **QDA** o **Bayes-LogReg** como **miembro probabilístico calibrado** junto al LightGBM y al audio (WavLM).
- Su `confidence` calibrado (Brier bajo) mejora el desempate y la calibración global del `POST /detect`.
- Se pueden promediar/stackear con los otros; al ser conductuales, **no** sobreajustan a voces.

## 7. Advertencia

`val` es speaker-disjoint pero **mismo dominio de captura** que `train`. La firma de latencia es
conductual y debería sostenerse en el set oculto, pero conviene no fijar el umbral solo con `val`
(ver calibración cross-corpus del audio en `plan_v4_deteccion.md`).

### Reproducir
```
python scripts/eda_altur.py            # EDA + matriz models/altur_features.csv
python scripts/train_probabilistic.py  # entrena y evalua los 5 modelos -> src/models/saved/prob_*.joblib
```
