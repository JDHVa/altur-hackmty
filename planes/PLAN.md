# Plan Maestro — Altur Challenge (HackMTY 2026)

> Plan oficial del proyecto. Fusiona `plan-jesus-altur.md` y
> `Plan_Arquitectura_Altur_HackMTY2026.docx`, corregido con los datos reales y el
> estado del arte 2025 en detección de voz sintética.
> **Asignación por persona → ver [`EQUIPO.md`](EQUIPO.md).**

## Estado actual (avance)

- ✅ **Señal A (conversacional)** — features con paridad exacta, LightGBM entrenado (VAL AUC 0.99).
- ✅ **API `POST /detect`** — probada HTTP end-to-end (acc 0.93 en val, pipeline VAD-consistente).
- ✅ **Ensemble + calibración** — isotónica OOF + fusión A+B (promedio); **API fusionada en val: 0.958**.
- ✅ **Señal B (audio)** — ResNet entrenado (Emilio) VAL AUC 0.9976, **integrado al ensemble**. (Zero-shot HF descartado: AUC 0.41.)
- ⬜ **Señal C (LLM zero-shot)** — no empezada (requiere decisión Gemini vs LLM local).
- 🟡 **Datasets externos + Edge-TTS MX** — MLAAD parcial; falta generador MX y reanudar descargas.
- ⬜ **Hardware "Centinel Altur"** (Camino C) — no empezado.

## Contexto

**Decisiones tomadas:** infraestructura **híbrida** (core local que siempre funciona +
sponsors oportunistas); **voiceprints ECAPA-TDNN y FHE fuera del clasificador** (los
voiceprints codifican identidad → sobreajustan a voces vistas; FHE resuelve verificación
con enrolamiento, que este reto no tiene). FHE queda como demo opcional de presentación.

**Datos (verificados):** 353 llamadas. Train 113 human / 169 synthetic (282);
Val 37h / 34s (71). Ya existen **45 features conversacionales** en `dataset_turns_features.csv`.
Audio estéreo 8 kHz, canal 0 = caller (a clasificar), canal 1 = agente.
**Eval oculta = voces nuevas** → el enemigo #1 es el overfitting a hablantes.

**Entregable:** `POST /detect` (WAV estéreo 8 kHz base64) → `{"is_synthetic": bool, "confidence": float}`, < 5 s.

---

## Tecnologías recomendadas (estado del arte 2025)

| Componente | Recomendado | Por qué / evidencia |
|---|---|---|
| **Front-end de audio** | SSL **frozen**: **Wav2Vec2-XLS-R / WavLM / HuBERT** | Las representaciones SSL wav2vec2/WavLM **rinden mejor** en anti-spoofing. Frozen = no memoriza hablantes (clave con voces nuevas + solo 282 calls). |
| **Back-end anti-spoofing** | **AASIST** (graph attention) sobre SSL → **Wav2Vec2-AASIST** | **SOTA** en ASVspoof2021. Variantes (AASIST3, Scalable AASIST con encoder frozen) mejoran en datos limitados. |
| **Clasificador final** | **XGBoost** sobre scores + features | **iWAX (Wav2vec-AASIST-XGBoost, 2025)** ≈ nuestro ensemble → valida el diseño e interpreta. |
| **Generalización a voces nuevas** | encoder frozen + augmentation; opcional **LoRA meta-aprendido** | "Generalizable deepfake detection via meta-learned LoRA" ataca justo dominios/voces no vistas. |
| **Modelos listos (HuggingFace)** | `Gustking/wav2vec2-large-xlsr-deepfake-audio-classification`, `mo-thecreator/Deepfake-audio-detection`, `garystafford/wav2vec2-deepfake-voice-detector`, `lab260/AASIST3` | **Score zero-shot inmediato** sin entrenar; luego se afina la cabeza. |
| **8 kHz telefónico** | resamplear canal 0 a 16 kHz + SSL frozen + cabeza propia | No hay modelo específico 8 kHz; frozen SSL + fine-tune de cabeza es lo robusto. |
| **ASR / transcripción** | `faster-whisper` local (o GCP STT `es-MX` telephony) | Para la señal semántica. |
| **Prosodia** | `librosa` + `parselmouth`/Praat | F0/jitter/shimmer/energía — sintéticos = prosodia plana, respiración ausente. |

**Ensemble de 3 señales** (todas generalizan a voces nuevas), combinadas por stacking/promedio
ponderado **calibrado** (Platt/isotónica) para un `confidence` significativo:

- **A. Conversacional** — 45 features del CSV → XGBoost. Backbone (comportamiento, no identidad).
- **B. Audio anti-spoofing** — Wav2Vec2-AASIST frozen + prosodia sobre canal 0.
- **C. Semántico / LLM zero-shot** — transcripción + análisis LLM (cadencia robótica, prompt leakage,
  reacción a preguntas trampa; canal 1 da contexto). Degrada a neutro (0.5) si falla.

---

## Recursos de cómputo (asignación de máquinas)

Hay **dos GPUs NVIDIA locales** → el entrenamiento de audio se hace en casa; la **GPU de Vultr no hace falta**
(Vultr solo hospeda el endpoint, opcional).

| Máquina | Specs | Rol |
|---|---|---|
| **1 — Jesús** | i7-1255U, 16 GB, Intel Iris Xe (sin CUDA), ~54 GB libres | **Camino A** (CPU): tabular, semántico, ensemble, API, orquestación. Flasheo/dev de la Pi 5. |
| **2 — Emilio** | i5-13420H, 16 GB, **RTX 4050 6 GB**, ~152 GB libres | **Camino B** (GPU dev): extracción SSL, entrenar cabeza anti-spoofing, export ONNX. |
| **3 — Alonso (QTROCIOUS2)** | i5-13450HX, **24 GB**, **RTX 5050 8 GB**, ~100 GB libres | **Camino C (hardware)** + **GPU worker**: dueño del Centinel Altur (Pi 5); su máquina lanza corridas pesadas (fine-tune/LoRA, augmentation, pre-train, batch inference) coordinando con B. |

VRAM: 6–8 GB alcanzan para **SSL frozen + cabeza** y **LoRA** (`fp16`, batches chicos); fine-tune completo
de wav2vec2-XLS-R (~300M) va mejor en la RTX 5050. Datasets externos grandes → Máquina 2 o 3, no la 1.

---

## Los 3 caminos (resumen)

Independientes, unidos por **contratos de interfaz** (cada quien programa contra un *stub* de los demás).

**Contratos (acordar en la hora 1):**
- **B entrega** `src/features/audio.py`: `audio_score(wave, sr) -> float` y `audio_features(wave, sr) -> dict`; cache `.npy`; export **ONNX**.
- **A entrega** el ensemble y `POST /detect`; consume B (y C) como caja negra.
- **C consume** el pipeline exportado (ONNX de B + tabular de A) y el endpoint de A; entrega el dispositivo físico.
- Común: `anon_id`, canal 0 = caller. Mientras B no esté, A y C usan stub (score 0.5).

Detalle de tareas por persona en [`EQUIPO.md`](EQUIPO.md).

---

## Estructura de archivos

```
Altur/
├── PLAN.md            ← este archivo
├── EQUIPO.md          ← asignación por persona
├── notebooks/{01_eda, 02_baseline_tabular, 03_audio_model}.ipynb
├── src/
│   ├── config.py
│   ├── features/{conversational.py, audio.py, semantic.py}   # audio.py = contrato Camino B
│   ├── train_tabular.py         # Camino A
│   ├── ensemble.py              # Camino A
│   ├── api/{main.py, inference.py, Dockerfile}   # Camino A
│   ├── sponsors/{gcp.py, snowflake.py, tigergraph.py}         # opcional, aislado
│   └── edge/{capture.py, infer_onnx.py, gpio_ui.py}           # Camino C — Pi 5
├── models/            # joblib + onnx (gitignore / LFS)
└── requirements.txt
```

## Verificación (end-to-end)

1. `unzip -o altur-challenge-audio.zip` → `ls audio | wc -l` = 353.
2. **A**: `python src/train_tabular.py` reporta AUC/EER/Brier en `val` calibrado; test de **paridad** de features (WAV+turns ≈ fila del CSV).
3. **B**: `audio_score` de modelo HF zero-shot > azar en `val`; cabeza entrenada mejora AUC.
4. Ensemble: cada señal añadida no empeora AUC vs baseline A.
5. **API**: contenedor arriba, POST con WAV de `val` base64 → JSON válido < 5 s; validar sobre las 71 de `val`.
6. **Edge (bonus)**: Pi clasifica un WAV reproducido en vivo y enciende LED correcto; medir latencia.

## Orden recomendado

**Hora 1 (juntos):** setup, EDA rápido, acordar contratos y stubs.
**En paralelo:** A1→A4 (baseline + API con stub) ‖ B0→B3 (score audio real) ‖ C (GPU worker + prep Pi contra stub).
**Merge:** enchufar B en el ensemble de A → calibración → evaluar `val` una vez → C integra el pipeline final.
**Bonus:** sponsors (Gemini/Vultr), add-ons del Centinel, demo FHE.

---

### Fuentes (estado del arte)
- [iWAX: Wav2vec-AASIST-XGBoost (Sci. Reports 2025)](https://www.nature.com/articles/s41598-025-24361-5)
- [Towards Scalable AASIST (frozen wav2vec2, datos limitados)](https://arxiv.org/html/2507.11777)
- [Generalizable deepfake detection via meta-learned LoRA](https://arxiv.org/pdf/2502.10838)
- [Audio Anti-Spoofing Detection: A Survey](https://arxiv.org/html/2404.13914v1)
- [Off-the-shelf HuggingFace models for audio deepfake detection (PyData 2024)](https://pydata.org/global2024/schedule/talk/LJPSKA/)
