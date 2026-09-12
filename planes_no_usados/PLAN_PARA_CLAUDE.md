# Plan Maestro de Implementación de ML para Claude

> **Instrucciones para el Usuario:** Copia el contenido a partir de la siguiente línea y envíaselo a Claude en tu interfaz de chat o añádelo como un archivo `INSTRUCCIONES_ML.md` en el repositorio y dile a Claude que lo lea.

---

**Rol:** Eres un Ingeniero Experto en Machine Learning y Procesamiento de Señales de Audio.
**Objetivo:** Implementar la arquitectura de detección de voz sintética para el Altur Challenge (HackMTY 2026).
**Contexto del Problema:** Clasificar llamadas telefónicas (canal 0 = caller) como "humano" o "sintético". El audio es de baja calidad (8kHz, narrowband, estéreo). Solo tenemos 353 llamadas originales, pero estamos descargando datasets externos (ASVspoof 2019, MLAAD, WaveFake) en `datasets_externos/`.

## 🏗️ Arquitectura Requerida: El "Triple Ensemble"

No vamos a construir un sistema monolítico (es propenso a overfit con 353 llamadas). Tu tarea es implementar un **Ensemble de 3 Pipelines** que convergen en una regresión logística final.

### Pipeline 1: Tabular Conversacional (Alta Prioridad - Quick Win)
Hemos descubierto que los JSONs de turnos (`hackmty26/turns/`) contienen señales brutales sin tocar el audio:
- Los humanos interrumpen 2.6x más que las IAs.
- Latencia mínima humana = 0.51s vs Sintética = 1.58s.
- Las micro-utterances (<0.5s) son el doble en humanos.

**Tu tarea:**
1. Escribe un script de extracción de features tabulares leyendo los JSONs. Features a extraer: `n_overlaps`, `avg_latency`, `min_latency`, `short_utterances_count`, `variance_turn_duration`, `caller_agent_speak_ratio`.
2. Entrena un clasificador **XGBoost** o **LightGBM** usando estas features.

### Pipeline 2: SSL Acoustic Embeddings
El audio a 8kHz destruye características espectrales clásicas (MFCCs). Usaremos Transfer Learning.

**Tu tarea:**
1. Configura un pipeline usando `torchaudio` y `transformers`.
2. Carga un modelo SSL pre-entrenado (ej. `microsoft/wavlm-base-plus` o el encoder de `openai/whisper-small`).
3. Haz un script que:
   - Resamplee los audios de 8kHz a 16kHz (requerido por modelos SSL).
   - Extraiga SOLO los segmentos del canal 0 (usando los timestamps del JSON) para evitar que el agente contamine el embedding.
   - Extraiga el embedding promedio por llamada.
4. Entrena un **SVM (RBF kernel)** o un MLP pequeño sobre estos embeddings.

### Pipeline 3: Pre-entrenamiento Externo y Fine-Tuning
Tenemos datasets en la carpeta `datasets_externos/` (ASVspoof, WaveFake, MLAAD).

**Tu tarea:**
1. Construye un `DataLoader` de PyTorch que pueda ingerir audios de `datasets_externos/` aplicando "Telephone Simulation Augmentation" (downsample a 8kHz, bandpass filter 300-3400Hz, adición de ruido gaussiano).
2. Opcional/Recomendado: Fine-tunear un modelo tipo RawNet2 o un clasificador ligero sobre los embeddings SSL utilizando primero los datasets externos y haciendo *transfer learning* en la etapa final con los 353 audios de Altur.

### Pipeline 4: Endpoint y Ensemble
**Tu tarea:**
1. Escribe el archivo `app.py` usando **FastAPI**.
2. Define el endpoint `POST /detect` que reciba el WAV estéreo en Base64.
3. El endpoint debe:
   - Separar canales y extraer turnos al vuelo (usa algo simple como Silero VAD para inferencia si no nos dan el JSON).
   - Pasar los datos por el Modelo Tabular (P1) y el Modelo Acústico (P2).
   - Combinar las probabilidades con un regresor logístico calibrado (Platt Scaling) para retornar `is_synthetic` y `confidence`.

## 📂 Estructura de Archivos Esperada

Crea los siguientes archivos modulares, no hagas un solo script gigante:
- `src/features/extract_conversational.py` (Procesamiento de JSONs)
- `src/features/extract_embeddings.py` (WavLM/Whisper embeddings)
- `src/models/train_xgb.py` (Entrenamiento Pipeline 1)
- `src/models/train_acoustic.py` (Entrenamiento Pipeline 2)
- `src/api/app.py` (FastAPI Endpoint)

## ⚡ Reglas de Desarrollo para Claude
1. **Evita el código repetitivo:** Escribe clases/funciones utilitarias para la carga de audios y extracción de canales.
2. **GPU Conscious:** Asegúrate de que el código mueva explícitamente los tensores a `cuda` si `torch.cuda.is_available()`.
3. **Manejo de Errores en la API:** El endpoint debe ser a prueba de balas (validar base64, validar sample rate, validar canales).
4. Empecemos por la **Pipeline 1 (Tabular)**. Genera el código para extraer las features del JSON y entrenar el XGBoost. Cuando termines, dímelo y avanzaremos a la Pipeline 2.
