# Plan Maestro de Implementación (plan-jesus-altur)

Este plan define el roadmap oficial para el proyecto (Inferencia Local CPU + Tiger Data + Encriptación Homomórfica).

### Fase 1: Ingeniería Acústica Local (CPU-Only)
*Dado que no usamos GPUs, utilizaremos transferencia de aprendizaje (Transfer Learning) con modelos pre-entrenados para procesar el audio rápida y eficientemente.*

- **Librerías a instalar:** `torch`, `torchaudio`, `speechbrain`, `transformers`.
- **Pasos Concretos:**
  1. Extraer el **Canal 0** de los audios WAV (frecuencia de muestreo 8kHz).
  2. Utilizar un modelo preentrenado **ECAPA-TDNN** (vía SpeechBrain) para convertir los audios del dataset en *Voiceprints* (embeddings densos de 192 dimensiones).
  3. Utilizar un modelo preentrenado de anti-spoofing (como **AASIST** o Wav2Vec2-Deepfake) para obtener el puntaje de "liveness" inicial.
  4. Guardar estos vectores y puntajes localmente en un formato intermedio (`.npy` o CSV) para no repetir el proceso pesado.

### Fase 2: Integración con Tiger Data (Almacenamiento Vectorial y Telemetría)
*Conectaremos nuestro pipeline a Tiger Data para simular el entorno productivo de un banco.*

- **Librerías a instalar:** `psycopg2-binary`, `sqlalchemy`, `pgvector`.
- **Pasos Concretos:**
  1. Crear la conexión a la instancia de Tiger Data.
  2. Habilitar la extensión `pgvector` o `pgvectorscale` en la base de datos.
  3. Crear una tabla de **Huellas Vocales (Voiceprints)** donde insertaremos los embeddings (192-D) calculados en la Fase 1.
  4. Crear una *Hypertable* (si Tiger Data usa Timescale) para registrar la telemetría: cada llamada procesada, su timestamp, latencias de turnos, y el score antifraude final.

### Fase 3: Prototipado de Seguridad Financiera (FHE)
*Demostrar el más alto estándar de privacidad biométrica exigido en el documento bancario.*

- **Librerías a instalar:** `tenseal`.
- **Pasos Concretos:**
  1. Generar un contexto de encriptación CKKS.
  2. Crear un script de demostración que tome un embedding de Tiger Data, lo encripte, encripte también un embedding de prueba de audio entrante, y calcule la similitud del coseno *totalmente en el dominio cifrado*.
  3. Retornar la verificación de identidad (humano legímito vs atacante) sin jamás exponer el vector biométrico en texto plano.

### Fase 4: Ensamble SASV y API de Inferencia
*Unir el análisis de turnos, los vectores de Tiger Data y lanzar el servidor.*

- **Librerías a instalar:** `xgboost`, `scikit-learn`, `fastapi`, `uvicorn`.
- **Pasos Concretos:**
  1. Entrenar un modelo **XGBoost** rápido. Los *inputs* serán las 45 características conversacionales (los silencios y overlaps del CSV) sumados a las proyecciones acústicas (scores de AASIST y distancias de Tiger Data).
  2. Programar el endpoint `POST /detect`. Al recibir un Base64:
     - El audio se procesa en tiempo real (ECAPA-TDNN).
     - (Opcional) Se consulta Tiger Data para similitud.
     - Se clasifica con XGBoost.
     - Retorna `{"is_synthetic": true, "confidence": 0.87}`.

