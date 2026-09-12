> ✅ PLAN DEFINITIVO — CONCRETADO (no re-analizar; solo ejecutar)

# Plan de Arquitectura Definitiva y Datos (Altur HackMTY)

Este documento reemplaza planes anteriores y consolida la estrategia final basándose en la literatura científica moderna: **Machine Learning para la conversación + Deep Learning para el audio.** Se rechaza el análisis matemático clásico por su ineficacia ante vocoders neuronales y audios telefónicos de 8kHz.

---

## 1. Investigación y Estrategia de Datasets

Dado que los "artefactos" de la Inteligencia Artificial al generar voz (como errores de fase) son independientes del idioma, nos apalancaremos fuertemente en datasets en inglés para enseñar a nuestro modelo a distinguir entre Humano e IA, y luego cerraremos la brecha del idioma (Domain Gap) con datos sintéticos en español.

### Datasets a Utilizar:
1. **ASVspoof 2019/2021 (Track LA/DF) - *El Estándar de la Industria***
   - **Qué es:** Audios en inglés con docenas de sistemas TTS y de clonación distintos. El track DF incluye compresión telefónica.
   - **Por qué lo usamos:** Es el dataset más masivo. Enseñará a nuestro modelo de Deep Learning acústico a identificar las huellas genéricas que dejan los modelos generativos en el espectro, invisibles para el ojo humano.
2. **WaveFake - *Diversidad de Vocoders***
   - **Qué es:** Más de 100k audios generados con 6 arquitecturas de vocoders (MelGAN, HiFi-GAN, WaveGlow).
   - **Por qué lo usamos:** Los matemáticos fallan contra estos vocoders. Usar WaveFake obliga a nuestra red neuronal a aprender qué errores microscópicos comete cada vocoder.
3. **MLAAD (Multilingual Audio Anti-Spoofing) - *El Puente de Idioma***
   - **Qué es:** Uno de los pocos datasets con síntesis multilingüe usando arquitecturas modernas (XTTS v2).
   - **Por qué lo usamos:** Contiene español. Nos ayuda a iniciar la adaptación de nuestro modelo al idioma del reto.
4. **Custom Synthetic MX (Edge-TTS) - *Nuestra Arma Secreta***
   - **Qué es:** Un dataset que nosotros mismos vamos a programar y generar. Usaremos las voces neuronales gratuitas de Microsoft (`es-MX-JorgeNeural`, `es-MX-DaliaNeural`) aplicándoles un filtro de 8kHz (telefonía).
   - **Por qué lo usamos:** Hace "match" perfecto al 100% con las características acústicas y fonéticas de las llamadas del banco Altur.

---

## 2. Arquitectura del Modelo: El "Dual Ensemble"

La arquitectura no será monolítica, se dividirá en dos pipelines que se entrenan por separado y se unen al final, maximizando los 353 audios del challenge.

### Pipeline A: "La Dinámica" (Conversacional - XGBoost)
- **Concepto:** Analiza los tiempos, no el audio. Las máquinas piensan diferente que los humanos.
- **Entrada:** Los JSONs de turnos (`hackmty26/turns/`).
- **Features:** Ratio de interrupciones (overlaps), varianza de duración de turno, latencia mínima (Time-To-First-Token de las IAs), cantidad de micro-respuestas ("ajá", "sí").
- **Modelo:** XGBoost o LightGBM. Es inmune al ruido de 8kHz porque solo mira marcas de tiempo.

### Pipeline B: "La Acústica" (Deep Learning SSL)
- **Concepto:** Extraer semántica profunda del audio para vencer a los vocoders. No usamos matemáticas clásicas.
- **Entrada:** El WAV estéreo de 8kHz (se extrae solo el Canal 0 del cliente).
- **Extracción de Características (SSL):** Se procesa el audio usando un modelo pre-entrenado masivo (ej. **WavLM Base+** o **Whisper Encoder**). Estos modelos fueron entrenados con 90,000 horas de audio y entienden la fonética.
- **Clasificador Acústico:** Una red neuronal pequeña (MLP) o un SVM que toma el embedding de WavLM y decide Humano/Sintético, habiendo sido pre-entrenado con ASVspoof/WaveFake y fine-tuneado con nuestro Edge-TTS y los 353 audios de Altur.

### El Ensamblador (Fusión)
- Un modelo de Regresión Logística simple que toma la probabilidad de la Pipeline A y la probabilidad de la Pipeline B.
- Retorna `is_synthetic` y el `confidence` (como pide el endpoint de evaluación).

---

## 3. Fases de Ejecución del Código (Next Steps)

Para completar el trabajo pesado sin chocar, estas son las tareas que debe ejecutar el equipo (incluyendo Claude o yo):

- [ ] **Fase 1: Preparación de Datos:** 
  - Ejecutar el script generador de datos mexicanos (Edge-TTS).
  - Terminar las descargas de ASVspoof y WaveFake.
- [ ] **Fase 2: Código Pipeline A:**
  - Escribir `extract_conversational_features.py` que barra los JSONs.
  - Entrenar el XGBoost y guardar el modelo (`.pkl` o `.json`).
- [ ] **Fase 3: Código Pipeline B:**
  - Escribir el extractor de embeddings de WavLM adaptado a 8kHz.
  - Entrenar el clasificador MLP usando los datos extra.
- [ ] **Fase 4: Endpoint API:**
  - Montar el `app.py` en FastAPI que una A y B para devolver el JSON esperado.

---
*Este plan documenta la estrategia final para el hackathon. Está diseñado para ser leído por humanos y por otros agentes de IA.*
