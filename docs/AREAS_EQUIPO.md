# Centinela Altur — 4 áreas para dominar en el pitch (una por integrante)

> Basado en `altur_100/`. Cada área: qué es, tecnologías, lo que debes poder explicar en 30 s, preguntas típicas del juez con respuesta, y dónde está en el código.
> Sugerencia de reparto: Jesús → Área 3 y 4 · Emilio → Área 2 · Alonso → Área 1 (ajusten según quién lo construyó).

---

## Área 1 — Señal A · Conversacional (cómo *se comporta* la llamada)

**Qué es.** 42 features numéricas calculadas a partir de los turnos de habla de cada canal (quien llama vs agente): latencias de respuesta, solapes, silencios, tiempo hasta el primer turno del que llama, ratios de habla, número y duración de turnos.

**Tecnologías.** VAD (webrtcvad, fallback por energía) · NumPy/pandas · `turns_from_audio` → `extract_features_from_turns`.

**Explícalo en 30 s.** "Un bot responde siempre con la misma latencia, nunca interrumpe y no deja silencios naturales. Convertimos eso en 42 números por llamada. No usamos la identidad de la voz, por eso generaliza a hablantes nuevos."

**Preguntas del juez.**
- *¿Por qué no usar reconocimiento de hablante?* → Codifica identidad; el set oculto tiene voces nuevas y sobreajustaría. Las señales conversacionales son independientes del hablante.
- *¿Qué pasa si el audio es mono?* → Funciona: el canal del agente se trata como silencio y las features de latencia se degradan, pero el sistema no falla.
- *¿Por qué VAD por energía en despliegue?* → Es el mismo VAD con el que se entrenó el modelo; cambiarlo movería la distribución de features.

**Código.** `altur_100/src/features/conversational.py`, `pipeline.py::feature_vector`.

---

## Área 2 — Señal B · Acústica (cómo *suena* la voz)

**Qué es.** Tres detectores de artefactos de síntesis sobre el canal del que llama, todos construidos sobre modelos de voz auto-supervisados **congelados**.

**Tecnologías.**
- **XLS-R-SLS**: `facebook/wav2vec2-xls-r-300m` (SSL multilingüe, 300 M parámetros) → 25 capas → *Sensitive Layer Selection* (peso aprendido por capa) → MLP. Señal más fuerte.
- **WavLM head**: `microsoft/wavlm-base-plus` → mean/std del último estado → cabeza sklearn.
- **Flow-LLR**: normalizing flows (zuko, NSF) por clase sobre embeddings WavLM → razón de verosimilitud. Apagado en la versión rápida.
- Ventanas de 6 s promediadas (XLS-R: 3 ventanas en modo rápido, ~2.7× más veloz).

**Explícalo en 30 s.** "Los modelos SSL ya saben de voz; nosotros no los reentrenamos, solo ponemos una cabeza chica encima. Así aprenden a ver artefactos de síntesis y no a memorizar la voz de la persona. XLS-R con selección de capas es el que mejor separa; WavLM y el flow son segundas opiniones."

**Preguntas del juez.**
- *¿Por qué congelar los modelos?* → Dataset chico (353 llamadas): fine-tune completo sobreajusta a los hablantes; congelado + cabeza pequeña generaliza.
- *¿Qué es la selección de capas?* → Cada capa del SSL codifica cosas distintas (fonética, prosodia, canal); aprendemos un peso por capa para quedarnos con las que discriminan real vs sintético.
- *¿Corre en CPU?* → Sí, pero lento (XLS-R 300M); por eso el despliegue es en GPU y existe la versión rápida.

**Código.** `altur_100/src/features/heavy_audio.py`, `src/features/audio.py`, `src/models/xlsr_sls.py`, `src/models/flow_llr.py`.

---

## Área 3 — Fusión, entrenamiento y evaluación (por qué confiar en el número)

**Qué es.** Un clasificador único que combina las 42 features conversacionales + 3 señales de audio y produce `p_synthetic`; umbral 0.5.

**Tecnologías.** `HistGradientBoostingClassifier` (scikit-learn; `max_iter=400, learning_rate=0.05, max_depth=3, l2=1.0`) · `StratifiedKFold` 5×5 semillas · holdout de 71 llamadas · AUC, accuracy · `feats_cache.csv` para reentrenar rápido.

**Explícalo en 30 s.** "No promediamos a ojo: un gradient boosting aprende cuánto pesa cada señal. Lo validamos de tres formas: split oficial (100 %), validación cruzada 5×5 (99.3 % ± 0.2) y un holdout que nunca vio (98.6 %). La cifra honesta para voces nuevas es la de CV."

**Preguntas del juez.**
- *¿100 % no es sobreajuste?* → Por eso reportamos CV y holdout; el modelo desplegado se entrenó con todo (353) para máxima data, pero la cifra realista es ~99.3 %.
- *¿Por qué boosting y no una red?* → Datos tabulares chicos: árboles con regularización ganan, entrenan en segundos y son estables entre semillas (±0.23).
- *¿Qué significa `confidence`?* → Probabilidad del modelo; `is_synthetic = p ≥ 0.5`, `confidence = p` o `1−p` según la clase.

**Código.** `altur_100/train.py`, `train_holdout.py`, `pipeline.py::predict`, `holdout_test.csv`.

---

## Área 4 — API y despliegue (que funcione a la hora del juez)

**Qué es.** El contrato del reto en producción: `POST /detect {audio_base64} → {is_synthetic, confidence}`, más una consola web para la demo.

**Tecnologías.** FastAPI + uvicorn + Pydantic · Base64/WAV (soundfile, ffmpeg como fallback de formato) · Docker (`pytorch/pytorch:2.3.0-cuda12.1`) · **Modal** (GPU A10G, pago por uso, Volume para pesos, escala a cero) · **Vultr Cloud GPU** (24/7, sin cold start, ~1–2 s) · **Hugging Face Space** (consola estática) · caché de pesos HF.

**Explícalo en 30 s.** "Es una carpeta autocontenida con sus pesos: un `docker run` y está arriba. La tenemos en Modal por costo, en Vultr para la demo sin cold start, y la consola pública en Hugging Face. El endpoint respeta el contrato exacto del reto."

**Preguntas del juez.**
- *¿Cuánto tarda?* → ~1–2 s por llamada en GPU caliente; el primer arranque descarga XLS-R y WavLM (se cachean).
- *¿Qué pasa si mando un mp3?* → ffmpeg lo convierte a WAV PCM antes de inferir.
- *¿Por qué dos nubes?* → Modal escala a cero (barato) pero tiene cold start; Vultr está siempre encendido para la hora del juez.

**Código.** `altur_100/serve.py`, `modal_serve.py`, `vultr/` (Dockerfile, docker-compose.yml, setup.sh), `hf_space/index.html`, `DEPLOY.md`.

---

## Regla para todos
Cada quien domina su área **y** sabe decir en una frase las otras tres. La frase común: *"Miramos cómo se comporta la llamada y cómo suena la voz, lo fusionamos con boosting y lo servimos en GPU con el contrato del reto."*
