# Plan v4 — Detección robusta de voz humana vs sintética (Camino B v2)

> Plan **metodológico** para mejorar la señal de audio del reto Altur, enfocado en
> **generalizar a llamadas reales** (voces y canales no vistos). Derivado de dos
> investigaciones internas: *"Detección de IA en Bancos"* y *"Fraude Bancario con Bots Vocales"*.
> **Dueño:** Emilio (Camino B). No re-analizar señales A/C aquí; ver `plan_arquitectura_v3.md`.
>
> Estado: borrador para discutir. No lleva el marcador de plan concretado todavía.

## 1. Problema que ataca este plan

El modelo actual (`best_audio_resnet.pth`) da **VAL AUC 0.997** pero clasifica **voces humanas reales fuera de dominio como sintéticas con 0.99 de confianza** (probado con 3 grabaciones propias).

**Causa raíz (confirmada por la literatura): channel mismatch / "codec laundering".**
Los códecs telefónicos (G.711 μ-law/A-law, AMR-NB) amputan todo arriba de ~4 kHz, justo
donde viven los artefactos de vocoder. Un detector entrenado en un solo canal:
- rinde casi perfecto en su propio dominio (`val` = mismo pipeline de captura),
- colapsa ante otro canal (EER 0.83% → 16–29%; falsos rechazos de humanos 5% → 70%).

El `val` **no mide** esto porque comparte canal con `train`. El enemigo real del set
oculto es la **generalización cross-canal y cross-TTS**.

## 2. Principios de diseño (del estado del arte)

- **P1 — Augmentation de canal es la defensa central.** Entrenar y evaluar con códecs
  telefónicos reales + pérdida de paquetes + ruido/ganancia.
- **P2 — Features biológicas interpretables.** Jitter, shimmer, HNR (micro-prosodia) y
  bispectro (acoplamiento de fase cuadrático) codifican la diferencia humano/IA real y
  sobreviven mejor al códec que una CNN de magnitud pura.
- **P3 — SSL frozen generaliza.** WavLM/Wav2Vec2 congelado → robusto a TTS "zero-day" e
  independiente del idioma. Evita memorizar hablantes (dataset chico).
- **P4 — Validar cross-corpus, no solo `val`.** TTS es-MX nuevo + humanos es reales.
- **P5 — Coste asimétrico + opción de diferir.** Dejar pasar un bot cuesta mucho más que
  rechazar un humano (minDCF); permitir veredicto "incierto" cuando falta confianza.
- **P6 — El contrato nunca se rompe.** `audio_score`/`audio_features` siguen estables para
  A y C en cada fase (fallback al modelo actual si algo falla).

## 3. Fases y entregables

Cada fase deja el repo funcional y es testeable por separado.

### Fase B-A — Augmentation telefónica (fix de raíz)
| Entregable | Criterio de aceptación |
|---|---|
| `src/features/telephony_aug.py`: `apply_codec(wave, sr, codec)` para G.711 μ-law/A-law, AMR-NB, Opus, + `packet_loss()`, `add_noise()`, `random_gain()` | Cada códec corre sobre un WAV de `val` y produce audio 8 kHz válido; test unitario de forma/rango |
| `scripts/diag_codec.py`: aplica μ-law real a las 3 grabaciones propias y reporta `audio_score` | Reporte reproducible; documenta si el score se voltea a humano o no |

**Salida esperada:** confirmar cuantitativamente que el gap es de canal → justifica reentrenar con augmentation.

### Fase B-B — Features biológicas interpretables
| Entregable | Criterio de aceptación |
|---|---|
| Ampliar `audio_features` con `jitter`, `shimmer`, `hnr` (parselmouth/Praat) y estadísticos de bispectro | Corre sobre 353 audios sin depender de numba; valores en rangos plausibles |
| `notebooks/04_bio_features.ipynb` (o script): distribución humano vs sintético por feature, con GroupKFold por hablante | Al menos 1–2 features con separación estadística clara (AUC univariada > 0.6) |

**Salida esperada:** panel de features explicables para el demo ("esta voz tiene jitter natural = humano").

### Fase B-C — Dataset externo cross-corpus
| Entregable | Criterio de aceptación |
|---|---|
| TTS es-MX nuevo vía `generate_synthetic_mx.py` (Edge-TTS) pasado por augmentation telefónica | ≥ 40 clips sintéticos 8 kHz etiquetados |
| Humanos es reales (Common Voice es / grabaciones propias) pasados por augmentation | ≥ 40 clips humanos 8 kHz etiquetados |
| `models/ssl_cache/…` extendido a este set externo | `.npz` externo con X, y, fuente |

**Salida esperada:** conjunto de prueba de **generalización real** (voces y TTS no vistos).

### Fase B-D — Reentrenar cabeza con augmentation
| Entregable | Criterio de aceptación |
|---|---|
| `src/training/train_head_ssl.py`: cabeza (MLP/logística o AASIST ligera) sobre WavLM frozen, entrenada con augmentation de canal on-the-fly | Corre en RTX 4050; guarda pesos en `models/saved/` (LFS) |
| Comparativa vs ResNet actual | No empeora `val`; **mejora en el set cross-corpus (B-C)** |

**Salida esperada:** modelo que sostiene desempeño cuando cambian voz y canal.

### Fase B-E — Calibración, umbral y export
| Entregable | Criterio de aceptación |
|---|---|
| Calibración (Platt/isotónica) + umbral por coste asimétrico (minDCF) + opción "incierto" | `confidence` calibrado; umbral documentado |
| Export ONNX del modelo nuevo, con el preprocesado **dentro** del grafo (audio crudo → prob) | Paridad ONNX vs PyTorch < 1e-3; la Pi alimenta WAV 16 k sin replicar mel |
| `audio_score`/`audio_features` apuntan al modelo nuevo con fallback al anterior | Contrato estable; A y C no cambian su interfaz |

## 4. Métricas de éxito

- **Primaria:** AUC/EER en el set **cross-corpus (B-C)** (generalización real).
- **Secundaria:** minDCF con coste asimétrico (falsa aceptación de bot penalizada).
- **Guardarraíl:** las 3 grabaciones propias humanas dejan de marcarse sintético con alta confianza.
- **No-regresión:** `val` AUC no baja respecto al ResNet actual.
- **Operativa:** `audio_score` < ~1 s por llamada en CPU para el endpoint.

## 5. Dependencias e interfaces

- **Contrato B → A/C:** `audio_score(wave, sr)`, `audio_features(wave, sr)` (ver `EQUIPO.md`).
- **RTX 5050 (Alonso):** corridas pesadas (reentrenos grandes, pre-train con ASVspoof/WaveFake).
- **Datos externos:** `generate_synthetic_mx.py` (es-MX) + Common Voice es.

## 6. Riesgos

- **Augmentation irreal** → si los códecs simulados no reflejan el canal del reto, no ayuda.
  Mitigación: usar implementaciones de códec reales, no aproximaciones.
- **Sobreajuste al set externo** → validar con fuentes múltiples, no una sola.
- **Tiempo de hackathon** → B-A y B-B son de bajo costo y alto valor; B-D es el grande.
  El endpoint con el modelo actual sigue siendo entregable válido en todo momento.

## 7. Orden recomendado

`B-A (diagnóstico μ-law) → B-B (features bio) → B-C (datos externos) → B-D (reentreno) → B-E (calibración+ONNX)`
Empezar por B-A: es barato, confirma la causa raíz y produce el módulo que habilita todo lo demás.

---

### Fuentes internas
- `Detección de IA en Bancos.docx` — micro-prosodia, bispectro, AASIST/SSL, codec laundering, streaming, ASVspoof5.
- `Fraude Bancario con Bots Vocales.docx` — arquitectura de bots, latencia de turnos, fracaso de ASV, GFW/subbandas, t-DCF.
