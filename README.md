# Centinel Altur — HackMTY 2026

> **La carpeta principal del proyecto es [`altur_100/`](altur_100/).**
> Ahí vive el sistema final: el detector de voz humana vs. IA para llamadas Altur,
> autocontenido y desplegado. Todo lo demás en el repo es contexto, datos o piezas de apoyo.

Detección de **voz sintética (IA) vs. humano** en llamadas telefónicas a un banco (español mexicano),
expuesta como `POST /detect` y como una consola web en tiempo real.

---

## `altur_100/` — el sistema principal

Modelo **A+B**: fusiona señal **conversacional** (timing de turnos por VAD) + señal de **audio**
(XLS-R, WavLM, flow) con un clasificador **HistGradientBoosting**.

- Accuracy en Altur: **~99% (CV 5-fold)**, 100% en el split de validación oficial.
- Enfocado 100% en la distribución de Altur (el juez evalúa con datos similares).

### Estructura de `altur_100/`
```
altur_100/
├── pipeline.py        extracción de features + predict / explain / audio_only
├── serve.py           FastAPI local (endpoints + consola web + push al Pi)
├── modal_serve.py     despliegue GPU en Modal
├── train.py           reentrena el modelo A+B desde hackmty26/
├── static/index.html  consola web (subir · grabar · en vivo, tema claro/oscuro)
├── hf_space/          consola para Hugging Face Space (apunta al backend de Modal)
├── vultr/             despliegue alternativo (GPU siempre encendida)
└── src/               código + pesos del modelo (features/, models/, models/saved/)
```

### Correr en local
```bash
cd altur_100
python -m uvicorn serve:app --port 8010
# consola:  http://127.0.0.1:8010/console/
```

### Endpoints
| Endpoint | Qué hace |
|---|---|
| `POST /detect` | Contrato del reto: `{audio_base64}` → `{is_synthetic, confidence}`. |
| `POST /detect/detailed` | Igual + `p_final`, `p_audio`, `signals`, `turns`, `duration_s`. |
| `POST /detect/audio` | Solo-audio, rápido (para el modo **en vivo**). |
| `POST /explain` | Transcribe (Whisper) + razones + score sintético por segmento. |
| `GET /console/` | Consola web del operador. |

### En línea (desplegado)
- **Backend GPU (Modal):** `https://emilioyt929394--altur-100-detect-web.modal.run`
- **Consola pública (HF Space):** `https://elelimios-centinela-altur.static.hf.space`

Reentrenar: `python altur_100/train.py` (usa `hackmty26/`).

---

## `hardware/` — Centinel físico (opcional, solo local)

Raspberry Pi 5 con pantalla TFT (ILI9341) + matriz LED + bocina que **refleja el veredicto**
del sistema web. La compu hace toda la IA y le manda el resultado al Pi por WiFi.

- `centinela_display.py` (en el Pi): recibe `{state}` y muestra **HUMANO** (morado) / **INTELIGENCIA ARTIFICIAL** (rojo) + sonido.
- El servidor local prende el Pi si defines `ALTUR_PI_URL=http://<IP_PI>:8080/state`.
- Es opcional y no intrusivo: sin Pi, todo funciona igual (la versión en línea no se toca).

Ver [`hardware/README.md`](hardware/README.md) para los comandos exactos.

---

## Otras carpetas (contexto / apoyo)
- `hackmty26/` — dataset del reto (audio, manifest, turns).
- `api/`, `web/` — versión anterior (FastAPI + consola Next.js); histórica.
- `src/`, `scripts/`, `planes/`, `docs/` — features, entrenamiento, planeación y documentación.

Contexto del reto y división de trabajo: [`CLAUDE.md`](CLAUDE.md), [`PLAN.md`](planes/PLAN.md), [`EQUIPO.md`](EQUIPO.md).
