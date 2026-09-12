# EQUIPO — Asignación de tareas por persona

> Complementa [`PLAN.md`](PLAN.md). Tres caminos **independientes** unidos por contratos de
> interfaz. Cada quien programa contra un *stub* de los demás y no se bloquea.

## Contratos de interfaz (acordar en la HORA 1, todos juntos)

- **B → todos:** `src/features/audio.py` con
  `audio_score(caller_wave: np.ndarray, sr: int) -> float` (prob. sintético 0–1) y
  `audio_features(caller_wave, sr) -> dict`. Cache `.npy` + export **ONNX**.
- **A → todos:** `src/ensemble.py` + endpoint `POST /detect`. Consume B (y C) como caja negra.
- **C → consume:** ONNX de B + `models/tabular.joblib` de A + endpoint de A.
- **Común:** `anon_id`, canal 0 = caller, canal 1 = agente. Stub de audio = `0.5` mientras B no exporte.

---

## 👤 Persona 1 — JESÚS · Camino A (Conversacional + Semántico + Ensemble + API)
**Máquina:** i7-1255U, 16 GB, sin CUDA (CPU). **Dueño del entregable que puntúa.**

Estado: ✅ hecho · 🟡 en progreso/parcial · ⬜ pendiente

| Estado | # | Tarea | Entregable |
|---|---|---|---|
| 🟡 | A0 | Setup + EDA | Hecho: `requirements.txt`, robustez de features y métricas EER/AUC/Brier. Falta notebook formal `01_eda.ipynb` |
| ✅ | A1 | **Backbone tabular** (45 features → LightGBM) | `src/features/conversational.py` (paridad EXACTA vs CSV), `src/training/train_tabular.py`, `lgbm_tabular.pkl`. VAL AUC 0.99 |
| ⬜ | A2 | Señal C — semántico zero-shot | `src/features/semantic.py`: `faster-whisper` + LLM. **Bloqueado por decisión: Gemini (API key) vs LLM local** |
| ✅ | A3 | **Ensemble + calibración** | `src/ensemble.py`: calibración isotónica OOF + umbral por EER; pipeline VAD-consistente. Fusión A+B se activa sola cuando B dé scores |
| ✅ | A4 | **API `POST /detect`** | `api/main.py` + `api/inference.py` + `Dockerfile`. Probado HTTP end-to-end en val: acc 0.93, confianza calibrada |
| ⬜ | A5 | Sponsors (bonus, aislado) | `src/sponsors/`: Gemini y Vultr primero; Snowflake/TigerGraph solo si sobra |

**Prioridad:** A0→A1→A4→A3 ✅ hechos. Falta A2 (necesita tu decisión de LLM) y A5 (bonus).

---

## 👤 Persona 2 — EMILIO · Camino B (Audio Anti-spoofing)
**Máquina:** RTX 4050 6 GB. **Dueño de la señal de audio SOTA.**

| Estado | # | Tarea | Entregable |
|---|---|---|---|
| 🟡 | B0 | **Score zero-shot inmediato** (modelo HF pre-entrenado) | `src/features/audio_zeroshot.py` ya existe (mo-thecreator/Deepfake-audio-detection). Falta **evaluarlo en val e integrarlo** al ensemble |
| 🟡 | B1 | Features SSL frozen + prosodia | `src/features/audio.py` (contrato `audio_score`/`audio_features`) existe. Prosodia con `librosa` |
| ⬜ | B2 | **Cabeza anti-spoofing** (entrenar en RTX 4050) | Sin pesos entrenados aún (`best_audio_resnet.pth` no existe). Augmentation es-MX→teléfono→8 kHz |
| ⬜ | B3 | **Export ONNX INT8** | `src/models/export_onnx.py` existe; falta el modelo real que exportar |

**Prioridad:** B0 el primer día (número real para el ensemble). Luego B1→B2→B3.
**Coordina** con Alonso el uso de la RTX 5050 para fine-tune/LoRA/pre-train.

---

## 👤 Persona 3 — ALONSO · Camino C (Hardware "Centinela Altur") + GPU Worker
**Máquina:** RTX 5050 8 GB, 24 GB RAM. **Dueña del dispositivo físico y de las corridas pesadas.**

### Rol GPU worker (arranca aquí, mientras B madura)
- Lanzar y monitorear en la RTX 5050: **fine-tune completo / LoRA**, **augmentation** (TTS→teléfono→8 kHz),
  **pre-train** con datasets externos (ASVspoof/WaveFake), **batch inference** de los 353 audios.
- Guardar datasets externos grandes aquí (disco/VRAM), entregar checkpoints a Emilio.

### Rol hardware — Centinela Altur (Raspberry Pi 5 4 GB)
Dispositivo de banca antifraude que **delata voz sintética en vivo**. Corre el **mismo pipeline** (paridad con `/detect`).

| # | Tarea | Entregable |
|---|---|---|
| C0 | Preparar Pi 5 contra **stub** | SO + ONNX runtime `arm64` + mic/VAD (`webrtcvad`) + salida (LED/pantalla). Todo con score stub 0.5 |
| C1 | **Modo Reto Adversarial** | La Pi hace de agente: pregunta trampa por bocina → escucha → mide **latencia en vivo** → veredicto |
| C2 | Visualización | Anillo LED verde↔rojo, medidor de "humanidad", espectrograma en vivo, desglose 3 señales, TTS (`piper`) del veredicto |
| C3 | Integrar modelo real | Enchufar ONNX de B + tabular de A. Fallback **modo offload** (Pi = captura+UI, modelo pesado en RTX 5050 por HTTP) |
| C4 | Add-ons (si sobra) | "Teléfono trucado", botón espejo `/detect` (borde vs nube), contador de fraudes |

**Componentes:** Pi 5 4 GB, ReSpeaker/mic USB, microSD A2, pantalla (LCD/HDMI), anillo LED, bocina, disipador + ventilador.
**Framing:** el hardware es **bonus** — no debe robar tiempo al endpoint que puntúa. Empezar el hardware
en firme cuando B1–B3 estén estables.

---

## Ruta crítica (qué desbloquea qué)

```
Hora 1 (todos): setup + EDA + contratos + stubs
   │
   ├─ A: A1 baseline ──► A4 API (stub B=0.5) ──► A2 ──► A3 ensemble ──► eval val
   ├─ B: B0 zero-shot ──► B1 ──► B2 (usa RTX 5050 de P3) ──► B3 ONNX ──┐
   └─ C: GPU worker (corre B2/pesados) + C0 prep Pi (stub) ───────────┘
                                                                       ▼
                                          MERGE: B→ensemble de A · C integra ONNX+tabular
```

**Regla de oro:** el endpoint `POST /detect` con **solo la señal A** ya es entregable válido.
Todo lo demás (audio, semántico, hardware, sponsors) suma pero **nunca bloquea** ese mínimo.
