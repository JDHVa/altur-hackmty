# Despliegue — Modal (GPU) + Hugging Face Space (consola)

Arquitectura:

```
navegador → HF Space (consola, CPU gratis) → Modal /detect (GPU) → {is_synthetic, confidence}
```

## 1) Backend GPU en Modal

Requisitos: cuenta en modal.com.

```bash
pip install modal
modal token new                       # autentica (abre el navegador)
# ejecutar DESDE LA RAIZ del repo (para que suba la carpeta altur_100/):
modal deploy altur_100/modal_serve.py
```

Al terminar imprime la URL pública, algo como:
`https://<tu-workspace>--altur-100-detect-web.modal.run`

Pruébalo:

```bash
curl https://<tu-workspace>--altur-100-detect-web.modal.run/health
# consola directa de Modal:  .../console/
```

Contrato del reto:
`POST /detect`  body `{"audio_base64": "<wav base64>"}` → `{"is_synthetic": bool, "confidence": float}`

Notas:
- GPU `A10G`, se apaga sola a los 5 min sin uso (`scaledown_window=300`); pagas por uso.
- El primer request descarga XLS-R + WavLM de Hugging Face (se cachean en un Volume `altur-hf-cache`; los siguientes arranques son rápidos).
- No se instala `webrtcvad` a propósito: así el VAD es el mismo (energía) con el que se entrenó el modelo.

## 2) Consola pública en Hugging Face Space

1. En huggingface.co crea un **Space** nuevo con **SDK = Static**.
2. Sube los dos archivos de `altur_100/hf_space/`: `index.html` y `README.md`
   (o haz `git push` al repo del Space).
3. Edita en `index.html` la línea `window.API_BASE = "..."` y pon tu URL de Modal.
4. Abre la Space: la consola ya le pega al backend GPU.

(Alternativa: si no quieres HF, la consola ya vive en `.../console/` del propio Modal.)

## Reentrenar el modelo (opcional)

```bash
python altur_100/train.py            # reconstruye altur_ab.joblib desde hackmty26/
```
