import logging
import os

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from api.inference import predict, predict_detailed, decode_wav, warmup, heavy_available, InvalidAudioError
from api.dataset import router as dataset_router
from api.live import router as live_router

logger = logging.getLogger("altur.detect")

app = FastAPI(title="Altur Voice Spoofing Detector")

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.environ.get("ALTUR_CORS_ORIGINS", "http://localhost:3000").split(","),
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(dataset_router)
app.include_router(live_router)


class DetectRequest(BaseModel):
    audio_base64: str = Field(..., min_length=1)


class DetectResponse(BaseModel):
    is_synthetic: bool
    confidence: float


@app.on_event("startup")
def _warmup():
    if os.environ.get("ALTUR_WARMUP", "1") != "0":
        warmup(async_=True)


@app.get("/health")
def health():
    return {"status": "ok", "heavy": heavy_available()}


@app.post("/detect", response_model=DetectResponse)
def detect(payload: DetectRequest):
    try:
        return predict(payload.audio_base64)
    except InvalidAudioError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        logger.exception("error interno en /detect")
        raise HTTPException(status_code=500, detail="error interno procesando el audio")


@app.post("/detect/detailed")
def detect_detailed(payload: DetectRequest):
    try:
        data, sr = decode_wav(payload.audio_base64)
        return predict_detailed(data, sr)
    except InvalidAudioError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception:
        logger.exception("error interno en /detect/detailed")
        raise HTTPException(status_code=500, detail="error interno procesando el audio")


@app.post("/detectar", response_model=DetectResponse)
def detectar(payload: DetectRequest):
    try:
        return predict(payload.audio_base64)
    except InvalidAudioError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception:
        logger.exception("error interno en /detectar")
        raise HTTPException(status_code=500, detail="error interno procesando el audio")


_STATIC = os.path.join(os.path.dirname(__file__), "static")
if os.path.isdir(_STATIC):
    from fastapi.staticfiles import StaticFiles
    app.mount("/console", StaticFiles(directory=_STATIC, html=True), name="console")
