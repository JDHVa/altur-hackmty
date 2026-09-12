import logging

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from api.inference import predict, InvalidAudioError

logger = logging.getLogger("altur.detect")

app = FastAPI(title="Altur Voice Spoofing Detector")


class DetectRequest(BaseModel):
    audio_base64: str = Field(..., min_length=1)


class DetectResponse(BaseModel):
    is_synthetic: bool
    confidence: float


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/detect", response_model=DetectResponse)
def detect(payload: DetectRequest):
    try:
        return predict(payload.audio_base64)
    except InvalidAudioError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        logger.exception("error interno en /detect")
        raise HTTPException(status_code=500, detail="error interno procesando el audio")
