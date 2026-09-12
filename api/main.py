from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from api.inference import predict

app = FastAPI(title="Altur Voice Spoofing Detector")


class DetectRequest(BaseModel):
    audio_base64: str


class DetectResponse(BaseModel):
    is_synthetic: bool
    confidence: float


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/detect", response_model=DetectResponse)
def detect(payload: DetectRequest):
    try:
        result = predict(payload.audio_base64)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return result
