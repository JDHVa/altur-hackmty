import os
import sys
import modal

app = modal.App("altur-100-detect")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg")
    .pip_install(
        "torch", "torchaudio", "torchvision", "transformers",
        "soundfile", "scikit-learn", "fastapi[standard]",
        "pydantic", "pandas", "numpy", "joblib", "imageio-ffmpeg",
        "faster-whisper",
    )
    .add_local_dir("altur_100", "/app/altur_100")
)

hf_cache = modal.Volume.from_name("altur-hf-cache", create_if_missing=True)


MIN_CONTAINERS = int(os.environ.get("ALTUR_MIN_CONTAINERS", "0"))


@app.function(image=image, gpu="A10G", volumes={"/cache": hf_cache}, timeout=600,
              scaledown_window=1200, min_containers=MIN_CONTAINERS)
@modal.asgi_app()
def web():
    import io
    import base64
    import binascii
    import subprocess
    import numpy as np
    import soundfile as sf
    from fastapi import FastAPI, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel

    os.environ["HF_HOME"] = "/cache"
    sys.path.insert(0, "/app/altur_100")
    sys.path.insert(0, "/app/altur_100/src")
    from pipeline import predict, explain, audio_only

    try:
        predict(np.zeros((16000 * 3, 2), dtype="float32"), 16000)
    except Exception:
        pass

    api = FastAPI(title="Altur 100 - A+B (GPU)")
    api.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

    class Req(BaseModel):
        audio_base64: str

    def decode(b64):
        raw = base64.b64decode(b64, validate=True)
        try:
            return sf.read(io.BytesIO(raw), always_2d=True, dtype="float32")
        except Exception:
            import imageio_ffmpeg
            ff = imageio_ffmpeg.get_ffmpeg_exe()
            p = subprocess.run([ff, "-hide_banner", "-loglevel", "error", "-i", "pipe:0", "-f", "wav", "pipe:1"],
                               input=raw, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            return sf.read(io.BytesIO(p.stdout), always_2d=True, dtype="float32")

    @api.get("/health")
    def health():
        return {"status": "ok"}

    @api.post("/detect")
    def detect(r: Req):
        try:
            data, sr = decode(r.audio_base64)
        except (binascii.Error, ValueError, RuntimeError):
            raise HTTPException(status_code=400, detail="audio invalido")
        d = predict(data, sr)
        return {"is_synthetic": d["is_synthetic"], "confidence": d["confidence"]}

    @api.post("/detect/detailed")
    def detailed(r: Req):
        try:
            data, sr = decode(r.audio_base64)
        except (binascii.Error, ValueError, RuntimeError):
            raise HTTPException(status_code=400, detail="audio invalido")
        return predict(data, sr)

    @api.post("/detect/audio")
    def detect_audio(r: Req):
        try:
            data, sr = decode(r.audio_base64)
        except (binascii.Error, ValueError, RuntimeError):
            raise HTTPException(status_code=400, detail="audio invalido")
        return audio_only(data, sr)

    @api.post("/explain")
    def explain_route(r: Req):
        try:
            data, sr = decode(r.audio_base64)
        except (binascii.Error, ValueError, RuntimeError):
            raise HTTPException(status_code=400, detail="audio invalido")
        return explain(data, sr)

    static = "/app/altur_100/static"
    if os.path.isdir(static):
        from fastapi.staticfiles import StaticFiles
        api.mount("/console", StaticFiles(directory=static, html=True), name="console")

    return api
