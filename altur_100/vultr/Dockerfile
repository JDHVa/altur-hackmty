FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime

RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg git && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN pip install --no-cache-dir \
    transformers soundfile scikit-learn "fastapi" "uvicorn[standard]" \
    pydantic pandas numpy joblib imageio-ffmpeg

COPY pipeline.py serve.py ./
COPY src ./src

ENV HF_HOME=/app/.hfcache
ENV PYTHONUNBUFFERED=1
EXPOSE 80

CMD ["uvicorn", "serve:app", "--host", "0.0.0.0", "--port", "80"]
