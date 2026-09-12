import os

import pandas as pd
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

DATA_DIR = os.environ.get('ALTUR_DATA_DIR', 'hackmty26')
MANIFEST = os.path.join(DATA_DIR, 'manifest.csv')
AUDIO_DIR = os.path.join(DATA_DIR, 'audio')

router = APIRouter(prefix='/dataset', tags=['dataset'])

_manifest = None


def _load():
    global _manifest
    if _manifest is None:
        _manifest = pd.read_csv(MANIFEST)
    return _manifest


def _row(anon_id):
    df = _load()
    hit = df[df['anon_id'] == anon_id]
    if hit.empty:
        raise HTTPException(status_code=404, detail='llamada no encontrada')
    return hit.iloc[0]


@router.get('/calls')
def list_calls(split: str | None = None, limit: int = 400):
    df = _load()
    if split:
        df = df[df['split'] == split]
    available = set(os.listdir(AUDIO_DIR)) if os.path.isdir(AUDIO_DIR) else set()
    rows = []
    for r in df.head(limit).itertuples(index=False):
        if f'{r.anon_id}.wav' in available:
            rows.append({'anon_id': r.anon_id, 'split': r.split, 'duration_s': float(r.duration_s)})
    return {'calls': rows}


@router.get('/calls/{anon_id}/audio')
def call_audio(anon_id: str):
    _row(anon_id)
    path = os.path.join(AUDIO_DIR, f'{anon_id}.wav')
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail='audio no disponible (descomprime altur-challenge-audio.zip)')
    return FileResponse(path, media_type='audio/wav', filename=f'{anon_id}.wav')


@router.get('/calls/{anon_id}/label')
def call_label(anon_id: str):
    r = _row(anon_id)
    return {'anon_id': anon_id, 'label': r['label']}
