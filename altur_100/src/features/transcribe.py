import os
import numpy as np

WHISPER_SIZE = os.environ.get('WHISPER_SIZE', 'small')
TARGET_SR = 16000
_MODEL = None


def _model():
    global _MODEL
    if _MODEL is None:
        from faster_whisper import WhisperModel
        dev = os.environ.get('WHISPER_DEVICE', 'cpu')
        ct = os.environ.get('WHISPER_CT', 'int8')
        _MODEL = WhisperModel(WHISPER_SIZE, device=dev, compute_type=ct)
    return _MODEL


def _to16k(wave, sr):
    w = np.asarray(wave, dtype=np.float32)
    if w.ndim > 1:
        w = w[:, 0]
    if sr == TARGET_SR:
        return w
    import torch
    import torchaudio
    t = torch.from_numpy(np.ascontiguousarray(w)).float().unsqueeze(0)
    t = torchaudio.transforms.Resample(sr, TARGET_SR)(t)
    return t.squeeze(0).numpy()


def transcribe_segments(caller_wave, sr):
    audio = _to16k(caller_wave, sr)
    segments, _ = _model().transcribe(audio, language='es', vad_filter=True)
    out = []
    for s in segments:
        txt = s.text.strip()
        if txt:
            out.append({'start': float(s.start), 'end': float(s.end), 'text': txt})
    return out
