# -*- coding: utf-8 -*-
import os
import numpy as np
import torch
import torchaudio

_BACKEND = os.environ.get('STT_BACKEND', 'auto')
_WHISPER_SIZE = os.environ.get('WHISPER_SIZE', 'large-v3')
_PARAKEET_ID = 'nvidia/parakeet-tdt-0.6b-v3'
_SR = 16000

_whisper = None
_parakeet = None
_active = None


def _to_16k_mono(wave, sr):
    if wave.ndim > 1:
        wave = wave[0]
    w = torch.from_numpy(np.ascontiguousarray(wave)).float()
    if sr != _SR:
        w = torchaudio.transforms.Resample(sr, _SR)(w.unsqueeze(0)).squeeze(0)
    return w.numpy().astype(np.float32)


def _try_parakeet():
    global _parakeet
    if _parakeet is None:
        import nemo.collections.asr as nemo_asr
        _parakeet = nemo_asr.models.ASRModel.from_pretrained(_PARAKEET_ID)
        _parakeet.eval()
    return _parakeet


def _get_whisper():
    global _whisper
    if _whisper is None:
        from faster_whisper import WhisperModel
        try:
            _whisper = WhisperModel(_WHISPER_SIZE, device='cuda', compute_type='float16')
        except Exception:
            _whisper = WhisperModel(_WHISPER_SIZE, device='cpu', compute_type='int8')
    return _whisper


def _parakeet_transcribe(x16):
    m = _try_parakeet()
    out = m.transcribe([x16], batch_size=1)
    hyp = out[0]
    text = hyp.text if hasattr(hyp, 'text') else (hyp[0] if isinstance(hyp, (list, tuple)) else str(hyp))
    return text, []


def _whisper_transcribe(x16):
    m = _get_whisper()
    segments, _info = m.transcribe(x16, language='es', beam_size=1, word_timestamps=True)
    text_parts, words = [], []
    for seg in segments:
        text_parts.append(seg.text)
        for w in (seg.words or []):
            words.append({'word': w.word, 'start': w.start, 'end': w.end})
    return ' '.join(text_parts).strip(), words


def transcribe(caller_wave: np.ndarray, sr: int) -> dict:
    global _active
    x16 = _to_16k_mono(caller_wave, sr)
    dur = len(x16) / _SR
    text, words, backend = '', [], 'none'

    order = []
    if _BACKEND in ('auto', 'parakeet'):
        order.append('parakeet')
    if _BACKEND in ('auto', 'whisper'):
        order.append('whisper')

    for b in order:
        try:
            if b == 'parakeet':
                text, words = _parakeet_transcribe(x16)
            else:
                text, words = _whisper_transcribe(x16)
            backend = b
            _active = b
            break
        except Exception as e:
            print(f'[STT] backend {b} no disponible ({str(e)[:80]})')
            continue

    n_words = len(text.split())
    return {
        'text': text,
        'words': words,
        'duration_s': float(dur),
        'n_words': n_words,
        'speaking_rate_wps': float(n_words / dur) if dur > 0 else 0.0,
        'backend': backend,
    }
