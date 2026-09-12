# -*- coding: utf-8 -*-
import os
import torch
import torchaudio
import numpy as np
from transformers import AutoModelForAudioClassification, AutoFeatureExtractor

_MODEL_ID = os.environ.get("AUDIO_HF_MODEL", "mo-thecreator/Deepfake-audio-detection")
_TARGET_SR = 16000
_WINDOW_S = 10
_MAX_WINDOWS = 6

_DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
_MODEL = None
_EXTRACTOR = None
_FAKE_IDX = None


def _load():
    global _MODEL, _EXTRACTOR, _FAKE_IDX
    if _MODEL is None:
        _EXTRACTOR = AutoFeatureExtractor.from_pretrained(_MODEL_ID)
        _MODEL = AutoModelForAudioClassification.from_pretrained(_MODEL_ID).to(_DEVICE).eval()
        _FAKE_IDX = 0
        for i, lab in _MODEL.config.id2label.items():
            l = str(lab).lower()
            if any(k in l for k in ('fake', 'spoof', 'synth', 'clone', 'ai')):
                _FAKE_IDX = int(i)
                break
    return _MODEL, _EXTRACTOR, _FAKE_IDX


def zeroshot_score(caller_wave: np.ndarray, sr: int) -> float:
    model, extractor, fake_idx = _load()

    if caller_wave.ndim > 1:
        caller_wave = caller_wave[0]

    wave = torch.from_numpy(np.ascontiguousarray(caller_wave)).float().unsqueeze(0)

    if sr != _TARGET_SR:
        wave = torchaudio.transforms.Resample(orig_freq=sr, new_freq=_TARGET_SR)(wave)

    wave = wave.squeeze(0)
    win = _TARGET_SR * _WINDOW_S
    total = wave.shape[0]

    if total <= win:
        chunks = [wave]
    else:
        n = min(_MAX_WINDOWS, total // win)
        starts = np.linspace(0, total - win, n).astype(int)
        chunks = [wave[s:s + win] for s in starts]

    probs = []
    with torch.no_grad():
        for ch in chunks:
            inputs = extractor(ch.numpy(), sampling_rate=_TARGET_SR, return_tensors='pt')
            inputs = {k: v.to(_DEVICE) for k, v in inputs.items()}
            logits = model(**inputs).logits
            p = torch.softmax(logits, dim=-1)[0, fake_idx].item()
            probs.append(p)

    return float(np.mean(probs))
