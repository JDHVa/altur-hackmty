# -*- coding: utf-8 -*-
import os
import numpy as np
import torch
import torchaudio

_WAVLM_ID = 'microsoft/wavlm-base-plus'
_SR = 16000
_WIN = _SR * 20
_MAXW = int(os.environ.get('AUDIO_MAX_WINDOWS', '3'))
_DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

_fe = None
_model = None
_cache = {'key': None, 'val': None}


def _load():
    global _fe, _model
    if _model is None:
        from transformers import AutoFeatureExtractor, WavLMModel
        _fe = AutoFeatureExtractor.from_pretrained(_WAVLM_ID)
        _model = WavLMModel.from_pretrained(_WAVLM_ID, use_safetensors=True).to(_DEVICE).eval()
    return _fe, _model


def _key(x):
    n = x.shape[0]
    if n == 0:
        return (0,)
    return (n, float(x[0]), float(x[-1]), float(x[n // 2]))


@torch.no_grad()
def embed(caller_wave: np.ndarray, sr: int) -> np.ndarray:
    if caller_wave.ndim > 1:
        caller_wave = caller_wave[0]
    caller_wave = np.ascontiguousarray(caller_wave, dtype=np.float32)
    key = (_key(caller_wave), sr)
    if _cache['key'] == key:
        return _cache['val']

    fe, m = _load()
    w = torch.from_numpy(caller_wave).float()
    if sr != _SR:
        w = torchaudio.transforms.Resample(sr, _SR)(w.unsqueeze(0)).squeeze(0)
    total = w.shape[0]
    if total <= _WIN:
        starts = [0]
    else:
        n = min(_MAXW, total // _WIN + 1)
        starts = np.linspace(0, max(0, total - _WIN), n).astype(int).tolist()
    vecs = []
    for s in starts:
        ch = w[s:s + _WIN]
        if ch.shape[0] < _SR:
            continue
        inp = fe(ch.numpy(), sampling_rate=_SR, return_tensors='pt').input_values.to(_DEVICE)
        h = m(inp).last_hidden_state.squeeze(0)
        vecs.append(torch.cat([h.mean(0), h.std(0)]).cpu().numpy())
    if not vecs:
        inp = fe(w.numpy(), sampling_rate=_SR, return_tensors='pt').input_values.to(_DEVICE)
        h = m(inp).last_hidden_state.squeeze(0)
        vecs.append(torch.cat([h.mean(0), h.std(0)]).cpu().numpy())
    v = np.mean(vecs, axis=0).astype(np.float32)
    _cache['key'] = key
    _cache['val'] = v
    return v
