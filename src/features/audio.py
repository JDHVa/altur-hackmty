# -*- coding: utf-8 -*-
import os
import sys
import numpy as np
import torch
import torchaudio
import joblib

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.audio_model import VoiceSpoofResNet

_DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
_BACKEND = os.environ.get('AUDIO_BACKEND', 'wavlm')
_WAVLM_ID = 'microsoft/wavlm-base-plus'
_TARGET_SR = 16000
_WINDOW_S = 20
_MAX_WINDOWS = int(os.environ.get('AUDIO_MAX_WINDOWS', '3'))
_HEAD_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models', 'saved', 'wavlm_head.joblib')

_RESNET = None
_WAVLM = None
_FE = None
_HEAD = None


def _get_head():
    global _HEAD
    if _HEAD is None:
        _HEAD = joblib.load(_HEAD_PATH)
    return _HEAD


def _get_wavlm():
    global _WAVLM, _FE
    if _WAVLM is None:
        from transformers import AutoFeatureExtractor, WavLMModel
        _FE = AutoFeatureExtractor.from_pretrained(_WAVLM_ID)
        _WAVLM = WavLMModel.from_pretrained(_WAVLM_ID, use_safetensors=True).to(_DEVICE).eval()
    return _FE, _WAVLM


def _to_16k(caller_wave, sr):
    if caller_wave.ndim > 1:
        caller_wave = caller_wave[0]
    w = torch.from_numpy(np.ascontiguousarray(caller_wave)).float()
    if sr != _TARGET_SR:
        w = torchaudio.transforms.Resample(sr, _TARGET_SR)(w.unsqueeze(0)).squeeze(0)
    return w


@torch.no_grad()
def _embed_wavlm(caller_wave, sr):
    fe, m = _get_wavlm()
    w = _to_16k(caller_wave, sr)
    win = _TARGET_SR * _WINDOW_S
    total = w.shape[0]
    if total <= win:
        starts = [0]
    else:
        n = min(_MAX_WINDOWS, total // win + 1)
        starts = np.linspace(0, max(0, total - win), n).astype(int).tolist()
    vecs = []
    for s in starts:
        ch = w[s:s + win]
        if ch.shape[0] < _TARGET_SR:
            continue
        inp = fe(ch.numpy(), sampling_rate=_TARGET_SR, return_tensors='pt').input_values.to(_DEVICE)
        h = m(inp).last_hidden_state.squeeze(0)
        vecs.append(torch.cat([h.mean(0), h.std(0)]).cpu().numpy())
    if not vecs:
        inp = fe(w.numpy(), sampling_rate=_TARGET_SR, return_tensors='pt').input_values.to(_DEVICE)
        h = m(inp).last_hidden_state.squeeze(0)
        vecs.append(torch.cat([h.mean(0), h.std(0)]).cpu().numpy())
    return np.mean(vecs, axis=0).astype(np.float32)


def _chunks(caller_wave, sr, secs=6, maxc=8):
    if caller_wave.ndim > 1:
        caller_wave = caller_wave[0]
    win = int(sr * secs)
    n = caller_wave.shape[0]
    if n <= win:
        return [caller_wave]
    k = min(maxc, n // win + 1)
    starts = np.linspace(0, n - win, k).astype(int)
    return [caller_wave[s:s + win] for s in starts]


def _score_wavlm(caller_wave, sr):
    head = _get_head()
    from features.wavlm_embed import embed
    ps = []
    for ch in _chunks(caller_wave, sr):
        v = embed(ch, sr)
        ps.append(head['clf'].predict_proba(head['scaler'].transform(v.reshape(1, -1)))[0, 1])
    return float(np.mean(ps))


def _get_resnet():
    global _RESNET
    if _RESNET is None:
        model = VoiceSpoofResNet(pretrained=False).to(_DEVICE)
        model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models', 'saved', 'best_audio_resnet.pth')
        state = torch.load(model_path, map_location=_DEVICE, weights_only=True)
        if 'resnet.fc.weight' in state and 'resnet.fc.1.weight' not in state:
            state['resnet.fc.1.weight'] = state.pop('resnet.fc.weight')
            state['resnet.fc.1.bias'] = state.pop('resnet.fc.bias')
        model.load_state_dict(state)
        model.eval()
        _RESNET = model
    return _RESNET


def _score_resnet(caller_wave, sr):
    model = _get_resnet()
    wave_tensor = _to_16k(caller_wave, sr).unsqueeze(0)
    max_frames = _TARGET_SR * 8
    if wave_tensor.shape[1] > max_frames:
        wave_tensor = wave_tensor[:, :max_frames]
    else:
        wave_tensor = torch.nn.functional.pad(wave_tensor, (0, max_frames - wave_tensor.shape[1]))
    with torch.no_grad():
        out = model(wave_tensor.unsqueeze(0).to(_DEVICE)).squeeze(1)
        return float(torch.sigmoid(out).item())


def audio_score(caller_wave: np.ndarray, sr: int) -> float:
    if _BACKEND == 'wavlm':
        try:
            return _score_wavlm(caller_wave, sr)
        except Exception as e:
            print(f"[B-Camino] WavLM no disponible ({e}); fallback ResNet")
    try:
        return _score_resnet(caller_wave, sr)
    except Exception as e:
        print(f"[B-Camino] ResNet no disponible ({e}); fallback stub 0.5")
        return 0.5


def audio_features(caller_wave: np.ndarray, sr: int) -> dict:
    if caller_wave.ndim > 1:
        caller_wave = caller_wave[0]

    wave = torch.from_numpy(np.ascontiguousarray(caller_wave)).float()

    if sr != 16000:
        wave = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)(wave.unsqueeze(0)).squeeze(0)
    sr = 16000

    frame_len, hop = 400, 160
    n = 1 + max(0, (wave.shape[0] - frame_len) // hop)
    if n < 1:
        n = 1
    frames = torch.stack([wave[i * hop:i * hop + frame_len] for i in range(n)]) if wave.shape[0] >= frame_len else wave.unsqueeze(0)

    rms = torch.sqrt(torch.mean(frames ** 2, dim=1) + 1e-10)
    signs = torch.sign(frames)
    zcr = torch.mean(torch.abs(signs[:, 1:] - signs[:, :-1]), dim=1) / 2.0

    n_fft = 1024
    spec = torchaudio.transforms.Spectrogram(n_fft=n_fft, hop_length=512)(wave)
    freqs = torch.linspace(0, sr / 2, spec.shape[0]).unsqueeze(1)
    mag_sum = torch.sum(spec, dim=0) + 1e-10
    centroid = torch.sum(freqs * spec, dim=0) / mag_sum

    f0 = torchaudio.functional.detect_pitch_frequency(wave, sr)
    voiced = (f0 >= 70) & (f0 <= 400)
    valid_f0 = f0[voiced]

    features = {
        'rms_mean': float(rms.mean()),
        'rms_std': float(rms.std()),
        'zcr_mean': float(zcr.mean()),
        'spectral_centroid_mean': float(centroid.mean()),
        'f0_mean': float(valid_f0.mean()) if valid_f0.numel() > 0 else 0.0,
        'f0_std': float(valid_f0.std()) if valid_f0.numel() > 1 else 0.0,
        'voiced_ratio': float(voiced.float().mean()) if f0.numel() > 0 else 0.0
    }

    return features
