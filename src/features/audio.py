# -*- coding: utf-8 -*-
import os
import torch
import torchaudio
import numpy as np
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.audio_model import VoiceSpoofResNet

_DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
_AUDIO_MODEL = None

def _get_model():
    global _AUDIO_MODEL
    if _AUDIO_MODEL is None:
        try:
            model = VoiceSpoofResNet(pretrained=False).to(_DEVICE)
            model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models', 'saved', 'best_audio_resnet.pth')
            state = torch.load(model_path, map_location=_DEVICE, weights_only=True)
            if 'resnet.fc.weight' in state and 'resnet.fc.1.weight' not in state:
                state['resnet.fc.1.weight'] = state.pop('resnet.fc.weight')
                state['resnet.fc.1.bias'] = state.pop('resnet.fc.bias')
            model.load_state_dict(state)
            model.eval()
            _AUDIO_MODEL = model
        except Exception as e:
            print(f"[B-Camino] Error cargando modelo acustico: {e}")
            return None
    return _AUDIO_MODEL

def audio_score(caller_wave: np.ndarray, sr: int) -> float:
    model = _get_model()
    if model is None:
        return 0.5

    if caller_wave.ndim > 1:
        caller_wave = caller_wave[0]

    wave_tensor = torch.from_numpy(caller_wave).float().unsqueeze(0)

    if sr != 16000:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)
        wave_tensor = resampler(wave_tensor)

    max_frames = 16000 * 8
    if wave_tensor.shape[1] > max_frames:
        wave_tensor = wave_tensor[:, :max_frames]
    else:
        padding = max_frames - wave_tensor.shape[1]
        wave_tensor = torch.nn.functional.pad(wave_tensor, (0, padding))

    with torch.no_grad():
        out = model(wave_tensor.unsqueeze(0).to(_DEVICE)).squeeze(1)
        prob = torch.sigmoid(out).item()

    return float(prob)

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
