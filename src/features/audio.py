# -*- coding: utf-8 -*-
import os
import torch
import torchaudio
import numpy as np
import librosa
import sys

# Asegurar que el modulo src sea accesible
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.audio_model import VoiceSpoofResNet

# --- Carga perezosa (Lazy Load) del Modelo ---
_DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
_AUDIO_MODEL = None

def _get_model():
    global _AUDIO_MODEL
    if _AUDIO_MODEL is None:
        try:
            model = VoiceSpoofResNet(pretrained=False).to(_DEVICE)
            model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models', 'saved', 'best_audio_resnet.pth')
            model.load_state_dict(torch.load(model_path, map_location=_DEVICE, weights_only=True))
            model.eval()
            _AUDIO_MODEL = model
        except Exception as e:
            print(f\"[B-Camino] Error cargando modelo acustico: {e}\")
            return None
    return _AUDIO_MODEL

def audio_score(caller_wave: np.ndarray, sr: int) -> float:
    \"\"\"
    Contrato B -> Todos.
    Devuelve la probabilidad (0.0 a 1.0) de que la onda de audio pertenezca a una IA (sintetica).
    \"\"\"
    model = _get_model()
    if model is None:
        return 0.5 # Stub (Fallback en caso de que falten los pesos)
        
    # Validar forma del arreglo (debe ser 1D para un solo canal)
    if caller_wave.ndim > 1:
        caller_wave = caller_wave[0] # Tomar primer canal si pasan estereo por error
        
    # Convertir a Tensor de PyTorch y resamplear si es necesario
    wave_tensor = torch.from_numpy(caller_wave).float().unsqueeze(0) # [1, time]
    
    if sr != 16000:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)
        wave_tensor = resampler(wave_tensor)
        
    # Cortar o padear a 8 segundos
    max_frames = 16000 * 8
    if wave_tensor.shape[1] > max_frames:
        wave_tensor = wave_tensor[:, :max_frames]
    else:
        padding = max_frames - wave_tensor.shape[1]
        wave_tensor = torch.nn.functional.pad(wave_tensor, (0, padding))
        
    # Inferencia
    with torch.no_grad():
        # Modelo espera: [batch, channel, time] -> [1, 1, time]
        out = model(wave_tensor.unsqueeze(0).to(_DEVICE)).squeeze(1)
        prob = torch.sigmoid(out).item()
        
    return float(prob)

def audio_features(caller_wave: np.ndarray, sr: int) -> dict:
    \"\"\"
    Contrato B -> Todos.
    Devuelve diccionario con caracteristicas acusticas de prosodia y espectro.
    \"\"\"
    if caller_wave.ndim > 1:
        caller_wave = caller_wave[0]
        
    # Caracteristicas basicas rapidas (Prosodia y calidad de voz)
    rms = librosa.feature.rms(y=caller_wave)[0]
    zcr = librosa.feature.zero_crossing_rate(caller_wave)[0]
    spectral_centroid = librosa.feature.spectral_centroid(y=caller_wave, sr=sr)[0]
    
    # Calcular pitch (F0) rudimentario
    f0, voiced_flag, voiced_probs = librosa.pyin(caller_wave, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'), sr=sr)
    
    # Filtrar solo zonas con voz
    valid_f0 = f0[~np.isnan(f0)]
    
    features = {
        'rms_mean': float(np.mean(rms)),
        'rms_std': float(np.std(rms)),
        'zcr_mean': float(np.mean(zcr)),
        'spectral_centroid_mean': float(np.mean(spectral_centroid)),
        'f0_mean': float(np.mean(valid_f0)) if len(valid_f0) > 0 else 0.0,
        'f0_std': float(np.std(valid_f0)) if len(valid_f0) > 0 else 0.0,
        'voiced_ratio': float(np.sum(voiced_flag) / len(voiced_flag)) if len(voiced_flag) > 0 else 0.0
    }
    
    return features
