# -*- coding: utf-8 -*-
import os
import numpy as np
import torch
import torchaudio

_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SAVED = os.path.join(_DIR, 'models', 'saved')
_DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
_SR = 16000
_WIN = _SR * 20
_MAXW = 3
_XLSR_ID = 'facebook/wav2vec2-xls-r-300m'
_WAVLM_ID = 'microsoft/wavlm-base-plus'

_xlsr_fe = None
_xlsr = None
_sls = None
_wavlm_fe = None
_wavlm = None
_flow = None


def _to_16k(wave, sr):
    if wave.ndim > 1:
        wave = wave[0]
    w = torch.from_numpy(np.ascontiguousarray(wave)).float()
    if sr != _SR:
        w = torchaudio.transforms.Resample(sr, _SR)(w.unsqueeze(0)).squeeze(0)
    return w


def _windows(w):
    total = w.shape[0]
    if total <= _WIN:
        return [w] if total >= _SR else [torch.nn.functional.pad(w, (0, _SR - total))]
    starts = np.linspace(0, total - _WIN, _MAXW).astype(int)
    return [w[s:s + _WIN] for s in starts]


def _load_xlsr():
    global _xlsr_fe, _xlsr, _sls
    if _sls is None:
        from transformers import AutoFeatureExtractor, Wav2Vec2Model
        from models.xlsr_sls import XLSRSLS
        _xlsr_fe = AutoFeatureExtractor.from_pretrained(_XLSR_ID)
        _xlsr = Wav2Vec2Model.from_pretrained(_XLSR_ID, use_safetensors=True, output_hidden_states=True).to(_DEVICE).eval()
        net = XLSRSLS().to(_DEVICE)
        ckpt = torch.load(os.path.join(_SAVED, 'xlsr_sls.pt'), map_location=_DEVICE, weights_only=True)
        net.load_state_dict(ckpt['state_dict'])
        net.eval()
        _sls = net
    return _xlsr_fe, _xlsr, _sls


@torch.no_grad()
def xlsr_sls_score(caller_wave: np.ndarray, sr: int) -> float:
    try:
        fe, xlsr, sls = _load_xlsr()
        w = _to_16k(caller_wave, sr)
        accs = []
        for ch in _windows(w):
            inp = fe(ch.numpy(), sampling_rate=_SR, return_tensors='pt').input_values.to(_DEVICE)
            hs = xlsr(inp).hidden_states
            accs.append(torch.stack([h.squeeze(0).mean(0) for h in hs]).cpu().numpy())
        feat = torch.tensor(np.mean(accs, axis=0)[None], device=_DEVICE)
        return float(sls.prob(feat).item())
    except Exception as e:
        print(f'[heavy] xlsr_sls no disponible ({str(e)[:80]})')
        return 0.5


def flow_llr_score(caller_wave: np.ndarray, sr: int) -> float:
    global _flow
    try:
        import joblib
        from features.wavlm_embed import embed
        if _flow is None:
            _flow = joblib.load(os.path.join(_SAVED, 'flow_llr.joblib'))
        v = embed(caller_wave, sr).reshape(1, -1)
        return float(_flow.predict_proba(v)[0, 1])
    except Exception as e:
        print(f'[heavy] flow_llr no disponible ({str(e)[:80]})')
        return 0.5
