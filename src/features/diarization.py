# -*- coding: utf-8 -*-
import os
import numpy as np
import torch

_TOKEN = os.environ.get('HF_TOKEN') or os.environ.get('HUGGINGFACE_TOKEN')
_SR = 16000
_vad = None


def available():
    if not _TOKEN:
        return False
    try:
        import pyannote.audio  # noqa
        return True
    except Exception:
        return False


def _get_vad():
    global _vad
    if _vad is None:
        from pyannote.audio import Model
        from pyannote.audio.pipelines import VoiceActivityDetection
        seg = Model.from_pretrained('pyannote/segmentation', token=_TOKEN)
        pipe = VoiceActivityDetection(segmentation=seg)
        pipe.instantiate({'onset': 0.5, 'offset': 0.5, 'min_duration_on': 0.0, 'min_duration_off': 0.0})
        if torch.cuda.is_available():
            pipe.to(torch.device('cuda'))
        _vad = pipe
    return _vad


def _resample(sig, sr):
    import torchaudio
    x = torch.from_numpy(np.ascontiguousarray(sig)).float().unsqueeze(0)
    if sr != _SR:
        x = torchaudio.transforms.Resample(sr, _SR)(x)
    return x


def _channel_turns(sig, sr, channel):
    vad = _get_vad()
    wav = _resample(sig, sr)
    ann = vad({'waveform': wav, 'sample_rate': _SR})
    turns = []
    for seg in ann.get_timeline().support():
        turns.append({'channel': channel, 'start': float(seg.start), 'end': float(seg.end)})
    return turns


def turns_from_audio_pyannote(caller_signal, agent_signal, sr):
    turns = _channel_turns(caller_signal, sr, 0) + _channel_turns(agent_signal, sr, 1)
    turns.sort(key=lambda t: t['start'])
    return turns
