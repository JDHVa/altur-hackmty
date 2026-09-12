# -*- coding: utf-8 -*-
import subprocess
import numpy as np
import soundfile as sf
import imageio_ffmpeg

_FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()

CODECS = {
    'g711_ulaw': ['-ar', '8000', '-ac', '1', '-acodec', 'pcm_mulaw', '-f', 'wav'],
    'g711_alaw': ['-ar', '8000', '-ac', '1', '-acodec', 'pcm_alaw', '-f', 'wav'],
    'amr_nb': ['-ar', '8000', '-ac', '1', '-acodec', 'amr_nb', '-b:a', '12.2k', '-f', 'amr'],
    'opus': ['-ar', '16000', '-ac', '1', '-acodec', 'libopus', '-b:a', '24k', '-f', 'ogg'],
}


def _run_ffmpeg(in_wav_bytes, args):
    p = subprocess.run([_FFMPEG, '-hide_banner', '-loglevel', 'error', '-i', 'pipe:0',
                        *args, 'pipe:1'],
                       input=in_wav_bytes, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if p.returncode != 0:
        raise RuntimeError(p.stderr.decode('utf-8', 'ignore')[:200])
    return p.stdout


def _to_wav_bytes(wave, sr):
    import io
    buf = io.BytesIO()
    sf.write(buf, wave, sr, format='WAV', subtype='PCM_16')
    return buf.getvalue()


def apply_codec(wave: np.ndarray, sr: int, codec: str) -> tuple:
    if codec not in CODECS:
        raise ValueError(f'codec desconocido: {codec}')
    if wave.ndim > 1:
        wave = wave[0] if wave.shape[0] < wave.shape[1] else wave[:, 0]
    enc = _run_ffmpeg(_to_wav_bytes(wave, sr), CODECS[codec])
    dec = _run_ffmpeg(enc, ['-ar', '8000', '-ac', '1', '-f', 'wav'])
    import io
    out, out_sr = sf.read(io.BytesIO(dec))
    return out.astype(np.float32), out_sr


def packet_loss(wave: np.ndarray, sr: int, loss_rate: float = 0.1, burst_ms: int = 40, seed: int = 0) -> np.ndarray:
    rng = np.random.RandomState(seed)
    out = wave.copy()
    burst = max(1, int(sr * burst_ms / 1000))
    i = 0
    while i < len(out):
        if rng.rand() < loss_rate:
            out[i:i + burst] = 0.0
            i += burst
        else:
            i += burst
    return out


def add_noise(wave: np.ndarray, snr_db: float = 20.0, seed: int = 0) -> np.ndarray:
    rng = np.random.RandomState(seed)
    p_sig = np.mean(wave ** 2) + 1e-12
    p_noise = p_sig / (10 ** (snr_db / 10))
    noise = rng.normal(0, np.sqrt(p_noise), size=wave.shape).astype(np.float32)
    return (wave + noise).astype(np.float32)


def random_gain(wave: np.ndarray, min_db: float = -6.0, max_db: float = 6.0, seed: int = 0) -> np.ndarray:
    rng = np.random.RandomState(seed)
    g = 10 ** (rng.uniform(min_db, max_db) / 20)
    return np.clip(wave * g, -1.0, 1.0).astype(np.float32)


def augment_chain(wave: np.ndarray, sr: int, codec: str = 'g711_ulaw',
                  loss_rate: float = 0.0, snr_db: float = None,
                  gain: bool = False, seed: int = 0) -> tuple:
    out, out_sr = apply_codec(wave, sr, codec)
    if snr_db is not None:
        out = add_noise(out, snr_db, seed)
    if loss_rate > 0:
        out = packet_loss(out, out_sr, loss_rate, seed=seed)
    if gain:
        out = random_gain(out, seed=seed)
    return out, out_sr
