# -*- coding: utf-8 -*-
import os
import io
import sys
import argparse
import subprocess
import numpy as np
import soundfile as sf
import imageio_ffmpeg

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from src.features.telephony_aug import augment_chain
from features.audio import audio_score
from features.heavy_audio import xlsr_sls_score, flow_llr_score
from features.prosody import prosody_features

WEIGHTS = {'wavlm': 0.25, 'xlsr': 0.4, 'prosody': 0.25, 'flow': 0.1}
FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()
_SAVED = os.path.join(os.path.dirname(__file__), '..', 'src', 'models', 'saved')
try:
    import joblib
    _PROS = joblib.load(os.path.join(_SAVED, 'prosody_clf.joblib'))
except Exception:
    _PROS = None


def load_any(path):
    try:
        w, sr = sf.read(path, always_2d=True, dtype='float32')
        return w, sr
    except Exception:
        pass
    p = subprocess.run([FFMPEG, '-hide_banner', '-loglevel', 'error', '-i', path,
                        '-ac', '1', '-ar', '16000', '-f', 'wav', 'pipe:1'],
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if p.returncode != 0:
        raise RuntimeError(p.stderr.decode('utf-8', 'ignore')[:200])
    w, sr = sf.read(io.BytesIO(p.stdout), always_2d=True, dtype='float32')
    return w, sr


def combine(sig):
    live = {k: v for k, v in sig.items() if v is not None and v != 0.5}
    if not live:
        return 0.5
    tw = sum(WEIGHTS.get(k, 0.2) for k in live)
    return float(sum(live[k] * WEIGHTS.get(k, 0.2) for k in live) / tw)


def bar(p, n=30):
    fill = int(round(p * n))
    return '[' + '#' * fill + '-' * (n - fill) + ']'


def main():
    ap = argparse.ArgumentParser(description='Analiza un audio: HUMANO vs IA')
    ap.add_argument('audio', help='ruta al audio (wav/mp3/m4a/...)')
    ap.add_argument('--threshold', type=float, default=0.2)
    ap.add_argument('--no-codec', action='store_true', help='no aplicar canal telefonico g711')
    args = ap.parse_args()

    if not os.path.exists(args.audio):
        print('No existe el archivo:', args.audio)
        return

    w, sr = load_any(args.audio)
    caller = w[:, 0]
    dur = len(caller) / sr

    if args.no_codec:
        proc, psr = caller.astype(np.float32), sr
    else:
        proc, psr = augment_chain(caller.astype(np.float32), sr, codec='g711_ulaw')

    win = psr * 6
    if proc.shape[0] <= win:
        chunks = [proc]
    else:
        n = int(np.ceil(proc.shape[0] / win))
        chunks = [proc[i * win:(i + 1) * win] for i in range(n)]
        chunks = [c for c in chunks if c.shape[0] >= psr * 2]

    per = []
    for c in chunks:
        s = {'wavlm': float(audio_score(c, psr)),
             'xlsr': float(xlsr_sls_score(c, psr)),
             'flow': float(flow_llr_score(c, psr))}
        per.append(s)
    sig = {k: float(np.mean([p[k] for p in per])) for k in ('wavlm', 'xlsr', 'flow')}
    bio = prosody_features(proc, psr)
    if _PROS is not None:
        import numpy as _np
        xb = _np.array([[bio[k] for k in _PROS['keys']]])
        sig['prosody'] = float(_PROS['model'].predict_proba(xb)[0, 1])
    final = combine(sig)

    veredicto = 'IA / SINTETICO' if final >= args.threshold else 'HUMANO'
    print()
    print('=' * 52)
    print(f'  Archivo : {os.path.basename(args.audio)}  ({dur:.1f}s, {w.shape[1]}ch, {sr}Hz)')
    if dur < 5:
        print('  [!] AVISO: audio muy corto (<5s) -> resultado poco fiable. Usa >=8s.')
    print(f'  Canal   : {"crudo" if args.no_codec else "telefonico g711 (8kHz)"}')
    print('=' * 52)
    print(f'  VEREDICTO:  >>> {veredicto} <<<')
    print(f'  Prob. sintetico: {final*100:5.1f}%   {bar(final)}')
    print(f'  (umbral {args.threshold})')
    print('-' * 52)
    print('  Senales de audio (prob. sintetico):')
    for k in ('wavlm', 'xlsr', 'prosody', 'flow'):
        if k in sig:
            print(f'    {k:8s} (w={WEIGHTS[k]}): {sig[k]*100:5.1f}%  {bar(sig[k], 20)}')
    print('-' * 52)
    print('  Prosodia (humano suele: HNR bajo, shimmer alto):')
    print(f'    HNR={bio.get("hnr",0):.1f}dB  shimmer={bio.get("shimmer_local",0):.3f}  '
          f'jitter={bio.get("jitter_local",0):.3f}  f0={bio.get("f0_mean",0):.0f}Hz')
    print('=' * 52)
    print()


if __name__ == '__main__':
    main()
