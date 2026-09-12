# -*- coding: utf-8 -*-
import os
import io
import sys
import glob
import subprocess
import numpy as np
import soundfile as sf
import imageio_ffmpeg

ROOT = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'src'))
from features.telephony_aug import augment_chain
from features.audio import audio_score
from features.heavy_audio import xlsr_sls_score, flow_llr_score
from features.prosody import prosody_features
import joblib

FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()
_PROS = joblib.load(os.path.join(ROOT, 'src', 'models', 'saved', 'prosody_clf.joblib'))


def load_any(path):
    try:
        w, sr = sf.read(path, always_2d=True, dtype='float32')
        return w, sr
    except Exception:
        p = subprocess.run([FFMPEG, '-hide_banner', '-loglevel', 'error', '-i', path,
                            '-ac', '1', '-ar', '16000', '-f', 'wav', 'pipe:1'],
                           stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        w, sr = sf.read(io.BytesIO(p.stdout), always_2d=True, dtype='float32')
        return w, sr


def chunks(c, sr, secs=6, maxc=8):
    win = int(sr * secs)
    n = len(c)
    if n <= win:
        return [c]
    k = min(maxc, n // win + 1)
    return [c[s:s + win] for s in np.linspace(0, n - win, k).astype(int)]


def main():
    rows = []
    sets = [('wav_demos/reales', 0), ('wav_demos/copias', 1)]
    for sub, label in sets:
        for f in sorted(glob.glob(os.path.join(ROOT, sub, '*'))):
            if not f.lower().endswith(('.wav', '.m4a', '.mp3')):
                continue
            w, sr = load_any(f)
            caller = w[:, 0].astype(np.float32)
            proc, psr = augment_chain(caller, sr, codec='g711_ulaw')
            per = [(audio_score(c, psr), xlsr_sls_score(c, psr), flow_llr_score(c, psr)) for c in chunks(proc, psr)]
            wv = float(np.mean([p[0] for p in per]))
            xl = float(np.mean([p[1] for p in per]))
            fl = float(np.mean([p[2] for p in per]))
            bio = prosody_features(proc, psr)
            pr = float(_PROS['model'].predict_proba(np.array([[bio[k] for k in _PROS['keys']]]))[0, 1])
            rows.append((os.path.basename(f), label, round(wv, 4), round(xl, 4), round(pr, 4), round(fl, 4)))
            print(f'{os.path.basename(f):32s} lab={label} wavlm={wv:.2f} xlsr={xl:.2f} pros={pr:.2f} flow={fl:.2f}', flush=True)

    import csv
    out = os.path.join(ROOT, 'demo_scores.csv')
    with open(out, 'w', newline='', encoding='utf-8') as fh:
        wcsv = csv.writer(fh)
        wcsv.writerow(['file', 'label', 'wavlm', 'xlsr', 'prosody', 'flow'])
        wcsv.writerows(rows)
    print(f'\nGuardado {out} ({len(rows)} audios)')


if __name__ == '__main__':
    main()
