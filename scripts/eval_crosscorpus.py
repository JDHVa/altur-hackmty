# -*- coding: utf-8 -*-
import os
import sys
import glob
import numpy as np
import soundfile as sf
from sklearn.metrics import roc_auc_score, roc_curve

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from features.audio import audio_score

HUMAN_DIR = 'datasets_externos/Human_MX_8kHz'
SYNTH_DIR = 'datasets_externos/Synthetic_MX_8kHz'


def load(f):
    w, sr = sf.read(f)
    return (w[:, 0] if w.ndim > 1 else w).astype(np.float32), sr


def main():
    hum = sorted(glob.glob(os.path.join(HUMAN_DIR, '*.wav')))
    syn = sorted(glob.glob(os.path.join(SYNTH_DIR, '*.wav')))
    if not hum or not syn:
        print('Faltan datos. Humanos:', len(hum), 'Sinteticos:', len(syn))
        return
    hs = [audio_score(*load(f)) for f in hum]
    sy = [audio_score(*load(f)) for f in syn]
    ys = np.array([0] * len(hs) + [1] * len(sy))
    ss = np.array(hs + sy)
    fpr, tpr, _ = roc_curve(ys, ss)
    fnr = 1 - tpr
    i = np.nanargmin(np.abs(fnr - fpr))
    print(f'CROSS-CORPUS ({len(hs)} humanos + {len(sy)} sinteticos)')
    print(f'  humanos    media={np.mean(hs):.3f}  mal marcados sintetico: {(np.array(hs) >= .5).sum()}/{len(hs)}')
    print(f'  sinteticos media={np.mean(sy):.3f}  bien cazados: {(np.array(sy) >= .5).sum()}/{len(sy)}')
    print(f'  AUC={roc_auc_score(ys, ss):.3f}  EER={(fpr[i] + fnr[i]) / 2:.3f}  ACC@0.5={((ss >= .5).astype(int) == ys).mean():.3f}')


if __name__ == '__main__':
    main()
