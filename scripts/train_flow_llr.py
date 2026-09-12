# -*- coding: utf-8 -*-
import os
import sys
import glob
import numpy as np
import joblib
import torch
from sklearn.metrics import roc_auc_score, roc_curve

ROOT = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, ROOT)
from src.models.flow_llr import FlowLLR

EXT = os.path.join(ROOT, 'models', 'ssl_cache', 'ext')
ALT = os.path.join(ROOT, 'models', 'ssl_cache', 'wavlm_base_plus')


def load_dir(sub, tag, pat='*.wav'):
    files = sorted(glob.glob(os.path.join(ROOT, 'datasets_externos', sub, pat)))
    return np.stack([np.load(os.path.join(EXT, tag + '__' + os.path.basename(f) + '.npy')) for f in files]) if files else np.zeros((0, 1536), np.float32)


def eer(y, s):
    fpr, tpr, th = roc_curve(y, s)
    j = np.nanargmin(np.abs((1 - tpr) - fpr))
    return (fpr[j] + (1 - tpr)[j]) / 2, th[j]


def main():
    d = np.load(os.path.join(ROOT, 'models', 'ssl_cache', 'wavlm_base_plus_dataset.npz'), allow_pickle=True)
    ids, ya, spa = d['ids'], d['y'].astype(int), d['split']
    Wa = np.stack([np.load(os.path.join(ALT, str(i) + '.npy')) for i in ids])
    tr, va = spa == 'train', spa == 'val'

    Xh = load_dir('Human_ES_train_8kHz', 'hes_tr')
    Smx = load_dir('Synthetic_MX_8kHz', 'smx')
    Sed = load_dir('Synthetic_Multi_8kHz', 'smulti', 'edge_*.wav')
    Sgt = load_dir('Synthetic_Multi_8kHz', 'smulti', 'gtts_*.wav')

    X = np.concatenate([Wa[tr], Xh, Smx, Sed, Sgt])
    y = np.concatenate([ya[tr], np.zeros(len(Xh)), np.ones(len(Smx)), np.ones(len(Sed)), np.ones(len(Sgt))]).astype(int)
    print(f'TRAIN={len(y)} (h={(y==0).sum()} s={(y==1).sum()}) | entrenando 2 flows (zuko, {"cuda" if torch.cuda.is_available() else "cpu"})...', flush=True)

    m = FlowLLR(n_pca=64, epochs=300).fit(X, y)

    def P(X):
        return m.predict_proba(X)[:, 1] if len(X) else np.array([])

    Xhte = load_dir('Human_ES_test_8kHz', 'hes_te')
    Xem = load_dir('Human_MX_8kHz', 'hemilio', 'human_emilio_*.wav')
    Xpi = load_dir('Synthetic_Piper_8kHz', 'piper')
    hum = np.r_[P(Xhte), P(Xem)]; piper = P(Xpi)
    yx = np.r_[np.zeros(len(hum)), np.ones(len(piper))]; sx = np.r_[hum, piper]
    _, t = eer(yx, sx)
    print('\n=== Flow-LLR held-out ===')
    print(f'  Altur val AUC={roc_auc_score(ya[va], P(Wa[va])):.3f}')
    print(f'  humanos ok@0.5={(hum<.5).mean()*100:.0f}%  media={hum.mean():.3f}')
    print(f'  Piper AUC={roc_auc_score(yx,sx):.3f} @0.5={(piper>=.5).mean()*100:.0f}% @EER({t:.2f})={(piper>=t).mean()*100:.0f}% media={piper.mean():.3f}')

    joblib.dump(m, os.path.join(ROOT, 'src', 'models', 'saved', 'flow_llr.joblib'))
    print('\nGuardado src/models/saved/flow_llr.joblib')


if __name__ == '__main__':
    main()
