# -*- coding: utf-8 -*-
import os
import sys
import glob
import numpy as np
import soundfile as sf
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score, roc_curve
import joblib

ROOT = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'scripts'))
from src.features.telephony_aug import augment_chain
from train_head_bd import embed_wave

AUG_CACHE = os.path.join(ROOT, 'models', 'ssl_cache', 'aug')
os.makedirs(AUG_CACHE, exist_ok=True)

TRAIN_CODECS = ['g711_ulaw', 'g711_alaw', 'opus']


MAX_SECONDS = 30


def load_mono(path):
    w, sr = sf.read(path)
    c = (w[:, 0] if w.ndim > 1 else w).astype(np.float32)
    return c[:sr * MAX_SECONDS], sr


def embed_aug(path, codec, tag):
    key = os.path.join(AUG_CACHE, f'{tag}__{codec}__{os.path.basename(path)}.npy')
    if os.path.exists(key):
        return np.load(key)
    c, sr = load_mono(path)
    proc, psr = augment_chain(c, sr, codec=codec)
    v = embed_wave(proc, psr)
    np.save(key, v)
    return v


def embed_plain(path, tag):
    key = os.path.join(AUG_CACHE, f'plain__{tag}__{os.path.basename(path)}.npy')
    if os.path.exists(key):
        return np.load(key)
    c, sr = load_mono(path)
    v = embed_wave(c, sr)
    np.save(key, v)
    return v


def aug_files(files, label, tag):
    X, y = [], []
    for f in files:
        for codec in TRAIN_CODECS:
            X.append(embed_aug(f, codec, tag))
            y.append(label)
    return X, y


def eer_thr(y, s):
    fpr, tpr, th = roc_curve(y, s)
    j = np.nanargmin(np.abs((1 - tpr) - fpr))
    return (fpr[j] + (1 - tpr)[j]) / 2, th[j]


def main():
    d = np.load(os.path.join(ROOT, 'models', 'ssl_cache', 'wavlm_base_plus_dataset.npz'), allow_pickle=True)
    ids, ya, spa = d['ids'], d['y'].astype(int), d['split']
    altur_audio = os.path.join(ROOT, 'hackmty26', 'audio')

    print('Extrayendo features con augmentation multi-codec (train)...', flush=True)
    X, y = [], []

    for i, aid in enumerate(ids):
        if spa[i] != 'train':
            continue
        Xa, _ = aug_files([os.path.join(altur_audio, str(aid) + '.wav')], int(ya[i]), 'altur')
        X += Xa
        y += [int(ya[i])] * len(Xa)

    for f in sorted(glob.glob(os.path.join(ROOT, 'datasets_externos', 'Human_ES_train_8kHz', '*.wav'))):
        Xh, yh = aug_files([f], 0, 'hes_tr')
        X += Xh; y += yh
    for d2 in ['Synthetic_MX_8kHz']:
        for f in sorted(glob.glob(os.path.join(ROOT, 'datasets_externos', d2, '*.wav'))):
            Xs, ys = aug_files([f], 1, 'smx')
            X += Xs; y += ys
    for f in sorted(glob.glob(os.path.join(ROOT, 'datasets_externos', 'Synthetic_Multi_8kHz', 'edge_*.wav'))):
        Xs, ys = aug_files([f], 1, 'smulti'); X += Xs; y += ys
    for f in sorted(glob.glob(os.path.join(ROOT, 'datasets_externos', 'Synthetic_Multi_8kHz', 'gtts_*.wav'))):
        Xs, ys = aug_files([f], 1, 'smulti'); X += Xs; y += ys

    X = np.stack(X); y = np.array(y).astype(int)
    print(f'TRAIN aug total={len(y)} (humanos={(y==0).sum()} sinteticos={(y==1).sum()}) codecs={TRAIN_CODECS}', flush=True)

    sc = StandardScaler().fit(X)
    clf = CalibratedClassifierCV(LogisticRegression(max_iter=3000, C=0.1, class_weight='balanced'), method='isotonic', cv=5)
    clf.fit(sc.transform(X), y)

    def P(files, tag):
        V = np.stack([embed_plain(f, tag) for f in files])
        return clf.predict_proba(sc.transform(V))[:, 1]

    va_ids = [str(a) for a, s in zip(ids, spa) if s == 'val']
    va_y = np.array([int(yy) for yy, s in zip(ya, spa) if s == 'val'])
    va_s = P([os.path.join(altur_audio, a + '.wav') for a in va_ids], 'altur')

    he = P(sorted(glob.glob(os.path.join(ROOT, 'datasets_externos', 'Human_ES_test_8kHz', '*.wav'))), 'hes_te')
    em = P(sorted(glob.glob(os.path.join(ROOT, 'datasets_externos', 'Human_MX_8kHz', 'human_emilio_*.wav'))), 'hemilio')
    pi = P(sorted(glob.glob(os.path.join(ROOT, 'datasets_externos', 'Synthetic_Piper_8kHz', '*.wav'))), 'piper')

    hum = np.r_[he, em]
    yx = np.r_[np.zeros(len(hum)), np.ones(len(pi))]
    sx = np.r_[hum, pi]
    _, t = eer_thr(yx, sx)
    print('\n=== EVAL held-out (AUG) ===')
    print(f'  Altur val AUC={roc_auc_score(va_y, va_s):.3f}')
    print(f'  humanos ok@0.5={(hum<.5).mean()*100:.0f}%  media={hum.mean():.3f}  (FLEURS {he.mean():.3f} / emilio {em.mean():.3f})')
    print(f'  Piper AUC={roc_auc_score(yx, sx):.3f}  @0.5={(pi>=.5).mean()*100:.0f}%  @EER({t:.2f})={(pi>=t).mean()*100:.0f}%  media={pi.mean():.3f}')

    out = os.path.join(ROOT, 'src', 'models', 'saved', 'wavlm_head_aug.joblib')
    joblib.dump({'scaler': sc, 'clf': clf, 'model_id': 'microsoft/wavlm-base-plus', 'dim': X.shape[1]}, out)
    print('\nGuardado', out)


if __name__ == '__main__':
    main()
