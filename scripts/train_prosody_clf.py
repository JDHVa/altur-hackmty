# -*- coding: utf-8 -*-
import os
import sys
import glob
import numpy as np
import soundfile as sf
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score

ROOT = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, os.path.join(ROOT, 'src'))
from features.prosody import prosody_features, _KEYS

HUMAN_DIRS = ['datasets_externos/Human_ES_train_8kHz', 'datasets_externos/Human_ES_test_8kHz',
              'datasets_externos/Human_MX_8kHz']
SYNTH_DIRS = ['datasets_externos/Synthetic_MX_8kHz', 'datasets_externos/Synthetic_Multi_8kHz',
              'datasets_externos/Synthetic_Piper_8kHz']


def collect(dirs, label):
    X, y = [], []
    for d in dirs:
        for f in sorted(glob.glob(os.path.join(ROOT, d, '*.wav'))):
            w, sr = sf.read(f)
            c = (w[:, 0] if w.ndim > 1 else w).astype(np.float32)
            fe = prosody_features(c, sr)
            X.append([fe[k] for k in _KEYS])
            y.append(label)
    return X, y


def main():
    Xh, yh = collect(HUMAN_DIRS, 0)
    Xs, ys = collect(SYNTH_DIRS, 1)
    X = np.array(Xh + Xs)
    y = np.array(yh + ys)
    print(f'humanos={len(yh)} sinteticos={len(ys)} features={_KEYS}')

    idx = np.arange(len(y))
    rng = np.random.RandomState(0)
    rng.shuffle(idx)
    cut = int(0.8 * len(idx))
    tr, te = idx[:cut], idx[cut:]

    pipe = Pipeline([('sc', StandardScaler()),
                     ('m', CalibratedClassifierCV(LogisticRegression(max_iter=2000, class_weight='balanced'),
                                                  method='isotonic', cv=3))])
    pipe.fit(X[tr], y[tr])
    p = pipe.predict_proba(X[te])[:, 1]
    print(f'prosody clf holdout AUC={roc_auc_score(y[te], p):.3f}')

    pipe.fit(X, y)
    joblib.dump({'model': pipe, 'keys': _KEYS}, os.path.join(ROOT, 'src', 'models', 'saved', 'prosody_clf.joblib'))
    print('Guardado src/models/saved/prosody_clf.joblib')


if __name__ == '__main__':
    main()
