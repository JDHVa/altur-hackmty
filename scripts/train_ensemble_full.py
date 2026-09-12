# -*- coding: utf-8 -*-
import os
import sys
import numpy as np
import pandas as pd
import joblib
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score, roc_curve, brier_score_loss, accuracy_score

ROOT = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, ROOT)
from src.models.xlsr_sls import XLSRSLS

SAVED = os.path.join(ROOT, 'src', 'models', 'saved')
XLSR_CACHE = os.path.join(ROOT, 'models', 'ssl_cache', 'xlsr_layers')
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def eer_thr(y, s):
    fpr, tpr, th = roc_curve(y, s)
    j = np.nanargmin(np.abs((1 - tpr) - fpr))
    return (fpr[j] + (1 - tpr)[j]) / 2, th[j]


def load_xlsr_net():
    net = XLSRSLS().to(DEVICE)
    ckpt = torch.load(os.path.join(SAVED, 'xlsr_sls.pt'), map_location=DEVICE, weights_only=True)
    net.load_state_dict(ckpt['state_dict'])
    return net.eval()


def main():
    df = pd.read_csv(os.path.join(ROOT, 'models', 'altur_features.csv'))
    feat_cols = [c for c in df.columns if c not in ('label', 'split', 'anon_id')]
    beh = joblib.load(os.path.join(SAVED, 'prob_qda.joblib'))
    p_beh = beh['model'].predict_proba(df[feat_cols].values)[:, 1]

    net = load_xlsr_net()
    p_xlsr = np.full(len(df), 0.5)
    for i, aid in enumerate(df['anon_id'].values):
        f = os.path.join(XLSR_CACHE, f'altur__{aid}.wav.npy')
        if os.path.exists(f):
            with torch.no_grad():
                v = torch.tensor(np.load(f)[None], device=DEVICE)
                p_xlsr[i] = float(net.prob(v).item())

    y = df['label'].values
    tr = df['split'].values == 'train'
    va = ~tr
    Z = np.column_stack([p_beh, p_xlsr])

    stk = LogisticRegression(max_iter=2000, C=1.0).fit(Z[tr], y[tr])
    raw_tr = stk.predict_proba(Z[tr])[:, 1]
    cal = IsotonicRegression(out_of_bounds='clip').fit(raw_tr, y[tr])

    def P(Zx):
        return cal.transform(stk.predict_proba(Zx)[:, 1])

    sva = P(Z[va])
    e, t = eer_thr(y[va], sva)
    if not (0.05 < t < 0.95):
        hi = sva[y[va] == 0].max() if (y[va] == 0).any() else 0.4
        lo = sva[y[va] == 1].min() if (y[va] == 1).any() else 0.6
        t = float(np.clip((hi + lo) / 2, 0.3, 0.6))
    print('=== Ensemble full (behavioral QDA + XLS-R-SLS) en Altur val ===')
    for name, s in [('behavioral', p_beh[va]), ('xlsr_sls', p_xlsr[va]), ('ENSEMBLE', sva)]:
        auc = roc_auc_score(y[va], s)
        acc = accuracy_score(y[va], (s >= (t if name == 'ENSEMBLE' else 0.5)).astype(int))
        print(f'  {name:12s} AUC={auc:.3f} Brier={brier_score_loss(y[va], s):.3f} ACC={acc:.3f}')
    print(f'  umbral EER={t:.3f} (EER={e:.3f})')

    out = {'stacker': stk, 'calibrator': cal, 'behavioral_path': 'src/models/saved/prob_qda.joblib',
           'feature_cols': feat_cols, 'signals': ['behavioral', 'xlsr_sls'], 'threshold': float(t)}
    joblib.dump(out, os.path.join(SAVED, 'ensemble_full.joblib'))
    print('\nGuardado src/models/saved/ensemble_full.joblib')


if __name__ == '__main__':
    main()
