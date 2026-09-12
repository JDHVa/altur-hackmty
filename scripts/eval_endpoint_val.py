# -*- coding: utf-8 -*-
import os
import sys
import json
import numpy as np
import pandas as pd
import soundfile as sf
import joblib

ROOT = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'src'))
from features.conversational import extract_features_from_turns, FEATURE_ORDER
from features.audio import audio_score
from features.heavy_audio import xlsr_sls_score, flow_llr_score
from sklearn.metrics import roc_auc_score, roc_curve, brier_score_loss, accuracy_score

AUD = os.path.join(ROOT, 'hackmty26', 'audio')
TURNS = os.path.join(ROOT, 'hackmty26', 'turns')
WEIGHTS = {'wavlm': 0.3, 'xlsr': 0.5, 'flow': 0.2}


def eer(y, s):
    fpr, tpr, _ = roc_curve(y, s)
    fnr = 1 - tpr
    i = np.nanargmin(np.abs(fnr - fpr))
    return (fpr[i] + fnr[i]) / 2


def report(name, y, s, thr=0.5):
    print(f'  {name:16s} AUC={roc_auc_score(y, s):.3f}  EER={eer(y, s):.3f}  '
          f'Brier={brier_score_loss(y, s):.3f}  ACC@{thr}={accuracy_score(y, (np.array(s) >= thr).astype(int)):.3f}')


def main():
    beh = joblib.load(os.path.join(ROOT, 'src', 'models', 'saved', 'prob_qda.joblib'))
    ens = joblib.load(os.path.join(ROOT, 'src', 'models', 'saved', 'ensemble_full.joblib'))
    mani = pd.read_csv(os.path.join(ROOT, 'hackmty26', 'manifest.csv'))
    val = mani[mani.split == 'val'].reset_index(drop=True)

    y, pb, pw, px, pf = [], [], [], [], []
    for i, r in val.iterrows():
        aid = r['anon_id']
        w, sr = sf.read(os.path.join(AUD, aid + '.wav'))
        caller = (w[:, 0] if w.ndim > 1 else w).astype(np.float32)
        turns = json.load(open(os.path.join(TURNS, aid + '.json'), encoding='utf-8')).get('turns', [])
        feats = extract_features_from_turns(turns, r['duration_s'])
        xb = np.array([[feats[c] for c in FEATURE_ORDER]])
        pb.append(float(beh['model'].predict_proba(xb)[0, 1]))
        pw.append(float(audio_score(caller, sr)))
        px.append(float(xlsr_sls_score(caller, sr)))
        pf.append(float(flow_llr_score(caller, sr)))
        y.append(1 if r['label'] == 'synthetic' else 0)
        if (i + 1) % 20 == 0:
            print(f'[{i+1}/{len(val)}]', flush=True)

    y = np.array(y)
    pb, pw, px, pf = map(np.array, (pb, pw, px, pf))
    audio_comb = (WEIGHTS['wavlm'] * pw + WEIGHTS['xlsr'] * px + WEIGHTS['flow'] * pf) / sum(WEIGHTS.values())
    z = np.column_stack([pb, px])
    ens_final = ens['calibrator'].transform(ens['stacker'].predict_proba(z)[:, 1])

    print('\n=== VAL (71) end-to-end con scorers reales ===')
    report('behavioral', y, pb)
    report('wavlm', y, pw)
    report('xlsr_sls', y, px)
    report('flow_llr', y, pf)
    report('audio_combine', y, audio_comb)
    report('ENSEMBLE_full', y, ens_final, thr=ens['threshold'])


if __name__ == '__main__':
    main()
