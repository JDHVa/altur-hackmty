# -*- coding: utf-8 -*-
import os
import sys
import glob
import argparse
import numpy as np
import soundfile as sf
import torch
import torchaudio
from transformers import AutoFeatureExtractor, WavLMModel
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score, roc_curve
import joblib

ROOT = os.path.join(os.path.dirname(__file__), '..')
MODEL_ID = 'microsoft/wavlm-base-plus'
TARGET_SR = 16000
WINDOW_S = 20
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
EXT_CACHE = os.path.join(ROOT, 'models', 'ssl_cache', 'ext')

_FE = None
_M = None


def _load_model():
    global _FE, _M
    if _M is None:
        _FE = AutoFeatureExtractor.from_pretrained(MODEL_ID)
        _M = WavLMModel.from_pretrained(MODEL_ID, use_safetensors=True).to(DEVICE).eval()
    return _FE, _M


@torch.no_grad()
def embed_wave(caller, sr):
    fe, m = _load_model()
    w = torch.from_numpy(np.ascontiguousarray(caller)).float()
    if sr != TARGET_SR:
        w = torchaudio.transforms.Resample(sr, TARGET_SR)(w.unsqueeze(0)).squeeze(0)
    win = TARGET_SR * WINDOW_S
    total = w.shape[0]
    starts = [0] if total <= win else list(range(0, total, win))
    vecs = []
    for s in starts:
        ch = w[s:s + win]
        if ch.shape[0] < TARGET_SR:
            continue
        inp = fe(ch.numpy(), sampling_rate=TARGET_SR, return_tensors='pt').input_values.to(DEVICE)
        h = m(inp).last_hidden_state.squeeze(0)
        vecs.append(torch.cat([h.mean(0), h.std(0)]).cpu().numpy())
    if not vecs:
        inp = fe(w.numpy(), sampling_rate=TARGET_SR, return_tensors='pt').input_values.to(DEVICE)
        h = m(inp).last_hidden_state.squeeze(0)
        vecs.append(torch.cat([h.mean(0), h.std(0)]).cpu().numpy())
    return np.mean(vecs, axis=0).astype(np.float32)


def embed_dir(dirpath, tag):
    os.makedirs(EXT_CACHE, exist_ok=True)
    files = sorted(glob.glob(os.path.join(dirpath, '*.wav')))
    X = []
    for f in files:
        key = os.path.join(EXT_CACHE, tag + '__' + os.path.basename(f) + '.npy')
        if os.path.exists(key):
            v = np.load(key)
        else:
            w, sr = sf.read(f)
            c = (w[:, 0] if w.ndim > 1 else w).astype(np.float32)
            v = embed_wave(c, sr)
            np.save(key, v)
        X.append(v)
    return np.stack(X) if X else np.zeros((0, 1536), np.float32)


def eer(y, s):
    fpr, tpr, _ = roc_curve(y, s)
    fnr = 1 - tpr
    i = np.nanargmin(np.abs(fnr - fpr))
    return (fpr[i] + fnr[i]) / 2


def report(name, y, s):
    y = np.asarray(y)
    s = np.asarray(s)
    if len(np.unique(y)) < 2:
        acc = ((s >= 0.5).astype(int) == y).mean()
        print(f'  {name:28s} n={len(y):3d}  ACC@0.5={acc:.3f}  media={s.mean():.3f}')
        return
    print(f'  {name:28s} n={len(y):3d}  AUC={roc_auc_score(y, s):.3f}  EER={eer(y, s):.3f}  ACC@0.5={((s>=.5).astype(int)==y).mean():.3f}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=os.path.join(ROOT, 'src', 'models', 'saved', 'wavlm_head.joblib'))
    ap.add_argument('--production', action='store_true')
    args = ap.parse_args()

    d = np.load(os.path.join(ROOT, 'models', 'ssl_cache', 'wavlm_base_plus_dataset.npz'), allow_pickle=True)
    Xa, ya, spa = d['X'], d['y'].astype(int), d['split']
    tr = spa == 'train'
    va = spa == 'val'
    print(f'Altur: train={tr.sum()} val={va.sum()}')

    print('Extrayendo features externas...')
    Xh_tr = embed_dir(os.path.join(ROOT, 'datasets_externos', 'Human_ES_train_8kHz'), 'hes_tr')
    Xh_te = embed_dir(os.path.join(ROOT, 'datasets_externos', 'Human_ES_test_8kHz'), 'hes_te')
    Xh_emilio = embed_dir(os.path.join(ROOT, 'datasets_externos', 'Human_MX_8kHz'), 'hemilio')

    smulti_dir = os.path.join(ROOT, 'datasets_externos', 'Synthetic_Multi_8kHz')
    embed_dir(smulti_dir, 'smulti')
    gtts_files = sorted(glob.glob(os.path.join(smulti_dir, 'gtts_*.wav')))
    edge_multi_files = sorted(glob.glob(os.path.join(smulti_dir, 'edge_*.wav')))

    Xs_gtts = np.stack([np.load(os.path.join(EXT_CACHE, 'smulti__' + os.path.basename(f) + '.npy')) for f in gtts_files])
    Xs_edge_multi = np.stack([np.load(os.path.join(EXT_CACHE, 'smulti__' + os.path.basename(f) + '.npy')) for f in edge_multi_files])
    Xs_edge_mx = embed_dir(os.path.join(ROOT, 'datasets_externos', 'Synthetic_MX_8kHz'), 'smx')
    Xs_piper = embed_dir(os.path.join(ROOT, 'datasets_externos', 'Synthetic_Piper_8kHz'), 'piper')

    train_parts = [Xa[tr], Xh_tr, Xs_edge_mx, Xs_edge_multi, Xs_gtts]
    train_labels = [ya[tr], np.zeros(len(Xh_tr)), np.ones(len(Xs_edge_mx)), np.ones(len(Xs_edge_multi)), np.ones(len(Xs_gtts))]
    if args.production:
        train_parts.append(Xs_piper)
        train_labels.append(np.ones(len(Xs_piper)))
    X_train = np.concatenate(train_parts)
    y_train = np.concatenate(train_labels).astype(int)
    print(f'TRAIN total={len(y_train)} (humanos={ (y_train==0).sum() } sinteticos={ (y_train==1).sum() })')
    print('  motores en train: Edge + gTTS' + (' + Piper (PRODUCCION)' if args.production else ' | held-out test: Piper (neural, no visto)'))

    sc = StandardScaler().fit(X_train)
    base = LogisticRegression(max_iter=3000, C=0.1, class_weight='balanced')
    clf = CalibratedClassifierCV(base, method='isotonic', cv=5)
    clf.fit(sc.transform(X_train), y_train)

    def P(X):
        return clf.predict_proba(sc.transform(X))[:, 1]

    print('\n=== EVAL (held-out) ===')
    report('Altur val (no-regresion)', ya[va], P(Xa[va]))
    yh = np.concatenate([np.zeros(len(Xh_te)), np.zeros(len(Xh_emilio)), np.ones(len(Xs_piper))])
    sh = np.concatenate([P(Xh_te), P(Xh_emilio), P(Xs_piper)])
    report('CROSS humanos+Piper', yh, sh)
    report('  FLEURS-test humanos', np.zeros(len(Xh_te)), P(Xh_te))
    report('  emilio humanos', np.zeros(len(Xh_emilio)), P(Xh_emilio))
    report('  Piper (motor neural no visto)', np.ones(len(Xs_piper)), P(Xs_piper))

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    joblib.dump({'scaler': sc, 'clf': clf, 'model_id': MODEL_ID, 'dim': X_train.shape[1]}, args.out)
    print('\nGuardado head:', args.out)


if __name__ == '__main__':
    main()
