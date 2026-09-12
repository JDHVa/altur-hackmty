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
sys.path.insert(0, os.path.join(ROOT, 'src'))
from features.prosody import prosody_vector

EXT_CACHE = os.path.join(ROOT, 'models', 'ssl_cache', 'ext')
ALT_CACHE = os.path.join(ROOT, 'models', 'ssl_cache', 'wavlm_base_plus')
PROS_CACHE = os.path.join(ROOT, 'models', 'ssl_cache', 'prosody')
os.makedirs(PROS_CACHE, exist_ok=True)


def pros_for(path, key):
    c = os.path.join(PROS_CACHE, key + '.npy')
    if os.path.exists(c):
        return np.load(c)
    w, sr = sf.read(path)
    x = (w[:, 0] if w.ndim > 1 else w).astype(np.float32)
    v = prosody_vector(x, sr)
    np.save(c, v)
    return v


def load_dir(dirpath, wavlm_tag, pattern='*.wav'):
    files = sorted(glob.glob(os.path.join(dirpath, pattern)))
    W, P = [], []
    for f in files:
        b = os.path.basename(f)
        wv = np.load(os.path.join(EXT_CACHE, wavlm_tag + '__' + b + '.npy'))
        W.append(wv)
        P.append(pros_for(f, wavlm_tag + '__' + b))
    return np.array(W), np.array(P)


def load_altur():
    d = np.load(os.path.join(ROOT, 'models', 'ssl_cache', 'wavlm_base_plus_dataset.npz'), allow_pickle=True)
    ids, y, sp = d['ids'], d['y'].astype(int), d['split']
    W = np.stack([np.load(os.path.join(ALT_CACHE, str(i) + '.npy')) for i in ids])
    P = np.stack([pros_for(os.path.join(ROOT, 'hackmty26', 'audio', str(i) + '.wav'), 'altur__' + str(i)) for i in ids])
    return W, P, y, sp


def main():
    Wa, Pa, ya, spa = load_altur()
    tr = spa == 'train'; va = spa == 'val'
    Wh_tr, Ph_tr = load_dir(os.path.join(ROOT, 'datasets_externos', 'Human_ES_train_8kHz'), 'hes_tr')
    Wh_te, Ph_te = load_dir(os.path.join(ROOT, 'datasets_externos', 'Human_ES_test_8kHz'), 'hes_te')
    We, Pe = load_dir(os.path.join(ROOT, 'datasets_externos', 'Human_MX_8kHz'), 'hemilio', 'human_emilio_*.wav')
    Ws_mx, Ps_mx = load_dir(os.path.join(ROOT, 'datasets_externos', 'Synthetic_MX_8kHz'), 'smx')
    Ws_ed, Ps_ed = load_dir(os.path.join(ROOT, 'datasets_externos', 'Synthetic_Multi_8kHz'), 'smulti', 'edge_*.wav')
    Ws_gt, Ps_gt = load_dir(os.path.join(ROOT, 'datasets_externos', 'Synthetic_Multi_8kHz'), 'smulti', 'gtts_*.wav')
    Wp, Pp = load_dir(os.path.join(ROOT, 'datasets_externos', 'Synthetic_Piper_8kHz'), 'piper')

    def cat(Ws, Ps, use_p):
        return np.concatenate([Ws, Ps], axis=1) if use_p else Ws

    for use_p in (False, True):
        tagp = 'WavLM+prosodia' if use_p else 'WavLM solo'
        Xtr = np.concatenate([
            cat(Wa[tr], Pa[tr], use_p), cat(Wh_tr, Ph_tr, use_p),
            cat(Ws_mx, Ps_mx, use_p), cat(Ws_ed, Ps_ed, use_p), cat(Ws_gt, Ps_gt, use_p),
        ])
        ytr = np.concatenate([ya[tr], np.zeros(len(Wh_tr)), np.ones(len(Ws_mx)), np.ones(len(Ws_ed)), np.ones(len(Ws_gt))]).astype(int)
        sc = StandardScaler().fit(Xtr)
        clf = CalibratedClassifierCV(LogisticRegression(max_iter=3000, C=0.1, class_weight='balanced'), method='isotonic', cv=5)
        clf.fit(sc.transform(Xtr), ytr)

        def P(Ws, Ps):
            return clf.predict_proba(sc.transform(cat(Ws, Ps, use_p)))[:, 1]

        hum = np.r_[P(Wh_te, Ph_te), P(We, Pe)]
        piper = P(Wp, Pp)
        y = np.r_[np.zeros(len(hum)), np.ones(len(piper))]
        s = np.r_[hum, piper]
        fpr, tpr, th = roc_curve(y, s); fnr = 1 - tpr; j = np.nanargmin(np.abs(fnr - fpr)); t = th[j]
        print(f'\n### {tagp} (dim={Xtr.shape[1]}) ###')
        print(f'  Altur val AUC={roc_auc_score(ya[va], P(Wa[va], Pa[va])):.3f}')
        print(f'  humanos ok@0.5={(hum<.5).mean()*100:.0f}%  media={hum.mean():.3f}')
        print(f'  Piper (no visto) AUC={roc_auc_score(np.r_[np.zeros(len(hum)),np.ones(len(piper))], s):.3f}  cazado@0.5={(piper>=.5).mean()*100:.0f}%  cazado@EER({t:.2f})={(piper>=t).mean()*100:.0f}%  media={piper.mean():.3f}')


if __name__ == '__main__':
    main()
