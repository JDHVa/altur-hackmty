import os
import sys
import numpy as np
import pandas as pd
import soundfile as sf
import joblib

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, 'src')
REPO = os.path.dirname(HERE)
sys.path.insert(0, SRC)

from features.conversational import extract_features_from_turns, turns_from_audio
from features.audio import audio_score
from features.heavy_audio import xlsr_sls_score, flow_llr_score
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.ensemble import HistGradientBoostingClassifier

AUD = os.path.join(REPO, 'hackmty26', 'audio')
MANI = os.path.join(REPO, 'hackmty26', 'manifest.csv')
OUT = os.path.join(SRC, 'models', 'saved', 'altur_ab.joblib')
CACHE = os.path.join(HERE, 'feats_cache.csv')


def build():
    mani = pd.read_csv(MANI)
    rows = []
    for i, (_, r) in enumerate(mani.iterrows()):
        aid = r['anon_id']
        w, sr = sf.read(os.path.join(AUD, aid + '.wav'), always_2d=True, dtype='float32')
        caller = w[:, 0]
        agent = w[:, 1] if w.shape[1] > 1 else np.zeros_like(caller)
        dur = len(caller) / sr if sr else 0.0
        turns = turns_from_audio(caller, agent, sr)
        f = extract_features_from_turns(turns, dur)
        f['wavlm'] = float(audio_score(caller, sr))
        f['xlsr'] = float(xlsr_sls_score(caller, sr))
        f['flow'] = float(flow_llr_score(caller, sr))
        f['anon_id'] = aid
        f['y'] = 1 if r['label'] == 'synthetic' else 0
        f['split'] = r['split']
        rows.append(f)
        if (i + 1) % 50 == 0:
            print(f'  {i+1}/{len(mani)}', flush=True)
    df = pd.DataFrame(rows).fillna(0.0)
    df.to_csv(CACHE, index=False)
    return df


def mk(s):
    return HistGradientBoostingClassifier(max_iter=400, learning_rate=0.05, max_depth=3, l2_regularization=1.0, random_state=s)


def main():
    df = pd.read_csv(CACHE) if os.path.exists(CACHE) else build()
    feats = [c for c in df.columns if c not in ('anon_id', 'y', 'split')]
    X = df[feats].values.astype(np.float64)
    y = df['y'].values

    accs, aucs = [], []
    for s in range(5):
        skf = StratifiedKFold(5, shuffle=True, random_state=s)
        p = cross_val_predict(mk(s), X, y, cv=skf, method='predict_proba')[:, 1]
        accs.append(accuracy_score(y, (p >= 0.5).astype(int)))
        aucs.append(roc_auc_score(y, p))
    print(f'CV 5-fold x5: acc={np.mean(accs)*100:.2f}% (+/-{np.std(accs)*100:.2f}) AUC={np.mean(aucs):.4f}')

    tr = (df.split == 'train').values
    va = (df.split == 'val').values
    clf = mk(0).fit(X[tr], y[tr])
    pv = clf.predict_proba(X[va])[:, 1]
    print(f'val oficial: acc@0.5={accuracy_score(y[va],(pv>=.5).astype(int))*100:.1f}% AUC={roc_auc_score(y[va],pv):.4f}')

    final = mk(0).fit(X, y)
    joblib.dump({'model': final, 'features': feats}, OUT)
    print('guardado', OUT, 'con', len(feats), 'features')


if __name__ == '__main__':
    main()
