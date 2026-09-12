import os
import sys
import numpy as np
import pandas as pd
import joblib

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, 'src')
sys.path.insert(0, SRC)
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score

CACHE = os.path.join(HERE, 'feats_cache.csv')
OUT = os.path.join(SRC, 'models', 'saved', 'altur_ab_holdout.joblib')
TEST = os.path.join(HERE, 'holdout_test.csv')


def mk():
    return HistGradientBoostingClassifier(max_iter=400, learning_rate=0.05, max_depth=3, l2_regularization=1.0, random_state=0)


def main():
    df = pd.read_csv(CACHE).fillna(0.0)
    feats = [c for c in df.columns if c not in ('anon_id', 'y', 'split')]
    X = df[feats].values.astype(np.float64)
    y = df['y'].values
    idx = np.arange(len(df))
    tr, ho = train_test_split(idx, test_size=0.20, stratify=y, random_state=42)

    clf = mk().fit(X[tr], y[tr])
    joblib.dump({'model': clf, 'features': feats, 'holdout_ids': df.iloc[ho]['anon_id'].tolist()}, OUT)

    ph = clf.predict_proba(X[ho])[:, 1]
    pred = (ph >= 0.5).astype(int)
    acc = accuracy_score(y[ho], pred)
    df.iloc[ho][['anon_id', 'y']].assign(p_synthetic=ph, pred=pred).to_csv(TEST, index=False)

    print(f'train={len(tr)}  holdout={len(ho)}')
    print(f'holdout accuracy = {int((pred==y[ho]).sum())}/{len(ho)} = {acc*100:.1f}%  AUC={roc_auc_score(y[ho],ph):.4f}')
    print('guardado', OUT)
    print('set de prueba holdout ->', TEST)


if __name__ == '__main__':
    main()
