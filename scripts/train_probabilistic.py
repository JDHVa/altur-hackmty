# -*- coding: utf-8 -*-
import os
import sys
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.mixture import GaussianMixture
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, roc_curve, brier_score_loss, accuracy_score

ROOT = os.path.join(os.path.dirname(__file__), '..')
SAVE = os.path.join(ROOT, 'src', 'models', 'saved')
os.makedirs(SAVE, exist_ok=True)


def eer(y, s):
    fpr, tpr, _ = roc_curve(y, s)
    fnr = 1 - tpr
    i = np.nanargmin(np.abs(fnr - fpr))
    return (fpr[i] + fnr[i]) / 2


class GMMLikelihoodRatio:
    def __init__(self, n_components=2, n_pca=10, reg=1e-3, seed=0):
        self.n_components = n_components
        self.n_pca = n_pca
        self.reg = reg
        self.seed = seed

    def fit(self, X, y):
        self.scaler_ = StandardScaler().fit(X)
        Xs = self.scaler_.transform(X)
        self.pca_ = PCA(n_components=min(self.n_pca, Xs.shape[1])).fit(Xs)
        Xp = self.pca_.transform(Xs)
        self.g0_ = GaussianMixture(self.n_components, covariance_type='diag', reg_covar=self.reg, random_state=self.seed).fit(Xp[y == 0])
        self.g1_ = GaussianMixture(self.n_components, covariance_type='diag', reg_covar=self.reg, random_state=self.seed).fit(Xp[y == 1])
        llr = self._llr(X)
        self.cal_ = LogisticRegression(max_iter=1000).fit(llr.reshape(-1, 1), y)
        return self

    def _llr(self, X):
        Xp = self.pca_.transform(self.scaler_.transform(X))
        return self.g1_.score_samples(Xp) - self.g0_.score_samples(Xp)

    def predict_proba(self, X):
        return self.cal_.predict_proba(self._llr(X).reshape(-1, 1))


def make_models():
    return {
        'gaussian_nb': Pipeline([('sc', StandardScaler()), ('m', GaussianNB())]),
        'lda': Pipeline([('sc', StandardScaler()), ('m', LinearDiscriminantAnalysis())]),
        'qda': Pipeline([('sc', StandardScaler()), ('m', QuadraticDiscriminantAnalysis(reg_param=0.5))]),
        'bayes_logreg': Pipeline([('sc', StandardScaler()),
                                  ('m', CalibratedClassifierCV(LogisticRegression(max_iter=3000, C=0.3, class_weight='balanced'), method='isotonic', cv=5))]),
        'gmm_llr': GMMLikelihoodRatio(n_components=2, n_pca=10),
    }


def main():
    df = pd.read_csv(os.path.join(ROOT, 'models', 'altur_features.csv'))
    feat_cols = [c for c in df.columns if c not in ('label', 'split', 'anon_id')]
    tr = df['split'].values == 'train'
    va = df['split'].values == 'val'
    Xtr, ytr = df.loc[tr, feat_cols].values, df.loc[tr, 'label'].values
    Xva, yva = df.loc[va, feat_cols].values, df.loc[va, 'label'].values

    print(f"{'modelo':16s} {'AUC':>6s} {'EER':>6s} {'Brier':>7s} {'ACC':>6s}")
    results = {}
    for name, model in make_models().items():
        model.fit(Xtr, ytr)
        p = model.predict_proba(Xva)[:, 1]
        auc = roc_auc_score(yva, p)
        e = eer(yva, p)
        br = brier_score_loss(yva, p)
        acc = accuracy_score(yva, (p >= 0.5).astype(int))
        results[name] = auc
        print(f'{name:16s} {auc:6.3f} {e:6.3f} {br:7.3f} {acc:6.3f}')
        joblib.dump({'model': model, 'feature_cols': feat_cols, 'kind': name}, os.path.join(SAVE, f'prob_{name}.joblib'))

    best = max(results, key=results.get)
    print(f'\nMejor por AUC: {best} ({results[best]:.3f})')
    print('Modelos guardados en src/models/saved/prob_*.joblib')


if __name__ == '__main__':
    main()
