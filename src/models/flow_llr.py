# -*- coding: utf-8 -*-
import numpy as np
import torch
import zuko
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression


class FlowLLR:
    def __init__(self, n_pca=64, transforms=3, hidden=(128, 128), epochs=300, lr=1e-3, seed=0, device=None):
        self.n_pca = n_pca
        self.transforms = transforms
        self.hidden = list(hidden)
        self.epochs = epochs
        self.lr = lr
        self.seed = seed
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

    def _new_flow(self, d):
        return zuko.flows.NSF(features=d, context=0, transforms=self.transforms, hidden_features=self.hidden).to(self.device)

    def _fit_flow(self, X):
        torch.manual_seed(self.seed)
        flow = self._new_flow(X.shape[1])
        opt = torch.optim.Adam(flow.parameters(), lr=self.lr, weight_decay=1e-5)
        xt = torch.tensor(X, dtype=torch.float32, device=self.device)
        flow.train()
        for _ in range(self.epochs):
            opt.zero_grad()
            loss = -flow().log_prob(xt).mean()
            loss.backward()
            opt.step()
        flow.eval()
        return flow

    def fit(self, X, y):
        self.scaler_ = StandardScaler().fit(X)
        Xs = self.scaler_.transform(X)
        self.pca_ = PCA(n_components=min(self.n_pca, Xs.shape[1])).fit(Xs)
        Xp = self.pca_.transform(Xs).astype(np.float32)
        self.f_human_ = self._fit_flow(Xp[y == 0])
        self.f_synth_ = self._fit_flow(Xp[y == 1])
        llr = self._llr(X)
        self.cal_ = LogisticRegression(max_iter=1000).fit(llr.reshape(-1, 1), y)
        return self

    def _llr(self, X):
        Xp = self.pca_.transform(self.scaler_.transform(X)).astype(np.float32)
        xt = torch.tensor(Xp, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            ls = self.f_synth_().log_prob(xt).cpu().numpy()
            lh = self.f_human_().log_prob(xt).cpu().numpy()
        return ls - lh

    def predict_proba(self, X):
        return self.cal_.predict_proba(self._llr(X).reshape(-1, 1))
