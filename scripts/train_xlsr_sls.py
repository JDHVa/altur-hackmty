# -*- coding: utf-8 -*-
import os
import sys
import glob
import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torchaudio
from transformers import AutoFeatureExtractor, Wav2Vec2Model
from sklearn.metrics import roc_auc_score, roc_curve

ROOT = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, ROOT)
from src.models.xlsr_sls import XLSRSLS

MODEL_ID = 'facebook/wav2vec2-xls-r-300m'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SR = 16000
WIN = SR * 20
MAXW = 3
CACHE = os.path.join(ROOT, 'models', 'ssl_cache', 'xlsr_layers')
os.makedirs(CACHE, exist_ok=True)

_FE = None
_M = None


def _model():
    global _FE, _M
    if _M is None:
        _FE = AutoFeatureExtractor.from_pretrained(MODEL_ID)
        _M = Wav2Vec2Model.from_pretrained(MODEL_ID, use_safetensors=True, output_hidden_states=True).to(DEVICE).eval()
    return _FE, _M


@torch.no_grad()
def layer_feats(caller, sr):
    fe, m = _model()
    w = torch.from_numpy(np.ascontiguousarray(caller)).float()
    if sr != SR:
        w = torchaudio.transforms.Resample(sr, SR)(w.unsqueeze(0)).squeeze(0)
    total = w.shape[0]
    starts = [0] if total <= WIN else np.linspace(0, total - WIN, MAXW).astype(int).tolist()
    accs = []
    for s in starts:
        ch = w[s:s + WIN]
        if ch.shape[0] < SR:
            continue
        inp = fe(ch.numpy(), sampling_rate=SR, return_tensors='pt').input_values.to(DEVICE)
        hs = m(inp).hidden_states
        pooled = torch.stack([h.squeeze(0).mean(0) for h in hs])
        accs.append(pooled.cpu().numpy())
    if not accs:
        inp = fe(w.numpy(), sampling_rate=SR, return_tensors='pt').input_values.to(DEVICE)
        hs = m(inp).hidden_states
        accs.append(torch.stack([h.squeeze(0).mean(0) for h in hs]).cpu().numpy())
    return np.mean(accs, axis=0).astype(np.float32)


def feat_cached(path, tag):
    key = os.path.join(CACHE, f'{tag}__{os.path.basename(path)}.npy')
    if os.path.exists(key):
        return np.load(key)
    w, sr = sf.read(path)
    c = (w[:, 0] if w.ndim > 1 else w).astype(np.float32)
    v = layer_feats(c, sr)
    np.save(key, v)
    return v


def collect(files, tag):
    return np.stack([feat_cached(f, tag) for f in files]) if files else np.zeros((0, 25, 1024), np.float32)


def eer(y, s):
    fpr, tpr, th = roc_curve(y, s)
    j = np.nanargmin(np.abs((1 - tpr) - fpr))
    return (fpr[j] + (1 - tpr)[j]) / 2, th[j]


def main():
    mani = pd.read_csv(os.path.join(ROOT, 'hackmty26', 'manifest.csv'))
    aud = os.path.join(ROOT, 'hackmty26', 'audio')
    tr_files, tr_y, va_files, va_y = [], [], [], []
    for _, r in mani.iterrows():
        p = os.path.join(aud, r['anon_id'] + '.wav')
        lab = 1 if r['label'] == 'synthetic' else 0
        (tr_files if r['split'] == 'train' else va_files).append(p)
        (tr_y if r['split'] == 'train' else va_y).append(lab)

    print('Extrayendo capas XLS-R (Altur)...', flush=True)
    Xtr = [collect(tr_files, 'altur')]
    ytr = [np.array(tr_y)]
    print('Altur listo. Externos...', flush=True)

    def dirfiles(sub, pat='*.wav'):
        return sorted(glob.glob(os.path.join(ROOT, 'datasets_externos', sub, pat)))

    hes_tr = dirfiles('Human_ES_train_8kHz')
    Xtr.append(collect(hes_tr, 'hes_tr')); ytr.append(np.zeros(len(hes_tr)))
    smx = dirfiles('Synthetic_MX_8kHz')
    Xtr.append(collect(smx, 'smx')); ytr.append(np.ones(len(smx)))
    edge = dirfiles('Synthetic_Multi_8kHz', 'edge_*.wav')
    Xtr.append(collect(edge, 'smulti')); ytr.append(np.ones(len(edge)))
    gtts = dirfiles('Synthetic_Multi_8kHz', 'gtts_*.wav')
    Xtr.append(collect(gtts, 'smulti')); ytr.append(np.ones(len(gtts)))

    X = np.concatenate(Xtr); y = np.concatenate(ytr).astype(np.float32)
    print(f'TRAIN={len(y)} (humanos={(y==0).sum()} sinteticos={(y==1).sum()})', flush=True)

    Xva = collect(va_files, 'altur'); yva = np.array(va_y)
    hte = dirfiles('Human_ES_test_8kHz'); Xhte = collect(hte, 'hes_te')
    em = dirfiles('Human_MX_8kHz', 'human_emilio_*.wav'); Xem = collect(em, 'hemilio')
    pi = dirfiles('Synthetic_Piper_8kHz'); Xpi = collect(pi, 'piper')

    Xt = torch.tensor(X, device=DEVICE); yt = torch.tensor(y, device=DEVICE)
    net = XLSRSLS().to(DEVICE)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4)
    lossf = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([(y == 0).sum() / max(1, (y == 1).sum())], device=DEVICE))
    net.train()
    n = len(y)
    for ep in range(60):
        perm = torch.randperm(n, device=DEVICE)
        tot = 0.0
        for i in range(0, n, 64):
            idx = perm[i:i + 64]
            opt.zero_grad()
            out = net(Xt[idx])
            loss = lossf(out, yt[idx])
            loss.backward(); opt.step()
            tot += loss.item()
        if (ep + 1) % 20 == 0:
            print(f'  epoch {ep+1} loss={tot:.3f}', flush=True)

    net.eval()

    def P(Xa):
        if len(Xa) == 0:
            return np.array([])
        with torch.no_grad():
            return net.prob(torch.tensor(Xa, device=DEVICE)).cpu().numpy()

    sva = P(Xva)
    hum = np.r_[P(Xhte), P(Xem)]
    piper = P(Xpi)
    yx = np.r_[np.zeros(len(hum)), np.ones(len(piper))]; sx = np.r_[hum, piper]
    _, t = eer(yx, sx)
    print('\n=== XLS-R-SLS held-out ===')
    print(f'  Altur val AUC={roc_auc_score(yva, sva):.3f} EER={eer(yva,sva)[0]:.3f}')
    print(f'  humanos ok@0.5={(hum<.5).mean()*100:.0f}%  media={hum.mean():.3f} (FLEURS {P(Xhte).mean():.3f}/emilio {P(Xem).mean():.3f})')
    print(f'  Piper AUC={roc_auc_score(yx,sx):.3f} @0.5={(piper>=.5).mean()*100:.0f}% @EER({t:.2f})={(piper>=t).mean()*100:.0f}% media={piper.mean():.3f}')

    out = os.path.join(ROOT, 'src', 'models', 'saved', 'xlsr_sls.pt')
    torch.save({'state_dict': net.state_dict(), 'model_id': MODEL_ID}, out)
    print('\nGuardado', out)


if __name__ == '__main__':
    main()
