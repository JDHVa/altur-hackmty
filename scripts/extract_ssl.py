# -*- coding: utf-8 -*-
import os
import sys
import csv
import argparse
import numpy as np
import soundfile as sf
import torch
import torchaudio
from transformers import AutoFeatureExtractor, WavLMModel, Wav2Vec2Model

ROOT = os.path.join(os.path.dirname(__file__), '..')
AUDIO = os.path.join(ROOT, 'hackmty26', 'audio')
MANIFEST = os.path.join(ROOT, 'hackmty26', 'manifest.csv')
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

WINDOW_S = 20
TARGET_SR = 16000


def load_manifest():
    rows = []
    with open(MANIFEST, encoding='utf-8') as f:
        for r in csv.DictReader(f):
            rows.append((r['anon_id'], 1 if r['label'] == 'synthetic' else 0, r['split']))
    return rows


def build_model(model_id, kind):
    fe = AutoFeatureExtractor.from_pretrained(model_id)
    cls = WavLMModel if kind == 'wavlm' else Wav2Vec2Model
    model = cls.from_pretrained(model_id, use_safetensors=True).to(DEVICE).eval()
    return fe, model


@torch.no_grad()
def embed(caller, sr, fe, model):
    wave = torch.from_numpy(np.ascontiguousarray(caller)).float()
    if sr != TARGET_SR:
        wave = torchaudio.transforms.Resample(sr, TARGET_SR)(wave.unsqueeze(0)).squeeze(0)
    win = TARGET_SR * WINDOW_S
    total = wave.shape[0]
    starts = [0] if total <= win else list(range(0, total, win))
    vecs = []
    for s in starts:
        chunk = wave[s:s + win]
        if chunk.shape[0] < TARGET_SR:
            continue
        inp = fe(chunk.numpy(), sampling_rate=TARGET_SR, return_tensors='pt').input_values.to(DEVICE)
        h = model(inp).last_hidden_state.squeeze(0)
        vecs.append(torch.cat([h.mean(0), h.std(0)]).cpu().numpy())
    if not vecs:
        inp = fe(wave.numpy(), sampling_rate=TARGET_SR, return_tensors='pt').input_values.to(DEVICE)
        h = model(inp).last_hidden_state.squeeze(0)
        vecs.append(torch.cat([h.mean(0), h.std(0)]).cpu().numpy())
    return np.mean(vecs, axis=0).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--kind', choices=['wavlm', 'w2v2'], required=True)
    ap.add_argument('--tag', required=True)
    args = ap.parse_args()

    cache_dir = os.path.join(ROOT, 'models', 'ssl_cache', args.tag)
    os.makedirs(cache_dir, exist_ok=True)
    fe, model = build_model(args.model, args.kind)

    rows = load_manifest()
    X, ids, ys, splits = [], [], [], []
    for i, (aid, y, split) in enumerate(rows):
        npy = os.path.join(cache_dir, aid + '.npy')
        if os.path.exists(npy):
            v = np.load(npy)
        else:
            w, sr = sf.read(os.path.join(AUDIO, aid + '.wav'))
            caller = (w[:, 0] if w.ndim > 1 else w).astype(np.float32)
            v = embed(caller, sr, fe, model)
            np.save(npy, v)
        X.append(v); ids.append(aid); ys.append(y); splits.append(split)
        if (i + 1) % 20 == 0:
            print(f'[{i+1}/{len(rows)}]', flush=True)

    X = np.stack(X)
    out = os.path.join(ROOT, 'models', 'ssl_cache', f'{args.tag}_dataset.npz')
    np.savez(out, X=X, ids=np.array(ids), y=np.array(ys), split=np.array(splits))
    print('dim=', X.shape, '-> guardado', out)


if __name__ == '__main__':
    main()
