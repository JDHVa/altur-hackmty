# -*- coding: utf-8 -*-
"""Ensamble (fusion tardia) de la senal tabular (A) y la de audio (B).

Correccion clave: el meta-modelo se ENTRENA con los scores del split 'train'
y se EVALUA en 'val'. Antes se hacia fit y evaluacion sobre el mismo 'val',
lo que daba una metrica falsamente optimista (data leakage).

Alineacion tabular<->audio por anon_id (no por posicion/truncado).
"""
import os
import sys
import torch
import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from src.data.dataset import AlturAudioDataset
from src.models.audio_model import VoiceSpoofResNet
from src.training.train_tabular import load_tabular


def audio_scores_by_id(model, split, device):
    """Devuelve {anon_id: prob_sintetico} para un split usando el modelo de audio."""
    ds = AlturAudioDataset('hackmty26/manifest.csv', 'hackmty26/audio',
                           'hackmty26/turns', split=split, max_duration_s=8)
    scores = {}
    with torch.no_grad():
        for i in range(len(ds)):
            sample = ds[i]
            if int(sample['label']) == -1:
                continue
            audio_tensor = sample['audio'].unsqueeze(0).to(device)
            out = model(audio_tensor).squeeze(1)
            scores[sample['id']] = torch.sigmoid(out).item()
    return scores


def build_matrix(split, tabular_model, audio_map):
    """Alinea por anon_id -> (X=[prob_tab, prob_audio], y)."""
    df, feature_cols = load_tabular(split)
    tab_probs = tabular_model.predict_proba(df[feature_cols])[:, 1]

    X, y = [], []
    for tab_prob, anon_id, label in zip(tab_probs, df['anon_id'], df['label_num']):
        if anon_id not in audio_map:
            continue  # sin score de audio -> se omite
        X.append([tab_prob, audio_map[anon_id]])
        y.append(int(label))
    return np.array(X), np.array(y)


def main():
    print('Cargando modelos base...')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    tabular_model = joblib.load('src/models/saved/lgbm_tabular.pkl')

    audio_model = VoiceSpoofResNet(pretrained=False).to(device)
    audio_model.load_state_dict(
        torch.load('src/models/saved/best_audio_resnet.pth', weights_only=True))
    audio_model.eval()

    print('Extrayendo scores de audio (train y val)...')
    audio_train = audio_scores_by_id(audio_model, 'train', device)
    audio_val = audio_scores_by_id(audio_model, 'val', device)

    X_train, y_train = build_matrix('train', tabular_model, audio_train)
    X_val, y_val = build_matrix('val', tabular_model, audio_val)

    print(f'Fusion: entrenando con {len(X_train)} (train), evaluando en {len(X_val)} (val)...')
    fusion_model = LogisticRegression()
    fusion_model.fit(X_train, y_train)  # <-- se ENTRENA en train

    val_probs = fusion_model.predict_proba(X_val)[:, 1]  # <-- se EVALUA en val
    val_preds = (val_probs > 0.5).astype(int)

    auc = roc_auc_score(y_val, val_probs)
    acc = accuracy_score(y_val, val_preds)

    print('\n=== Resultados del Ensamble (Fusion Tardia) — EVAL EN VAL ===')
    print(f'ROC AUC: {auc:.4f}')
    print(f'Accuracy: {acc:.4f}')
    print('\nReporte de Clasificacion:')
    print(classification_report(y_val, val_preds, target_names=['Human (0)', 'Synthetic (1)']))
    print(f'\nPesos aprendidos: Tabular={fusion_model.coef_[0][0]:.3f}, '
          f'Acustico={fusion_model.coef_[0][1]:.3f}')

    joblib.dump(fusion_model, 'src/models/saved/fusion_model.pkl')
    print('\nModelo de fusion guardado en src/models/saved/fusion_model.pkl')


if __name__ == '__main__':
    main()
