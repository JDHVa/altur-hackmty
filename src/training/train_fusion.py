# -*- coding: utf-8 -*-
import os
import torch
import joblib
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from src.data.dataset import AlturAudioDataset
from src.models.audio_model import VoiceSpoofResNet
from src.training.train_tabular import extract_features

def main():
    print('Cargando modelos base...')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    tabular_model = joblib.load('src/models/saved/lgbm_tabular.pkl')
    
    audio_model = VoiceSpoofResNet(pretrained=False).to(device)
    audio_model.load_state_dict(torch.load('src/models/saved/best_audio_resnet.pth', weights_only=True))
    audio_model.eval()
    
    manifest = pd.read_csv('hackmty26/manifest.csv')
    val_manifest = manifest[manifest['split'] == 'val']
    
    print('Extrayendo scores tabulares...')
    val_df = extract_features('hackmty26/turns', val_manifest)
    feature_cols = [c for c in val_df.columns if c not in ['anon_id', 'label']]
    tabular_probs = tabular_model.predict_proba(val_df[feature_cols])[:, 1]
    
    print('Extrayendo scores acusticos...')
    val_dataset = AlturAudioDataset('hackmty26/manifest.csv', 'hackmty26/audio', 'hackmty26/turns', split='val', max_duration_s=8)
    
    audio_probs = []
    labels = []
    
    with torch.no_grad():
        for i in range(len(val_dataset)):
            sample = val_dataset[i]
            if sample['label'] == -1: continue
            
            audio_tensor = sample['audio'].unsqueeze(0).to(device)
            out = audio_model(audio_tensor).squeeze(1)
            prob = torch.sigmoid(out).item()
            
            audio_probs.append(prob)
            labels.append(sample['label'].item())
            
    tabular_probs = tabular_probs[:len(audio_probs)]
    
    X_fusion = np.column_stack((tabular_probs, audio_probs))
    y = np.array(labels)
    
    print('\nEntrenando Meta-Modelo (Logistic Regression) para fusion...')
    fusion_model = LogisticRegression()
    fusion_model.fit(X_fusion, y)
    
    fusion_preds = fusion_model.predict(X_fusion)
    fusion_probs = fusion_model.predict_proba(X_fusion)[:, 1]
    
    auc = roc_auc_score(y, fusion_probs)
    acc = accuracy_score(y, fusion_preds)
    
    print('\n=== Resultados del Ensamble (Fusion Tardia) ===')
    print(f'ROC AUC: {auc:.4f}')
    print(f'Accuracy: {acc:.4f}')
    print('\nReporte de Clasificacion:')
    print(classification_report(y, fusion_preds, target_names=['Human (0)', 'Synthetic (1)']))
    
    print(f'\nPesos aprendidos: Tabular={fusion_model.coef_[0][0]:.3f}, Acustico={fusion_model.coef_[0][1]:.3f}')
    
    joblib.dump(fusion_model, 'src/models/saved/fusion_model.pkl')
    print('\nModelo de fusion guardado en src/models/saved/fusion_model.pkl')

if __name__ == '__main__':
    main()
