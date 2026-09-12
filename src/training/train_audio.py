# -*- coding: utf-8 -*-
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, accuracy_score
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from src.data.dataset import AlturAudioDataset
from src.models.audio_model import VoiceSpoofResNet

def train_audio_model():
    print('Verificando hardware...')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Dispositivo seleccionado: {device}')
    
    manifest_path = 'hackmty26/manifest.csv'
    audio_dir = 'hackmty26/audio'
    turns_dir = 'hackmty26/turns'
    
    batch_size = 16
    epochs = 5
    lr = 0.001
    
    train_dataset = AlturAudioDataset(manifest_path, audio_dir, turns_dir, split='train', max_duration_s=8)
    val_dataset = AlturAudioDataset(manifest_path, audio_dir, turns_dir, split='val', max_duration_s=8)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    print(f'Train samples: {len(train_dataset)} | Val samples: {len(val_dataset)}')
    
    # Transfer learning (pretrained=True) + weight_decay para reducir sobreajuste.
    # NOTA: el modelo SOTA del plan es Wav2Vec2-AASIST frozen (tarea de Emilio en GPU,
    # ver PLAN.md/EQUIPO.md). Este ResNet es el baseline de audio mientras tanto.
    model = VoiceSpoofResNet(pretrained=True).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)
    
    best_auc = 0.0
    os.makedirs('src/models/saved', exist_ok=True)
    
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        print(f'\n--- Epoch {epoch}/{epochs} ---')
        
        for batch in tqdm(train_loader, desc='Entrenando'):
            audios = batch['audio'].to(device)
            labels = batch['label'].float().unsqueeze(1).to(device)
            
            valid_idx = (labels != -1).squeeze()
            if not valid_idx.any(): continue
            audios, labels = audios[valid_idx], labels[valid_idx]
            
            optimizer.zero_grad()
            outputs = model(audios)
            
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
        avg_train_loss = train_loss / len(train_loader)
        
        model.eval()
        val_preds, val_labels = [], []
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc='Validando '):
                audios = batch['audio'].to(device)
                labels = batch['label'].float().to(device)
                
                valid_idx = (labels != -1)
                if not valid_idx.any(): continue
                audios, labels = audios[valid_idx], labels[valid_idx]
                
                outputs = model(audios).squeeze(1)
                probs = torch.sigmoid(outputs)
                
                val_preds.extend(probs.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())
                
        auc = roc_auc_score(val_labels, val_preds)
        preds_binary = [1 if p > 0.5 else 0 for p in val_preds]
        acc = accuracy_score(val_labels, preds_binary)
        
        print(f'Train Loss: {avg_train_loss:.4f} | Val ROC AUC: {auc:.4f} | Val Acc: {acc:.4f}')
        scheduler.step(auc)
        
        if auc > best_auc:
            best_auc = auc
            torch.save(model.state_dict(), 'src/models/saved/best_audio_resnet.pth')
            print('-> Nuevo mejor modelo guardado!')

if __name__ == '__main__':
    train_audio_model()
