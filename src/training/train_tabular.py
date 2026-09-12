# -*- coding: utf-8 -*-
import os
import json
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report
import joblib

def extract_features(turns_dir, manifest_df):
    features = []
    labels = []
    label_map = {'human': 0, 'synthetic': 1}
    
    for _, row in manifest_df.iterrows():
        anon_id = row['anon_id']
        label = label_map.get(row['label'], -1)
        if label == -1: continue
            
        json_path = os.path.join(turns_dir, f'{anon_id}.json')
        if not os.path.exists(json_path):
            continue
            
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            turns = data.get('turns', [])
            
        if not turns:
            continue
            
        latencies, overlaps, caller_durations, agent_durations = [], [], [], []
        
        for i in range(1, len(turns)):
            prev_turn = turns[i-1]
            curr_turn = turns[i]
            
            # Reaccion del caller al agente
            if curr_turn['channel'] == 0 and prev_turn['channel'] == 1:
                latency = curr_turn['start'] - prev_turn['end']
                if latency < 0:
                    overlaps.append(abs(latency))
                    latencies.append(0)
                else:
                    latencies.append(latency)
                    overlaps.append(0)
            
            if curr_turn['channel'] == 0:
                caller_durations.append(curr_turn['end'] - curr_turn['start'])
            else:
                agent_durations.append(curr_turn['end'] - curr_turn['start'])
                
        features.append({
            'anon_id': anon_id,
            'avg_latency': np.mean(latencies) if latencies else 0,
            'std_latency': np.std(latencies) if latencies else 0,
            'max_latency': np.max(latencies) if latencies else 0,
            'avg_overlap': np.mean(overlaps) if overlaps else 0,
            'sum_overlap': np.sum(overlaps) if overlaps else 0,
            'avg_caller_dur': np.mean(caller_durations) if caller_durations else 0,
            'avg_agent_dur': np.mean(agent_durations) if agent_durations else 0,
            'num_caller_turns': len(caller_durations),
            'num_agent_turns': len(agent_durations),
            'label': label
        })
        
    return pd.DataFrame(features)

def main():
    print('Extrayendo features conversacionales...')
    manifest = pd.read_csv('hackmty26/manifest.csv')
    train_manifest = manifest[manifest['split'] == 'train']
    val_manifest = manifest[manifest['split'] == 'val']
    
    train_df = extract_features('hackmty26/turns', train_manifest)
    val_df = extract_features('hackmty26/turns', val_manifest)
    
    feature_cols = [c for c in train_df.columns if c not in ['anon_id', 'label']]
    
    X_train, y_train = train_df[feature_cols], train_df['label']
    X_val, y_val = val_df[feature_cols], val_df['label']
    
    print(f'Entrenando LightGBM con {len(X_train)} ejemplos (train) y {len(X_val)} ejemplos (val)...')
    
    model = lgb.LGBMClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=5,
        random_state=42,
        importance_type='gain'
    )
    
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(stopping_rounds=20, verbose=False)]
    )
    
    val_preds = model.predict(X_val)
    val_probs = model.predict_proba(X_val)[:, 1]
    
    auc = roc_auc_score(y_val, val_probs)
    acc = accuracy_score(y_val, val_preds)
    
    print('\n=== Resultados del Modelo Tabular ===')
    print(f'ROC AUC: {auc:.4f}')
    print(f'Accuracy: {acc:.4f}')
    print('\nReporte de Clasificacion:')
    print(classification_report(y_val, val_preds, target_names=['Human (0)', 'Synthetic (1)']))
    
    importances = pd.DataFrame({
        'feature': feature_cols,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print('\nTop 5 Features mas importantes:')
    print(importances.head(5).to_string(index=False))
    
    os.makedirs('src/models/saved', exist_ok=True)
    joblib.dump(model, 'src/models/saved/lgbm_tabular.pkl')
    print('\nModelo guardado en src/models/saved/lgbm_tabular.pkl')

if __name__ == '__main__':
    main()
