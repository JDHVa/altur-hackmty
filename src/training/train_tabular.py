# -*- coding: utf-8 -*-
"""Senal A (conversacional): baseline tabular con LightGBM.

Usa las 45 features ya calculadas en hackmty26/dataset_turns_features.csv
(latencias, solapamientos, silencios, ratios de habla, etc.) en vez de
recalcular unas pocas a mano. Entrena en 'train' y evalua en 'val'.
"""
import os
import json
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report
import joblib

FEATURES_CSV = 'hackmty26/dataset_turns_features.csv'
LABEL_MAP = {'human': 0, 'synthetic': 1}
NON_FEATURE_COLS = {'anon_id', 'label', 'split', 'label_num'}


def load_tabular(split, features_csv=FEATURES_CSV):
    """Devuelve (df, feature_cols) para un split dado.

    df conserva anon_id (para alinear con la senal de audio en la fusion) y
    una columna label_num (0=human, 1=synthetic). feature_cols son las 42
    columnas numericas de features del reto.
    """
    df = pd.read_csv(features_csv)
    if split:
        df = df[df['split'] == split].reset_index(drop=True)
    df['label_num'] = df['label'].map(LABEL_MAP)
    df = df[df['label_num'].notna()].reset_index(drop=True)
    df['label_num'] = df['label_num'].astype(int)

    feature_cols = [
        c for c in df.columns
        if c not in NON_FEATURE_COLS and pd.api.types.is_numeric_dtype(df[c])
    ]
    return df, feature_cols


def main():
    print('Cargando features conversacionales (45 features del CSV)...')
    train_df, feature_cols = load_tabular('train')
    val_df, _ = load_tabular('val')

    X_train, y_train = train_df[feature_cols], train_df['label_num']
    X_val, y_val = val_df[feature_cols], val_df['label_num']

    print(f'Entrenando LightGBM con {len(X_train)} ejemplos (train) y '
          f'{len(X_val)} ejemplos (val) | {len(feature_cols)} features...')

    model = lgb.LGBMClassifier(
        n_estimators=400,
        learning_rate=0.03,
        max_depth=4,
        num_leaves=15,
        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        random_state=42,
        importance_type='gain',
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=False)],
    )

    val_probs = model.predict_proba(X_val)[:, 1]
    val_preds = (val_probs > 0.5).astype(int)

    auc = roc_auc_score(y_val, val_probs)
    acc = accuracy_score(y_val, val_preds)

    print('\n=== Resultados del Modelo Tabular (Senal A) ===')
    print(f'ROC AUC: {auc:.4f}')
    print(f'Accuracy: {acc:.4f}')
    print('\nReporte de Clasificacion:')
    print(classification_report(y_val, val_preds, target_names=['Human (0)', 'Synthetic (1)']))

    importances = pd.DataFrame({
        'feature': feature_cols,
        'importance': model.feature_importances_,
    }).sort_values('importance', ascending=False)
    print('\nTop 10 Features mas importantes:')
    print(importances.head(10).to_string(index=False))

    os.makedirs('src/models/saved', exist_ok=True)
    joblib.dump(model, 'src/models/saved/lgbm_tabular.pkl')
    # Guardar la lista de features para que la API use exactamente las mismas.
    with open('src/models/saved/tabular_features.json', 'w', encoding='utf-8') as f:
        json.dump(feature_cols, f, ensure_ascii=False, indent=2)
    print('\nModelo guardado en src/models/saved/lgbm_tabular.pkl')
    print('Features guardadas en src/models/saved/tabular_features.json')


if __name__ == '__main__':
    main()
