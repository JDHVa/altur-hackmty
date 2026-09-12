import json
import os

import numpy as np
import joblib
import lightgbm as lgb
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score, brier_score_loss

from src.features.vad_dataset import load as load_vad

ENSEMBLE_PATH = 'src/models/saved/ensemble.pkl'
AUDIO_DIR = 'hackmty26/audio'

LGBM_PARAMS = dict(
    n_estimators=400, learning_rate=0.03, max_depth=4, num_leaves=15,
    min_child_samples=20, subsample=0.8, colsample_bytree=0.8,
    reg_lambda=1.0, random_state=42, verbose=-1,
)


def eer_threshold(y_true, scores):
    fpr, tpr, thr = roc_curve(y_true, scores)
    fnr = 1 - tpr
    idx = int(np.nanargmin(np.abs(fpr - fnr)))
    eer = (fpr[idx] + fnr[idx]) / 2.0
    return float(thr[idx]), float(eer)


def _audio_available():
    try:
        import soundfile as sf
        from src.features.audio import audio_score
    except Exception:
        return None
    import glob
    probs = []
    for path in sorted(glob.glob(os.path.join(AUDIO_DIR, '*.wav')))[:5]:
        data, sr = sf.read(path, always_2d=True)
        probs.append(audio_score(data[:, 0], sr))
    if probs and any(abs(p - 0.5) > 1e-6 for p in probs):
        return audio_score
    return None


def _audio_scores_for(df, scorer):
    import soundfile as sf
    scores = []
    for anon_id in df['anon_id']:
        data, sr = sf.read(os.path.join(AUDIO_DIR, anon_id + '.wav'), always_2d=True)
        scores.append(scorer(data[:, 0], sr))
    return np.array(scores)


def main():
    train_df, feature_cols = load_vad('train')
    val_df, _ = load_vad('val')
    y_train = train_df['label_num'].values
    y_val = val_df['label_num'].values

    model = lgb.LGBMClassifier(**LGBM_PARAMS)
    model.fit(train_df[feature_cols], y_train)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof_tab = cross_val_predict(
        lgb.LGBMClassifier(**LGBM_PARAMS), train_df[feature_cols], y_train,
        cv=cv, method='predict_proba')[:, 1]
    p_tab_val = model.predict_proba(val_df[feature_cols])[:, 1]

    scorer = _audio_available()
    signals = ['tabular']

    if scorer is not None:
        signals.append('audio')
        p_aud_train = _audio_scores_for(train_df, scorer)
        p_aud_val = _audio_scores_for(val_df, scorer)
        X_oof = np.column_stack([oof_tab, p_aud_train])
        X_val = np.column_stack([p_tab_val, p_aud_val])
        meta = LogisticRegression()
        meta.fit(X_oof, y_train)
        raw_oof = meta.predict_proba(X_oof)[:, 1]
        raw_val = meta.predict_proba(X_val)[:, 1]
    else:
        meta = None
        raw_oof = oof_tab
        raw_val = p_tab_val

    calibrator = IsotonicRegression(out_of_bounds='clip')
    calibrator.fit(raw_oof, y_train)
    cal_val = calibrator.transform(raw_val)
    threshold, eer = eer_threshold(y_train, calibrator.transform(raw_oof))

    val_preds = (cal_val > threshold).astype(int)
    auc = roc_auc_score(y_val, cal_val)
    acc = accuracy_score(y_val, val_preds)
    brier = brier_score_loss(y_val, cal_val)

    print('=== Ensemble / Calibracion (dominio VAD, consistente con la API) ===')
    print('Senales usadas:', signals)
    print(f'Umbral por EER (OOF train): {threshold:.4f} | EER train: {eer:.4f}')
    print(f'VAL -> ROC AUC: {auc:.4f} | Accuracy@umbral: {acc:.4f} | Brier: {brier:.4f}')

    os.makedirs('src/models/saved', exist_ok=True)
    joblib.dump({
        'signals': signals,
        'model': model,
        'feature_cols': feature_cols,
        'meta': meta,
        'calibrator': calibrator,
        'threshold': threshold,
    }, ENSEMBLE_PATH)
    print('Ensemble guardado en', ENSEMBLE_PATH)


if __name__ == '__main__':
    main()
