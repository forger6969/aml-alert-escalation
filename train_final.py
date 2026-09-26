#!/usr/bin/env python3
"""Финальная сборка: лучшая конфигурация из v3 (без scale_pos_weight, мягкая
регуляризация), усреднённая по 5 разным random_state — снижает дисперсию
единичного прогона и даёт более надёжную оценку и предсказание для сдачи.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import lightgbm as lgb

D = Path(__file__).parent
DATA_DIR = D / "data"

train = pd.read_parquet(D / "train_features_v3.parquet")
test = pd.read_parquet(D / "test_features_v3.parquet")
test_signals = pd.read_csv(DATA_DIR / "test_signals.csv")
feature_cols = [c for c in train.columns if c not in ("signal_id", "eskalatsiya")]
X, y = train[feature_cols], train["eskalatsiya"]
X_test = test[feature_cols]

BASE_PARAMS = dict(n_estimators=600, learning_rate=0.02, num_leaves=15, max_depth=4,
                    min_child_samples=30, subsample=0.7, colsample_bytree=0.6,
                    reg_alpha=1.0, reg_lambda=1.0, verbosity=-1)
SEEDS = [42, 1, 7, 123, 2026]

all_oof = np.zeros((len(SEEDS), len(X)))
all_test_pred = np.zeros((len(SEEDS), len(X_test)))
seed_aucs = []

for si, seed in enumerate(SEEDS):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    oof = np.zeros(len(X))
    for tr_idx, val_idx in skf.split(X, y):
        model = lgb.LGBMClassifier(random_state=seed, **BASE_PARAMS)
        model.fit(X.iloc[tr_idx], y.iloc[tr_idx],
                  eval_set=[(X.iloc[val_idx], y.iloc[val_idx])], eval_metric="auc",
                  callbacks=[lgb.early_stopping(100, verbose=False)])
        oof[val_idx] = model.predict_proba(X.iloc[val_idx])[:, 1]
    auc = roc_auc_score(y, oof)
    seed_aucs.append(auc)
    all_oof[si] = oof
    print(f"seed {seed}: AUC = {auc:.4f}")

    final_model = lgb.LGBMClassifier(random_state=seed, **{**BASE_PARAMS, "n_estimators": 300})
    final_model.fit(X, y)
    all_test_pred[si] = final_model.predict_proba(X_test)[:, 1]

blended_oof = all_oof.mean(axis=0)
blended_auc = roc_auc_score(y, blended_oof)
print(f"\nAUC по отдельным seed: {[f'{a:.4f}' for a in seed_aucs]}")
print(f"среднее по seed: {np.mean(seed_aucs):.4f} +- {np.std(seed_aucs):.4f}")
print(f"AUC усреднённого (blended) oof: {blended_auc:.4f}")

blended_test_pred = all_test_pred.mean(axis=0)
submission = pd.DataFrame({"signal_id": test["signal_id"], "ehtimollik": blended_test_pred})

assert submission["signal_id"].duplicated().sum() == 0, "дубли signal_id"
assert submission["ehtimollik"].between(0, 1).all(), "вероятность вне [0,1]"
assert submission.isna().sum().sum() == 0, "есть пропуски"
assert len(submission) == 6000, f"ожидалось 6000, получено {len(submission)}"
assert set(submission["signal_id"]) == set(test_signals["signal_id"]), "signal_id в submission не совпадают с test_signals.csv"
assert list(submission.columns) == ["signal_id", "ehtimollik"], "неверные колонки"

submission.to_csv(D / "team_E418F3DF.csv", index=False)
print(f"\nсохранено ФИНАЛЬНОЕ: {D / 'team_E418F3DF.csv'}")
print(f"строк: {len(submission)}, все проверки пройдены")
print(f"\nИТОГОВЫЙ AUC (кросс-валидация, усреднение 5 seed): {blended_auc:.4f}")

pd.DataFrame({"signal_id": train["signal_id"], "eskalatsiya": y, "oof_pred": blended_oof}) \
    .to_csv(D / "oof_final.csv", index=False)

with open(D / "auc_history.txt", "a") as f:
    f.write(f"FINAL (усреднение 5 seed, {len(feature_cols)} признаков): {blended_auc:.4f}\n")
