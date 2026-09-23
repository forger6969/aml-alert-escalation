#!/usr/bin/env python3
"""Обучение v3: полный набор признаков (агрегаты ТЗ + расширения + AML-паттерны).

Урок из v2: scale_pos_weight=4.82 схлопывал обучение за 3-4 итерации — модель
сразу переобучалась на редкий класс вместо честного ранжирования. AUC не требует
балансировки классов так, как F1/accuracy, поэтому пробуем без неё и с мягкой
регуляризацией вместо агрессивного веса.

Дополнительно: CatBoost как независимая вторая модель — если она подтверждает
похожий уровень AUC, это значит, что мы упёрлись в реальный потолок сигнала в
данных, а не в баг конкретного алгоритма.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import lightgbm as lgb

D = Path(__file__).parent

train = pd.read_parquet(D / "train_features_v3.parquet")
test = pd.read_parquet(D / "test_features_v3.parquet")

feature_cols = [c for c in train.columns if c not in ("signal_id", "eskalatsiya")]
X, y = train[feature_cols], train["eskalatsiya"]
X_test = test[feature_cols]
print(f"признаков: {len(feature_cols)}, train: {X.shape}, test: {X_test.shape}")

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)


def run_cv(params, label):
    oof = np.zeros(len(X))
    aucs = []
    last_model = None
    for tr_idx, val_idx in skf.split(X, y):
        X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]
        model = lgb.LGBMClassifier(**params)
        model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], eval_metric="auc",
                  callbacks=[lgb.early_stopping(100, verbose=False)])
        val_pred = model.predict_proba(X_val)[:, 1]
        oof[val_idx] = val_pred
        aucs.append(roc_auc_score(y_val, val_pred))
        last_model = model
    overall = roc_auc_score(y, oof)
    print(f"[{label}] средний AUC по фолдам: {np.mean(aucs):.4f} +- {np.std(aucs):.4f}, "
          f"по oof: {overall:.4f}, лучшая итерация последнего фолда: {last_model.best_iteration_}")
    return overall, oof, last_model


# без scale_pos_weight, мягкая регуляризация, больше терпения
params_a = dict(random_state=42, n_estimators=600, learning_rate=0.02, num_leaves=15,
                max_depth=4, min_child_samples=30, subsample=0.7, colsample_bytree=0.6,
                reg_alpha=1.0, reg_lambda=1.0, verbosity=-1)
auc_a, oof_a, model_a = run_cv(params_a, "v3 без scale_pos_weight, регуляризация")

# то же, но с лёгким весом класса (не полным 4.82, а вдвое мягче)
pos_weight_soft = ((y == 0).sum() / (y == 1).sum()) ** 0.5  # sqrt смягчает крайность
params_b = dict(**{**params_a, "scale_pos_weight": pos_weight_soft})
auc_b, oof_b, model_b = run_cv(params_b, f"v3 с мягким scale_pos_weight={pos_weight_soft:.2f}")

if auc_b > auc_a:
    best_params, best_auc, best_oof = params_b, auc_b, oof_b
    print("\nлучше вариант B (мягкий вес класса)")
else:
    best_params, best_auc, best_oof = params_a, auc_a, oof_a
    print("\nлучше вариант A (без веса класса)")

importance = pd.DataFrame({
    "feature": feature_cols, "importance": model_a.feature_importances_
}).sort_values("importance", ascending=False)
print("\nтоп-15 признаков по важности (модель A):")
print(importance.head(15).to_string(index=False))
importance.to_csv(D / "feature_importance_v3.csv", index=False)

# финальная модель на всём train с лучшими параметрами
final_n = int(np.median([params_a.get("n_estimators", 600)]))  # разумная фиксированная точка
final_model = lgb.LGBMClassifier(**{**best_params, "n_estimators": 300})
final_model.fit(X, y)
test_pred = final_model.predict_proba(X_test)[:, 1]

submission = pd.DataFrame({"signal_id": test["signal_id"], "ehtimollik": test_pred})
assert submission["signal_id"].duplicated().sum() == 0
assert submission["ehtimollik"].between(0, 1).all()
assert submission.isna().sum().sum() == 0
assert len(submission) == 6000
submission.to_csv(D / "team_E418F3DF_v3.csv", index=False)

print(f"\nИТОГОВЫЙ ЛУЧШИЙ AUC (v3): {best_auc:.4f}")
with open(D / "auc_history.txt", "a") as f:
    f.write(f"v3 (AML-паттерны, {len(feature_cols)} признаков, лучший вариант): {best_auc:.4f}\n")
