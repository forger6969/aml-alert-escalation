#!/usr/bin/env python3
"""Feature engineering v3 — добавлены признаки на основе типовых паттернов AML,
которых нет в буквальном тексте ТЗ, потому что v1/v2 (агрегаты по ТЗ + очевидные
добавки) дали AUC ~0.59-0.60, почти случайность, и линейная корреляция признаков
с целью не превышает 0.08 — сигнал не в простых средних.

Добавлено (типовые AML red flags при отсутствии данных о контрагенте):
  - интервалы между последовательными транзакциями (короткие серии — burst)
  - максимум транзакций за один день (пиковая активность)
  - "layering": пары приход+расход в один день похожего объёма — классический
    паттерн быстрого перегона денег
  - всплеск активности перед сигналом относительно средней скорости за всю историю
  - аномальность последней транзакции относительно собственной истории клиента
"""
import pandas as pd
import numpy as np
from pathlib import Path

D = Path.home() / "Downloads" / "fintech_data"  # поправь на свой путь к данным
OUT = Path(__file__).parent
TXN_TYPES = ["bank_otkazmasi", "karta", "naqd", "xalqaro"]
RECENT_WINDOWS = [3, 7, 14]


def build_features(tx: pd.DataFrame, signals: pd.DataFrame) -> pd.DataFrame:
    tx = tx.merge(signals[["signal_id", "signal_sanasi"]], on="signal_id", how="left")
    tx["signal_sanasi"] = pd.to_datetime(tx["signal_sanasi"])
    tx["days_before_signal"] = (tx["signal_sanasi"] - tx["tranzaksiya_vaqti"]).dt.total_seconds() / 86400
    tx["hour"] = tx["tranzaksiya_vaqti"].dt.hour
    tx["weekday"] = tx["tranzaksiya_vaqti"].dt.weekday
    tx["is_night"] = tx["hour"].between(0, 5).astype(int)
    tx["is_weekend"] = (tx["weekday"] >= 5).astype(int)
    tx["date_only"] = tx["tranzaksiya_vaqti"].dt.date
    tx = tx.sort_values(["signal_id", "tranzaksiya_vaqti"])

    g = tx.groupby("signal_id")
    feats = pd.DataFrame(index=g.size().index)
    feats["n_txn"] = g.size()

    kirim_mask = tx["kirim_chiqim"] == "kirim"
    chiqim_mask = tx["kirim_chiqim"] == "chiqim"
    kirim = tx[kirim_mask].groupby("signal_id").size()
    chiqim = tx[chiqim_mask].groupby("signal_id").size()
    feats["n_kirim"] = kirim.reindex(feats.index, fill_value=0)
    feats["n_chiqim"] = chiqim.reindex(feats.index, fill_value=0)
    feats["kirim_chiqim_ratio"] = feats["n_kirim"] / (feats["n_chiqim"] + 1)
    feats["kirim_share"] = feats["n_kirim"] / feats["n_txn"]

    kirim_sum = tx[kirim_mask].groupby("signal_id")["miqdor_indeksi"].sum()
    chiqim_sum = tx[chiqim_mask].groupby("signal_id")["miqdor_indeksi"].sum()
    feats["kirim_sum"] = kirim_sum.reindex(feats.index, fill_value=0)
    feats["chiqim_sum"] = chiqim_sum.reindex(feats.index, fill_value=0)
    feats["net_flow"] = feats["kirim_sum"] - feats["chiqim_sum"]
    feats["turnover"] = feats["kirim_sum"].abs() + feats["chiqim_sum"].abs()
    feats["net_flow_ratio"] = feats["net_flow"] / (feats["turnover"] + 1e-6)

    for t in TXN_TYPES:
        sub = tx[tx["tranzaksiya_turi"] == t]
        cnt = sub.groupby("signal_id").size()
        feats[f"share_{t}"] = cnt.reindex(feats.index, fill_value=0) / feats["n_txn"]
        feats[f"mean_amt_{t}"] = sub.groupby("signal_id")["miqdor_indeksi"].mean().reindex(feats.index, fill_value=0)
        feats[f"sum_amt_{t}"] = sub.groupby("signal_id")["miqdor_indeksi"].sum().reindex(feats.index, fill_value=0)

    feats["amt_min"] = g["miqdor_indeksi"].min()
    feats["amt_max"] = g["miqdor_indeksi"].max()
    feats["amt_mean"] = g["miqdor_indeksi"].mean()
    feats["amt_std"] = g["miqdor_indeksi"].std().fillna(0)
    feats["amt_range"] = feats["amt_max"] - feats["amt_min"]
    feats["amt_median"] = g["miqdor_indeksi"].median()
    feats["amt_q25"] = g["miqdor_indeksi"].quantile(0.25)
    feats["amt_q75"] = g["miqdor_indeksi"].quantile(0.75)
    feats["amt_iqr"] = feats["amt_q75"] - feats["amt_q25"]
    feats["amt_cv"] = feats["amt_std"] / (feats["amt_mean"].abs() + 1e-6)

    feats["night_share"] = g["is_night"].mean()
    feats["weekend_share"] = g["is_weekend"].mean()
    feats["n_unique_hours"] = tx.groupby("signal_id")["hour"].nunique().reindex(feats.index, fill_value=0)

    for w in RECENT_WINDOWS:
        recent = tx[(tx["days_before_signal"] >= 0) & (tx["days_before_signal"] <= w)]
        rc = recent.groupby("signal_id").size()
        feats[f"n_txn_last_{w}d"] = rc.reindex(feats.index, fill_value=0)
        feats[f"share_txn_last_{w}d"] = feats[f"n_txn_last_{w}d"] / feats["n_txn"]

    feats["days_since_last_txn"] = g["days_before_signal"].min()
    feats["days_since_first_txn"] = g["days_before_signal"].max()
    feats["history_span_days"] = feats["days_since_first_txn"] - feats["days_since_last_txn"]

    n_unique_days = tx.groupby("signal_id")["date_only"].nunique()
    feats["n_unique_days"] = n_unique_days.reindex(feats.index, fill_value=1)
    feats["txn_velocity"] = feats["n_txn"] / feats["history_span_days"].clip(lower=1)
    feats["avg_gap_days"] = feats["history_span_days"] / feats["n_txn"].clip(lower=1)

    # --- v3: интервалы между последовательными транзакциями (burst detection) ---
    tx["prev_time"] = tx.groupby("signal_id")["tranzaksiya_vaqti"].shift(1)
    tx["gap_hours"] = (tx["tranzaksiya_vaqti"] - tx["prev_time"]).dt.total_seconds() / 3600
    gap_stats = tx.groupby("signal_id")["gap_hours"].agg(["min", "mean", "std"])
    feats["gap_min_hours"] = gap_stats["min"].reindex(feats.index)
    feats["gap_mean_hours"] = gap_stats["mean"].reindex(feats.index)
    feats["gap_std_hours"] = gap_stats["std"].reindex(feats.index).fillna(0)
    # доля транзакций с зазором меньше часа от предыдущей — признак автоматизированных серий
    tx["is_burst"] = (tx["gap_hours"] < 1).astype(float)
    feats["burst_share"] = tx.groupby("signal_id")["is_burst"].mean().reindex(feats.index).fillna(0)

    # --- v3: максимум транзакций за один день (пиковая активность) ---
    per_day = tx.groupby(["signal_id", "date_only"]).size().groupby("signal_id").max()
    feats["max_txn_per_day"] = per_day.reindex(feats.index, fill_value=1)

    # --- v3: layering — приход и расход похожего объёма в один и тот же день ---
    daily = tx.groupby(["signal_id", "date_only", "kirim_chiqim"])["miqdor_indeksi"].sum().unstack(fill_value=0)
    if "kirim" not in daily.columns:
        daily["kirim"] = 0
    if "chiqim" not in daily.columns:
        daily["chiqim"] = 0
    daily["both_present"] = (daily["kirim"] != 0) & (daily["chiqim"] != 0)
    daily["similarity"] = 1 - (daily["kirim"] - daily["chiqim"]).abs() / (daily["kirim"].abs() + daily["chiqim"].abs() + 1e-6)
    layering_days = daily[daily["both_present"] & (daily["similarity"] > 0.7)].groupby("signal_id").size()
    feats["layering_days"] = layering_days.reindex(feats.index, fill_value=0)
    feats["layering_share"] = feats["layering_days"] / feats["n_unique_days"]

    # --- v3: всплеск активности перед сигналом относительно средней скорости за всю историю ---
    avg_daily_rate = feats["n_txn"] / feats["history_span_days"].clip(lower=1)
    recent3 = tx[(tx["days_before_signal"] >= 0) & (tx["days_before_signal"] <= 3)]
    recent3_cnt = recent3.groupby("signal_id").size().reindex(feats.index, fill_value=0)
    feats["spike_ratio_3d"] = (recent3_cnt / 3) / (avg_daily_rate + 1e-6)

    # --- v3: аномальность последней транзакции относительно собственной истории ---
    last_txn = tx.sort_values("tranzaksiya_vaqti").groupby("signal_id").tail(1).set_index("signal_id")
    feats["last_txn_amt"] = last_txn["miqdor_indeksi"].reindex(feats.index)
    feats["last_txn_zscore"] = (feats["last_txn_amt"] - feats["amt_mean"]) / (feats["amt_std"] + 1e-6)

    feats = feats.reset_index()
    return feats


def main():
    print("читаю сигналы...")
    train_sig = pd.read_csv(D / "train_signals.csv")
    test_sig = pd.read_csv(D / "test_signals.csv")

    print("читаю транзакции train...")
    train_tx = pd.read_parquet(D / "train_transactions.parquet")
    print("строю признаки train (v3, AML-паттерны)...")
    train_feats = build_features(train_tx, train_sig)
    train_feats = train_feats.merge(train_sig[["signal_id", "eskalatsiya"]], on="signal_id")
    train_feats.to_parquet(OUT / "train_features_v3.parquet", index=False)
    print(f"train_features_v3: {train_feats.shape}")
    del train_tx

    print("читаю транзакции test...")
    test_tx = pd.read_parquet(D / "test_transactions.parquet")
    print("строю признаки test (v3)...")
    test_feats = build_features(test_tx, test_sig)
    test_feats.to_parquet(OUT / "test_features_v3.parquet", index=False)
    print(f"test_features_v3: {test_feats.shape}")

    n_feats = len([c for c in train_feats.columns if c not in ("signal_id", "eskalatsiya")])
    print(f"\nвсего признаков: {n_feats}")

    # быстрая проверка корреляции новых v3-фич с целью
    new_cols = ["gap_min_hours", "gap_mean_hours", "gap_std_hours", "burst_share",
                "max_txn_per_day", "layering_days", "layering_share",
                "spike_ratio_3d", "last_txn_amt", "last_txn_zscore"]
    corr = train_feats[new_cols + ["eskalatsiya"]].corr()["eskalatsiya"].drop("eskalatsiya")
    print("\nкорреляция новых AML-признаков с целью:")
    print(corr.sort_values(key=abs, ascending=False).to_string())


if __name__ == "__main__":
    main()
