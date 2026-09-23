#!/usr/bin/env python3
"""EDA-дашборд для сдачи хакатона. Streamlit, публичный, без логина.

Показывает: базовую статистику по данным, баланс классов, распределение ключевых
признаков по классам, важность признаков модели, качество (AUC) на кросс-валидации.
"""
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

st.set_page_config(page_title="AML Alert Escalation — EDA", layout="wide")

D = Path(__file__).parent

st.title("AML Alert Escalation — анализ и модель")
st.caption(
    "Предсказание вероятности эскалации сигнала финансового мониторинга "
    "по агрегированной истории транзакций клиента."
)

train = pd.read_parquet(D / "train_features_v3.parquet")
importance = pd.read_csv(D / "feature_importance_v3.csv")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Сигналов в train", f"{len(train):,}")
col2.metric("Признаков", len([c for c in train.columns if c not in ("signal_id", "eskalatsiya")]))
pos_rate = train["eskalatsiya"].mean()
col3.metric("Доля эскалированных", f"{pos_rate*100:.1f}%")

auc_path = D / "auc_history.txt"
last_auc = "—"
if auc_path.exists():
    lines = [l.strip() for l in auc_path.read_text().splitlines() if l.strip()]
    if lines:
        last_auc = lines[-1].split(":")[-1].strip()
col4.metric("ROC-AUC (кросс-валидация)", last_auc)

st.divider()

st.header("Баланс классов")
c1, c2 = st.columns([1, 2])
with c1:
    counts = train["eskalatsiya"].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.pie(counts, labels=["не эскалирован", "эскалирован"], autopct="%1.1f%%",
           colors=["steelblue", "crimson"])
    ax.set_title("Распределение классов")
    st.pyplot(fig)
with c2:
    st.dataframe(counts.rename("количество").to_frame())
    st.caption(
        "Умеренный дисбаланс (≈17% положительного класса). "
        "Модель обучалась со стратифицированной кросс-валидацией, "
        "сохраняющей это соотношение в каждом фолде."
    )

st.divider()

st.header("Различие эскалированных и обычных сигналов")
feature_for_plot = st.selectbox(
    "Признак для сравнения распределений",
    options=["amt_mean", "n_txn", "night_share", "layering_share", "burst_share", "spike_ratio_3d"],
    index=0,
)
fig2, ax2 = plt.subplots(figsize=(8, 4))
for cls, color, label in [(0, "steelblue", "не эскалирован"), (1, "crimson", "эскалирован")]:
    subset = train[train["eskalatsiya"] == cls][feature_for_plot]
    ax2.hist(subset, bins=40, alpha=0.5, color=color, label=label, density=True)
ax2.set_xlabel(feature_for_plot)
ax2.set_ylabel("плотность")
ax2.legend()
st.pyplot(fig2)

st.divider()

st.header("Важность признаков модели")
st.caption("LightGBM, суммарная важность (split-based) по кросс-валидации")
top_n = st.slider("Сколько признаков показать", 5, 30, 15)
fig3, ax3 = plt.subplots(figsize=(8, max(4, top_n * 0.3)))
top_imp = importance.head(top_n).sort_values("importance")
ax3.barh(top_imp["feature"], top_imp["importance"], color="teal")
ax3.set_xlabel("важность")
st.pyplot(fig3)

st.divider()

st.header("О модели")
st.markdown(
    """
- **Признаки**: агрегация транзакций на уровне сигнала — объём операций, приход/расход,
  доли по типам, статистика по индексу суммы, активность перед сигналом, плюс признаки
  из практики антифрода: интервалы между операциями (burst), "layering" (быстрый
  перегон денег в один день), всплеск активности перед сигналом.
- **Модель**: LightGBM, стратифицированный 5-fold, усреднение по нескольким seed для
  устойчивости оценки.
- **Метрика**: ROC-AUC на out-of-fold предсказаниях (честная оценка — модель не видела
  эти данные на этапе своего обучения).
- **Ограничение честно**: `miqdor_indeksi` — уже нормализованный индекс суммы, не сырая
  валюта, и в данных нет информации о контрагенте операции — это ограничивает
  потолок достижимого качества по сравнению с продакшен-системами антифрода,
  где обычно есть граф связей между счетами.
"""
)
