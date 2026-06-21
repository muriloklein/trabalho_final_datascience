"""
modeling.py
Treinamento e avaliacao dos modelos de classificacao (3 classes: home_win,
draw, away_win).

DECISAO METODOLOGICA IMPORTANTE: a divisao treino/teste e TEMPORAL, nao
aleatoria. Treinamos com partidas mais antigas e testamos com as mais
recentes. Um k-fold aleatorio "vazaria" informacao do futuro para o passado
(ex.: a forma recente de uma selecao em 2024 ja reflete coisas que só
fariam sentido cronologicamente depois de 2022), alem de nao refletir o uso
real do modelo, que e sempre prever jogos futuros a partir de dados
passados.
"""

from pathlib import Path
import json

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.dummy import DummyClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, f1_score, confusion_matrix, classification_report,
)

TEST_START_YEAR = 2023  # partidas a partir deste ano formam o conjunto de teste

NUMERIC_FEATURES = [
    "elo_diff", "home_elo_pre", "away_elo_pre",
    "home_win_rate_last5", "away_win_rate_last5",
    "home_win_rate_last10", "away_win_rate_last10",
    "home_goal_diff_last5", "away_goal_diff_last5",
    "home_goal_diff_last10", "away_goal_diff_last10",
    "home_rest_days", "away_rest_days",
    "home_matches_played_before", "away_matches_played_before",
    "h2h_home_win_rate", "h2h_matches_played",
]
CATEGORICAL_FEATURES = ["tournament_category"]
BOOLEAN_FEATURES = ["neutral"]
TARGET = "outcome"


def prepare_X_y(df: pd.DataFrame):
    """Monta a matriz de features (com one-hot para a categoria de torneio)
    e o vetor-alvo a partir do dataframe ja com as features calculadas."""
    X = df[NUMERIC_FEATURES + BOOLEAN_FEATURES].copy()
    X[BOOLEAN_FEATURES] = X[BOOLEAN_FEATURES].astype(int)
    cat_dummies = pd.get_dummies(df[CATEGORICAL_FEATURES], prefix="t")
    X = pd.concat([X, cat_dummies], axis=1)
    y = df[TARGET]
    return X, y


def time_based_split(df: pd.DataFrame, test_start_year: int = TEST_START_YEAR):
    train_df = df[df["year"] < test_start_year]
    test_df = df[df["year"] >= test_start_year]
    return train_df, test_df


def get_models() -> dict:
    return {
        "Baseline (classe majoritaria)": DummyClassifier(strategy="most_frequent"),
        "Regressao Logistica": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(class_weight="balanced", max_iter=2000)),
        ]),
        "Arvore de Decisao": DecisionTreeClassifier(
            class_weight="balanced", max_depth=8, min_samples_leaf=50, random_state=42
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, max_depth=12, min_samples_leaf=20,
            class_weight="balanced", random_state=42, n_jobs=-1
        ),
        "Gradient Boosting (HistGB)": HistGradientBoostingClassifier(
            max_iter=300, max_depth=6, class_weight="balanced", random_state=42
        ),
    }


def evaluate_model(name, model, X_train, y_train, X_test, y_test) -> dict:
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    acc = accuracy_score(y_test, preds)
    f1_macro = f1_score(y_test, preds, average="macro")
    cm = confusion_matrix(y_test, preds, labels=["home_win", "draw", "away_win"])
    report = classification_report(
        y_test, preds, labels=["home_win", "draw", "away_win"],
        output_dict=True, zero_division=0,
    )

    return {
        "model_name": name,
        "accuracy": acc,
        "f1_macro": f1_macro,
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
        "fitted_model": model,
    }


def run_all_models(X_train, y_train, X_test, y_test) -> list:
    results = []
    for name, model in get_models().items():
        res = evaluate_model(name, model, X_train, y_train, X_test, y_test)
        results.append(res)
        print(f"{name:32s} | acc={res['accuracy']:.3f} | f1_macro={res['f1_macro']:.3f}")
    return results


def save_metrics(results: list, path: Path):
    """Salva metricas (sem o objeto do modelo) em JSON, para uso no relatorio/dashboard."""
    serializable = [
        {k: v for k, v in r.items() if k != "fitted_model"} for r in results
    ]
    with open(path, "w") as f:
        json.dump(serializable, f, indent=2, ensure_ascii=False)
