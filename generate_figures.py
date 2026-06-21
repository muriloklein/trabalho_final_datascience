"""
generate_figures.py
Gera as figuras de Analise Exploratoria e de avaliacao de modelos, salvas em
outputs/figures/. Pensado para ser usado tanto no dashboard quanto no
relatorio tecnico (Quarto/Jupyter).

Uso:
    python generate_figures.py
"""

from pathlib import Path
import json

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid", palette="deep")
PROJECT_ROOT = Path(__file__).resolve().parent
FIG_DIR = PROJECT_ROOT / "outputs" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


def save(fig, name):
    fig.tight_layout()
    fig.savefig(FIG_DIR / name, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  salvo: {name}")


def main():
    df = pd.read_csv(PROJECT_ROOT / "data" / "processed" / "matches_features.csv",
                      parse_dates=["date"])

    print("Gerando figuras de EDA...")

    # 1. Partidas por ano (crescimento do futebol internacional)
    by_year = df.groupby("year").size()
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(by_year.index, by_year.values, color="#1b6f4a", linewidth=1.8)
    ax.set_title("Partidas internacionais por ano (1872-2026)")
    ax.set_xlabel("Ano")
    ax.set_ylabel("Numero de partidas")
    save(fig, "01_matches_per_year.png")

    # 2. Distribuicao do resultado (geral vs amostra de modelagem 2000+)
    overall = df["outcome"].value_counts(normalize=True)
    recent = df[df["year"] >= 2000]["outcome"].value_counts(normalize=True)
    comp = pd.DataFrame({"1872-2026 (completo)": overall, "2000-2026 (amostra de modelagem)": recent})
    comp = comp.reindex(["home_win", "draw", "away_win"])
    fig, ax = plt.subplots(figsize=(7, 4.5))
    comp.plot(kind="bar", ax=ax, color=["#1b6f4a", "#7fb685"])
    ax.set_title("Distribuicao do resultado das partidas")
    ax.set_ylabel("Proporcao")
    ax.set_xlabel("")
    ax.set_xticklabels(["Vitoria do mandante", "Empate", "Vitoria do visitante"], rotation=0)
    save(fig, "02_outcome_distribution.png")

    # 3. Vantagem de jogar em casa ao longo do tempo (por decada)
    df["decade"] = (df["year"] // 10) * 10
    home_adv = df[~df["neutral"]].groupby("decade").apply(
        lambda g: (g["outcome"] == "home_win").mean()
    )
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(home_adv.index.astype(str), home_adv.values, color="#1b6f4a", width=0.7)
    ax.axhline(home_adv.mean(), color="gray", linestyle="--", linewidth=1,
               label=f"media geral ({home_adv.mean():.0%})")
    ax.set_title("Taxa de vitoria do mandante por decada (jogos fora de campo neutro)")
    ax.set_xlabel("Decada")
    ax.set_ylabel("Proporcao de vitorias do mandante")
    ax.legend()
    plt.xticks(rotation=45)
    save(fig, "03_home_advantage_by_decade.png")

    # 4. Top 15 selecoes por rating Elo atual
    last_home = df.sort_values("date").groupby("home_team").tail(1)[
        ["home_team", "date", "home_elo_pre"]
    ].rename(columns={"home_team": "team", "home_elo_pre": "elo"})
    last_away = df.sort_values("date").groupby("away_team").tail(1)[
        ["away_team", "date", "away_elo_pre"]
    ].rename(columns={"away_team": "team", "away_elo_pre": "elo"})
    latest_elo = pd.concat([last_home, last_away]).sort_values("date").groupby("team").tail(1)
    top15 = latest_elo.sort_values("elo", ascending=False).head(15)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(top15["team"][::-1], top15["elo"][::-1], color="#1b6f4a")
    ax.set_title("Top 15 selecoes por rating Elo (junho/2026)")
    ax.set_xlabel("Rating Elo")
    save(fig, "04_top15_elo.png")

    # 5. Comparacao de modelos (accuracy + f1-macro)
    with open(PROJECT_ROOT / "outputs" / "metrics.json") as f:
        metrics = json.load(f)
    names = [m["model_name"] for m in metrics]
    acc = [m["accuracy"] for m in metrics]
    f1m = [m["f1_macro"] for m in metrics]

    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(names))
    width = 0.35
    ax.bar(x - width / 2, acc, width, label="Acuracia", color="#1b6f4a")
    ax.bar(x + width / 2, f1m, width, label="F1-macro", color="#a8d5ba")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=20, ha="right")
    ax.set_title(f"Comparacao de modelos - teste temporal ({metrics[0].get('test_period', '2023+')})")
    ax.set_ylabel("Score")
    ax.legend()
    save(fig, "05_model_comparison.png")

    # 6. Matriz de confusao do melhor modelo
    best = max(metrics, key=lambda m: m["f1_macro"])
    cm = np.array(best["confusion_matrix"])
    labels = ["Vit. mandante", "Empate", "Vit. visitante"]
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Greens", xticklabels=labels,
                yticklabels=labels, ax=ax, cbar=False)
    ax.set_title(f"Matriz de confusao - {best['model_name']}")
    ax.set_xlabel("Previsto")
    ax.set_ylabel("Real")
    save(fig, "06_confusion_matrix_best_model.png")

    print("\nFiguras geradas em", FIG_DIR)


if __name__ == "__main__":
    main()
