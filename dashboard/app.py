"""
dashboard/app.py
Dashboard interativo - Previsao de resultados de partidas internacionais de futebol.

Executar com:
    streamlit run dashboard/app.py
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "matches_features.csv"
MODELS_DIR = PROJECT_ROOT / "outputs" / "models"
METRICS_PATH = PROJECT_ROOT / "outputs" / "metrics.json"

st.set_page_config(page_title="Futebol Internacional - Previsao de Resultados",
                    page_icon="⚽", layout="wide")


@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH, parse_dates=["date"])
    return df


@st.cache_resource
def load_model_assets():
    model = joblib.load(MODELS_DIR / "production_model.pkl")
    feature_cols = joblib.load(MODELS_DIR / "feature_columns.pkl")
    with open(METRICS_PATH) as f:
        metrics = json.load(f)
    return model, feature_cols, metrics


df = load_data()
model, feature_cols, metrics = load_model_assets()

st.title("⚽ Futebol Internacional (1872-2026): Explorador e Preditor de Resultados")
st.caption(
    "Dataset: International Football Results (martj42/Kaggle, CC0). "
    f"{len(df):,} partidas. Modelo: classificacao de resultado "
    "(vitoria do mandante / empate / vitoria do visitante) com features de "
    "rating Elo, forma recente e historico de confrontos diretos."
)

tab_explore, tab_predict, tab_model = st.tabs(
    ["📊 Explorar dados", "🔮 Prever uma partida", "🧪 Desempenho dos modelos"]
)

# ---------------------------------------------------------------------------
# TAB 1: Explorar dados
# ---------------------------------------------------------------------------
with tab_explore:
    st.subheader("Filtros")
    col1, col2, col3 = st.columns(3)
    with col1:
        year_range = st.slider("Periodo", int(df["year"].min()), int(df["year"].max()),
                                (2000, int(df["year"].max())))
    with col2:
        tournaments = st.multiselect("Categoria de torneio",
                                      sorted(df["tournament_category"].unique()),
                                      default=sorted(df["tournament_category"].unique()))
    with col3:
        teams = sorted(set(df["home_team"]) | set(df["away_team"]))
        team_filter = st.multiselect("Filtrar por selecao (opcional)", teams)

    filtered = df[(df["year"] >= year_range[0]) & (df["year"] <= year_range[1])]
    filtered = filtered[filtered["tournament_category"].isin(tournaments)]
    if team_filter:
        filtered = filtered[
            filtered["home_team"].isin(team_filter) | filtered["away_team"].isin(team_filter)
        ]

    st.metric("Partidas no filtro", f"{len(filtered):,}")

    colA, colB = st.columns(2)
    with colA:
        outcome_counts = filtered["outcome"].value_counts(normalize=True).reindex(
            ["home_win", "draw", "away_win"]
        )
        fig = px.bar(
            x=["Vitoria mandante", "Empate", "Vitoria visitante"],
            y=outcome_counts.values,
            labels={"x": "Resultado", "y": "Proporcao"},
            title="Distribuicao de resultados no filtro",
            color_discrete_sequence=["#1b6f4a"],
        )
        st.plotly_chart(fig, width='stretch')

    with colB:
        by_year = filtered.groupby("year").size().reset_index(name="partidas")
        fig2 = px.line(by_year, x="year", y="partidas", title="Partidas por ano (no filtro)",
                        color_discrete_sequence=["#1b6f4a"])
        st.plotly_chart(fig2, width='stretch')

    if team_filter:
        st.subheader(f"Evolucao do rating Elo: {', '.join(team_filter)}")
        elo_rows = []
        for t in team_filter:
            home_rows = filtered[filtered["home_team"] == t][["date", "home_elo_pre"]].rename(
                columns={"home_elo_pre": "elo"})
            away_rows = filtered[filtered["away_team"] == t][["date", "away_elo_pre"]].rename(
                columns={"away_elo_pre": "elo"})
            team_elo = pd.concat([home_rows, away_rows]).sort_values("date")
            team_elo["team"] = t
            elo_rows.append(team_elo)
        elo_df = pd.concat(elo_rows)
        fig3 = px.line(elo_df, x="date", y="elo", color="team", title="Rating Elo ao longo do tempo")
        st.plotly_chart(fig3, width='stretch')

    st.subheader("Amostra dos dados filtrados")
    st.dataframe(
        filtered[["date", "home_team", "away_team", "home_score", "away_score",
                  "tournament", "outcome"]].sort_values("date", ascending=False).head(200),
        width='stretch',
    )

# ---------------------------------------------------------------------------
# TAB 2: Prever uma partida
# ---------------------------------------------------------------------------
with tab_predict:
    st.subheader("Simular a previsao do modelo para um confronto")

    teams_sorted = sorted(set(df["home_team"]) | set(df["away_team"]))

    upcoming = pd.read_csv(PROJECT_ROOT / "data" / "raw" / "results.csv", parse_dates=["date"])
    upcoming = upcoming[upcoming["home_score"].isnull() & (upcoming["date"] >= "2026-06-17")]

    use_upcoming = False
    if len(upcoming) > 0:
        use_upcoming = st.checkbox(
            f"Usar um dos {len(upcoming)} jogos reais ainda nao disputados da Copa do Mundo 2026",
            value=True,
        )

    if use_upcoming and len(upcoming) > 0:
        upcoming_display = upcoming.assign(
            label=lambda d: d["date"].dt.strftime("%d/%m") + " - " + d["home_team"] + " x " + d["away_team"]
        )
        choice = st.selectbox("Escolha a partida", upcoming_display["label"])
        row = upcoming_display[upcoming_display["label"] == choice].iloc[0]
        home_team, away_team = row["home_team"], row["away_team"]
        neutral = bool(row["neutral"])
        tournament_category = "World Cup"
        st.info(f"Partida real da Copa do Mundo 2026: **{home_team} x {away_team}** em {row['date'].strftime('%d/%m/%Y')}")
    else:
        col1, col2, col3 = st.columns(3)
        with col1:
            home_team = st.selectbox("Selecao mandante", teams_sorted, index=teams_sorted.index("Brazil") if "Brazil" in teams_sorted else 0)
        with col2:
            away_team = st.selectbox("Selecao visitante", teams_sorted, index=teams_sorted.index("Argentina") if "Argentina" in teams_sorted else 1)
        with col3:
            tournament_category = st.selectbox(
                "Tipo de partida",
                ["Friendly", "WC Qualifiers", "Other Qualifiers", "Continental/Major", "World Cup", "Regional/Minor Cup"],
            )
        neutral = st.checkbox("Jogo em campo neutro?", value=False)

    def get_latest_team_state(team):
        home_rows = df[df["home_team"] == team]
        away_rows = df[df["away_team"] == team]
        last_home = home_rows.sort_values("date").tail(1)
        last_away = away_rows.sort_values("date").tail(1)
        candidates = []
        if len(last_home):
            candidates.append((last_home["date"].iloc[0], "home", last_home.iloc[0]))
        if len(last_away):
            candidates.append((last_away["date"].iloc[0], "away", last_away.iloc[0]))
        if not candidates:
            return None
        candidates.sort(key=lambda c: c[0])
        _, side, row = candidates[-1]
        prefix = side
        return row, prefix

    if home_team == away_team:
        st.warning("Escolha duas selecoes diferentes.")
    else:
        home_state, home_prefix = get_latest_team_state(home_team)
        away_state, away_prefix = get_latest_team_state(away_team)

        def extract(row, prefix, field):
            return row[f"{prefix}_{field}"]

        feat = {
            "home_elo_pre": extract(home_state, home_prefix, "elo_pre"),
            "away_elo_pre": extract(away_state, away_prefix, "elo_pre"),
            "home_win_rate_last5": extract(home_state, home_prefix, "win_rate_last5"),
            "away_win_rate_last5": extract(away_state, away_prefix, "win_rate_last5"),
            "home_win_rate_last10": extract(home_state, home_prefix, "win_rate_last10"),
            "away_win_rate_last10": extract(away_state, away_prefix, "win_rate_last10"),
            "home_goal_diff_last5": extract(home_state, home_prefix, "goal_diff_last5"),
            "away_goal_diff_last5": extract(away_state, away_prefix, "goal_diff_last5"),
            "home_goal_diff_last10": extract(home_state, home_prefix, "goal_diff_last10"),
            "away_goal_diff_last10": extract(away_state, away_prefix, "goal_diff_last10"),
            "home_rest_days": 14,
            "away_rest_days": 14,
            "home_matches_played_before": extract(home_state, home_prefix, "matches_played_before") + 1,
            "away_matches_played_before": extract(away_state, away_prefix, "matches_played_before") + 1,
        }
        feat["elo_diff"] = feat["home_elo_pre"] - feat["away_elo_pre"]

        key = tuple(sorted([home_team, away_team]))
        h2h = df[
            ((df["home_team"] == home_team) & (df["away_team"] == away_team))
            | ((df["home_team"] == away_team) & (df["away_team"] == home_team))
        ]
        if len(h2h):
            wins_home_persp = (
                ((h2h["home_team"] == home_team) & (h2h["outcome"] == "home_win"))
                | ((h2h["away_team"] == home_team) & (h2h["outcome"] == "away_win"))
            ).sum()
            draws = (h2h["outcome"] == "draw").sum()
            feat["h2h_home_win_rate"] = (wins_home_persp + 0.5 * draws) / len(h2h)
            feat["h2h_matches_played"] = len(h2h)
        else:
            feat["h2h_home_win_rate"] = 0.45
            feat["h2h_matches_played"] = 0

        feat["neutral"] = int(neutral)

        X_pred = pd.DataFrame([feat])
        for cat in ["Friendly", "WC Qualifiers", "Other Qualifiers", "Continental/Major", "World Cup", "Regional/Minor Cup"]:
            X_pred[f"t_{cat}"] = 1 if cat == tournament_category else 0
        X_pred = X_pred.reindex(columns=feature_cols, fill_value=0)

        proba = model.predict_proba(X_pred)[0]
        classes = list(model.classes_)
        proba_map = dict(zip(classes, proba))

        st.markdown(f"### {home_team} 🆚 {away_team}")
        c1, c2, c3 = st.columns(3)
        c1.metric(f"Vitoria - {home_team}", f"{proba_map.get('home_win', 0):.0%}")
        c2.metric("Empate", f"{proba_map.get('draw', 0):.0%}")
        c3.metric(f"Vitoria - {away_team}", f"{proba_map.get('away_win', 0):.0%}")

        fig = px.bar(
            x=[home_team, "Empate", away_team],
            y=[proba_map.get("home_win", 0), proba_map.get("draw", 0), proba_map.get("away_win", 0)],
            labels={"x": "", "y": "Probabilidade"},
            color_discrete_sequence=["#1b6f4a"],
        )
        st.plotly_chart(fig, width='stretch')

        with st.expander("Ver os ratings/forma usados na previsao"):
            st.json({k: (round(v, 1) if isinstance(v, float) else v) for k, v in feat.items()})

# ---------------------------------------------------------------------------
# TAB 3: Desempenho dos modelos
# ---------------------------------------------------------------------------
with tab_model:
    st.subheader("Comparacao dos modelos (conjunto de teste: partidas de 2023 em diante)")
    metrics_df = pd.DataFrame([
        {"Modelo": m["model_name"], "Acuracia": m["accuracy"], "F1-macro": m["f1_macro"]}
        for m in metrics
    ])
    st.dataframe(metrics_df.style.format({"Acuracia": "{:.1%}", "F1-macro": "{:.3f}"}),
                 width='stretch')

    best = max(metrics, key=lambda m: m["f1_macro"])
    st.markdown(f"**Melhor modelo:** {best['model_name']}")

    cm = np.array(best["confusion_matrix"])
    labels = ["Vit. mandante", "Empate", "Vit. visitante"]
    fig = px.imshow(cm, text_auto=True, x=labels, y=labels,
                     labels={"x": "Previsto", "y": "Real"},
                     color_continuous_scale="Greens",
                     title=f"Matriz de confusao - {best['model_name']}")
    st.plotly_chart(fig, width='stretch')

    st.caption(
        "Divisao treino/teste temporal (nao aleatoria): o modelo treina com partidas "
        "mais antigas e e avaliado apenas em partidas futuras em relacao ao treino, "
        "evitando vazamento de informacao."
    )
