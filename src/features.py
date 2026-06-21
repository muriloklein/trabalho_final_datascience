"""
features.py
Engenharia de features para o problema de classificacao (vitoria do mandante /
empate / vitoria do visitante).

PRINCIPIO CENTRAL: toda feature usada para prever uma partida so pode usar
informacao disponivel ANTES daquela partida ser disputada (sem vazamento de
dados / data leakage). Por isso:
  - O rating Elo e atualizado partida a partida, em ordem cronologica, e o
    valor usado como feature e sempre o rating ANTES da atualizacao.
  - As estatisticas de forma recente usam .shift(1) antes de qualquer rolling.
  - O calculo percorre TODO o historico (desde 1872) para "aquecer" os
    ratings e o histórico de confrontos, mesmo que a amostra final de
    modelagem comece em MODELING_START_YEAR.
"""

import pandas as pd
import numpy as np
from collections import defaultdict

# ---------------------------------------------------------------------------
# 1. Rating Elo
# ---------------------------------------------------------------------------

BASE_ELO = 1500.0
HOME_ADVANTAGE = 60.0  # pontos somados ao Elo do mandante so para o calculo
                         # da probabilidade esperada (efeito de jogar em casa)

# K-factor varia com a importancia da partida (mesma ideia usada pelo
# World Football Elo Ratings): partidas de mata-mata/torneios importantes
# pesam mais do que amistosos.
K_BY_TOURNAMENT = {
    "World Cup": 45,
    "Continental/Major": 35,
    "WC Qualifiers": 25,
    "Other Qualifiers": 25,
    "Regional/Minor Cup": 20,
    "Friendly": 15,
}


def _expected_score(rating_a: float, rating_b: float) -> float:
    return 1.0 / (1.0 + 10 ** ((rating_b - rating_a) / 400.0))


def compute_elo_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Percorre o dataset em ordem cronologica calculando o rating Elo de cada
    selecao. Retorna o df original com duas colunas novas:
      home_elo_pre, away_elo_pre -> rating de cada time ANTES da partida.
    """
    df = df.sort_values("date").reset_index(drop=True)
    ratings = defaultdict(lambda: BASE_ELO)

    home_elo_pre = np.empty(len(df))
    away_elo_pre = np.empty(len(df))

    for i, row in enumerate(df.itertuples()):
        home, away = row.home_team, row.away_team
        r_home, r_away = ratings[home], ratings[away]
        home_elo_pre[i] = r_home
        away_elo_pre[i] = r_away

        k = K_BY_TOURNAMENT.get(row.tournament_category, 20)
        expected_home = _expected_score(r_home + HOME_ADVANTAGE, r_away)

        if row.outcome == "home_win":
            actual_home = 1.0
        elif row.outcome == "away_win":
            actual_home = 0.0
        else:
            actual_home = 0.5

        # margem de gols aumenta levemente o ajuste (goleadas pesam mais)
        margin_mult = np.log(abs(row.goal_diff) + 1) + 1.0

        delta = k * margin_mult * (actual_home - expected_home)
        ratings[home] = r_home + delta
        ratings[away] = r_away - delta

    df["home_elo_pre"] = home_elo_pre
    df["away_elo_pre"] = away_elo_pre
    df["elo_diff"] = df["home_elo_pre"] - df["away_elo_pre"]
    return df


# ---------------------------------------------------------------------------
# 2. Forma recente (rolling, por selecao) e dias de descanso
# ---------------------------------------------------------------------------

def _build_long_format(df: pd.DataFrame) -> pd.DataFrame:
    """Transforma o dataset (1 linha = 1 partida) em formato longo
    (1 linha = 1 selecao em 1 partida), facilitando o rolling por time."""
    home = df[["match_id", "date", "home_team", "home_score", "away_score"]].rename(
        columns={"home_team": "team", "home_score": "goals_for", "away_score": "goals_against"}
    )
    home["is_home"] = True

    away = df[["match_id", "date", "away_team", "home_score", "away_score"]].rename(
        columns={"away_team": "team", "away_score": "goals_for", "home_score": "goals_against"}
    )
    away["is_home"] = False

    long_df = pd.concat([home, away], ignore_index=True)
    long_df["win"] = (long_df["goals_for"] > long_df["goals_against"]).astype(int)
    long_df["draw"] = (long_df["goals_for"] == long_df["goals_against"]).astype(int)
    long_df["goal_diff"] = long_df["goals_for"] - long_df["goals_against"]
    long_df = long_df.sort_values(["team", "date"]).reset_index(drop=True)
    return long_df


def compute_form_features(df: pd.DataFrame, windows=(5, 10)) -> pd.DataFrame:
    """
    Para cada selecao, calcula, ANTES de cada partida (usando shift(1)):
      - taxa de vitorias nas ultimas N partidas
      - saldo de gols medio nas ultimas N partidas
      - dias desde a ultima partida (descanso)
    Depois reanexa essas estatisticas de volta no formato largo
    (home_form_*, away_form_*).
    """
    long_df = _build_long_format(df)

    grouped = long_df.groupby("team")
    for w in windows:
        long_df[f"win_rate_last{w}"] = grouped["win"].transform(
            lambda s: s.shift(1).rolling(w, min_periods=1).mean()
        )
        long_df[f"goal_diff_last{w}"] = grouped["goal_diff"].transform(
            lambda s: s.shift(1).rolling(w, min_periods=1).mean()
        )

    long_df["prev_match_date"] = grouped["date"].shift(1)
    long_df["rest_days"] = (long_df["date"] - long_df["prev_match_date"]).dt.days
    long_df["matches_played_before"] = grouped.cumcount()

    feature_cols = [c for c in long_df.columns if "last" in c] + ["rest_days", "matches_played_before"]

    home_feats = long_df[long_df["is_home"]][["match_id"] + feature_cols]
    home_feats = home_feats.rename(columns={c: f"home_{c}" for c in feature_cols})

    away_feats = long_df[~long_df["is_home"]][["match_id"] + feature_cols]
    away_feats = away_feats.rename(columns={c: f"away_{c}" for c in feature_cols})

    df = df.merge(home_feats, on="match_id", how="left")
    df = df.merge(away_feats, on="match_id", how="left")

    # Preenche times em sua primeira partida (sem historico ainda)
    for w in windows:
        for side in ("home", "away"):
            df[f"{side}_win_rate_last{w}"] = df[f"{side}_win_rate_last{w}"].fillna(0.45)
            df[f"{side}_goal_diff_last{w}"] = df[f"{side}_goal_diff_last{w}"].fillna(0.0)
    df["home_rest_days"] = df["home_rest_days"].fillna(180)
    df["away_rest_days"] = df["away_rest_days"].fillna(180)
    df["home_matches_played_before"] = df["home_matches_played_before"].fillna(0)
    df["away_matches_played_before"] = df["away_matches_played_before"].fillna(0)

    return df


# ---------------------------------------------------------------------------
# 3. Historico de confrontos diretos (head-to-head)
# ---------------------------------------------------------------------------

def compute_head_to_head(df: pd.DataFrame) -> pd.DataFrame:
    """
    Para cada partida, calcula a taxa de vitorias do mandante (sobre este
    adversario especifico) e o numero de confrontos anteriores entre os
    dois times, usando SOMENTE confrontos anteriores aquela data.
    """
    df = df.sort_values("date").reset_index(drop=True)
    h2h_record = defaultdict(lambda: [0, 0, 0])  # [vitorias_timeA, empates, vitorias_timeB]

    h2h_matches_played = np.empty(len(df))
    h2h_home_win_rate = np.empty(len(df))

    for i, row in enumerate(df.itertuples()):
        key = tuple(sorted([row.home_team, row.away_team]))
        wins_a, draws, wins_b = h2h_record[key]
        total = wins_a + draws + wins_b
        h2h_matches_played[i] = total

        if total == 0:
            h2h_home_win_rate[i] = 0.45  # sem historico: usa media global
        else:
            # wins_a sempre se refere ao time que vem primeiro na ordenacao alfabetica
            first_team = key[0]
            wins_home = wins_a if row.home_team == first_team else wins_b
            h2h_home_win_rate[i] = (wins_home + 0.5 * draws) / total

        # atualiza o registro depois de extrair a feature (pre-partida)
        first_team = key[0]
        if row.outcome == "draw":
            h2h_record[key][1] += 1
        elif (row.outcome == "home_win") == (row.home_team == first_team):
            h2h_record[key][0] += 1
        else:
            h2h_record[key][2] += 1

    df["h2h_matches_played"] = h2h_matches_played
    df["h2h_home_win_rate"] = h2h_home_win_rate
    return df


def build_feature_dataset(df_clean: pd.DataFrame) -> pd.DataFrame:
    """Pipeline completo de features, aplicado sobre o historico inteiro."""
    df = df_clean.copy()
    df = compute_elo_features(df)
    df = compute_form_features(df)
    df = compute_head_to_head(df)
    return df
