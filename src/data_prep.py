"""
data_prep.py
Carregamento e limpeza da base de resultados de partidas internacionais de futebol.

Fonte dos dados: martj42/international_results (GitHub, licença CC0 1.0 - dominio
publico), o mesmo repositorio que alimenta o dataset no Kaggle
"International football results from 1872 to 2026".
"""

from pathlib import Path
import pandas as pd

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"

# A partir de qual ano as partidas entram na amostra de MODELAGEM (treino/teste).
# Mantemos o historico completo (desde 1872) para calcular features de forma e
# rating Elo, mas só avaliamos o modelo no futebol "moderno", evitando misturar
# eras com regras, calendario de competicoes e nivel tecnico muito diferentes.
MODELING_START_YEAR = 2000


def load_raw_results() -> pd.DataFrame:
    """Carrega o results.csv bruto, sem nenhuma limpeza."""
    return pd.read_csv(RAW_DIR / "results.csv", parse_dates=["date"])


def load_shootouts() -> pd.DataFrame:
    """Carrega o shootouts.csv (vencedores de disputas de penaltis)."""
    return pd.read_csv(RAW_DIR / "shootouts.csv", parse_dates=["date"])


def load_former_names() -> pd.DataFrame:
    """Carrega o dicionario de nomes historicos de selecoes (referencia)."""
    return pd.read_csv(RAW_DIR / "former_names.csv")


def clean_results(df: pd.DataFrame) -> pd.DataFrame:
    """
    Limpeza principal do dataset de resultados:
      1. Remove partidas futuras/agendadas sem placar (ex.: jogos da Copa do
         Mundo 2026 ainda nao disputados na data de hoje) - elas nao podem
         compor treino/teste pois nao tem rótulo (resultado) ainda.
      2. Garante tipos corretos (data, placares inteiros).
      3. Ordena cronologicamente - pre-requisito para qualquer feature de
         forma/historico, que so pode olhar para o passado de cada partida.
      4. Cria a variavel-alvo (outcome): home_win / draw / away_win.
      5. Cria um id de partida e flags auxiliares.
    """
    df = df.copy()

    n_before = len(df)
    df = df.dropna(subset=["home_score", "away_score"]).copy()
    n_removed = n_before - len(df)
    if n_removed:
        print(f"[clean_results] Removidas {n_removed} partidas futuras/sem placar "
              f"(ex.: jogos da Copa do Mundo 2026 ainda nao disputados).")

    df["home_score"] = df["home_score"].astype(int)
    df["away_score"] = df["away_score"].astype(int)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    df["match_id"] = df.index

    df["goal_diff"] = df["home_score"] - df["away_score"]
    df["outcome"] = pd.cut(
        df["goal_diff"],
        bins=[-999, -1, 0, 999],
        labels=["away_win", "draw", "home_win"],
    ).astype(str)

    df["year"] = df["date"].dt.year
    df["neutral"] = df["neutral"].astype(bool)

    return df


def categorize_tournament(tournament: str) -> str:
    """
    Agrupa o campo 'tournament' (centenas de valores distintos) em poucas
    categorias relevantes para o modelo e para a EDA.
    """
    t = tournament.lower()
    if "friendly" in t:
        return "Friendly"
    if "world cup qualification" in t or "qualif" in t and "world cup" in t:
        return "WC Qualifiers"
    if t == "fifa world cup":
        return "World Cup"
    if "qualif" in t:
        return "Other Qualifiers"
    if any(k in t for k in ["euro", "copa am", "africa", "asian cup", "gold cup",
                              "nations league", "confederations cup"]):
        return "Continental/Major"
    return "Regional/Minor Cup"


def get_modeling_dataset(df_clean: pd.DataFrame) -> pd.DataFrame:
    """Recorta o dataset limpo para o periodo usado na modelagem (treino+teste)."""
    return df_clean[df_clean["year"] >= MODELING_START_YEAR].copy()


if __name__ == "__main__":
    raw = load_raw_results()
    clean = clean_results(raw)
    clean["tournament_category"] = clean["tournament"].apply(categorize_tournament)
    print(clean.shape)
    print(clean["outcome"].value_counts(normalize=True))
    print(clean["tournament_category"].value_counts())
