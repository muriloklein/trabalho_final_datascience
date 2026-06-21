"""
run_pipeline.py
Executa o pipeline completo do projeto, do dado bruto aos modelos avaliados:

  1. Carrega e limpa os dados brutos
  2. Calcula as features (Elo, forma recente, confrontos diretos)
  3. Salva o dataset processado (usado pelo dashboard)
  4. Separa treino/teste por tempo
  5. Treina e avalia os modelos
  6. Salva metricas (outputs/metrics.json) e o melhor modelo (outputs/models/)

Uso:
    python run_pipeline.py
"""

from pathlib import Path
import json
import joblib

from src.data_prep import load_raw_results, clean_results, categorize_tournament
from src.features import build_feature_dataset
from src.modeling import (
    prepare_X_y, time_based_split, run_all_models, save_metrics, TEST_START_YEAR
)

PROJECT_ROOT = Path(__file__).resolve().parent
PROCESSED_PATH = PROJECT_ROOT / "data" / "processed" / "matches_features.csv"
METRICS_PATH = PROJECT_ROOT / "outputs" / "metrics.json"
MODELS_DIR = PROJECT_ROOT / "outputs" / "models"


def main():
    print("=== 1. Carregando e limpando dados brutos ===")
    raw = load_raw_results()
    clean = clean_results(raw)
    clean["tournament_category"] = clean["tournament"].apply(categorize_tournament)
    print(f"   {len(clean)} partidas limpas (1872-2026, com placar confirmado)")

    print("\n=== 2. Calculando features (Elo, forma, confrontos diretos) ===")
    featured = build_feature_dataset(clean)

    print("\n=== 3. Salvando dataset processado ===")
    PROCESSED_PATH.parent.mkdir(parents=True, exist_ok=True)
    featured.to_csv(PROCESSED_PATH, index=False)
    print(f"   Salvo em {PROCESSED_PATH}")

    print("\n=== 4. Recortando amostra de modelagem e dividindo treino/teste ===")
    from src.data_prep import get_modeling_dataset
    modeling_df = get_modeling_dataset(featured)
    train_df, test_df = time_based_split(modeling_df, TEST_START_YEAR)
    print(f"   Treino: {len(train_df)} partidas (ate {TEST_START_YEAR - 1})")
    print(f"   Teste:  {len(test_df)} partidas ({TEST_START_YEAR} em diante)")

    X_train, y_train = prepare_X_y(train_df)
    X_test, y_test = prepare_X_y(test_df)
    # garante mesmas colunas em treino e teste (one-hot pode diferir)
    X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

    print("\n=== 5. Treinando e avaliando modelos ===")
    results = run_all_models(X_train, y_train, X_test, y_test)

    print("\n=== 6. Salvando metricas e modelos ===")
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    save_metrics(results, METRICS_PATH)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    best = max(results, key=lambda r: r["f1_macro"])
    joblib.dump(best["fitted_model"], MODELS_DIR / "best_model.pkl")
    joblib.dump(list(X_train.columns), MODELS_DIR / "feature_columns.pkl")
    print(f"   Melhor modelo: {best['model_name']} (f1_macro={best['f1_macro']:.3f}) "
          f"salvo em {MODELS_DIR / 'best_model.pkl'}")

    # Re-treina o melhor modelo com TODOS os dados disponiveis (treino+teste)
    # para uso no dashboard (previsao de jogos futuros reais, ex.: Copa 2026)
    print("\n=== 7. Re-treinando melhor modelo com todo o historico disponivel ===")
    from sklearn.base import clone
    X_full, y_full = prepare_X_y(modeling_df)
    X_full = X_full.reindex(columns=X_train.columns, fill_value=0)
    final_model = clone(best["fitted_model"])
    final_model.fit(X_full, y_full)
    joblib.dump(final_model, MODELS_DIR / "production_model.pkl")
    print(f"   Modelo de producao salvo em {MODELS_DIR / 'production_model.pkl'}")

    print("\nPipeline concluido com sucesso.")


if __name__ == "__main__":
    main()
