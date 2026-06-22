# ⚽ Previsão de Resultados de Partidas Internacionais de Futebol (1872–2026)

Projeto final da disciplina de Data Science — UPF.

**Grupo:** Murilo K. Klein (199121), Guilherme S. Machado (196890) , Samuel V. Zibetti (200388)

## 1. Tema e objetivo

O projeto busca responder à pergunta: **é possível prever o resultado de uma
partida internacional de futebol (vitória do mandante, empate ou vitória do
visitante) a partir do histórico de desempenho das seleções envolvidas?**

Para isso, construímos um pipeline completo de Data Science: coleta e limpeza de
dados, engenharia de features (rating Elo, forma recente, confrontos diretos),
modelagem com diferentes algoritmos de classificação, avaliação com validação
temporal, e um dashboard interativo para explorar os dados e simular previsões, inclusive para partidas reais da Copa do Mundo 2026 ainda não disputadas no
momento em que este projeto foi desenvolvido.

## 2. Dataset

- **Origem:** [International football results from 1872 to 2026](https://www.kaggle.com/datasets/martj42/international-football-results-from-1872-to-2017)
  (Kaggle, mantido por martj42), espelhado a partir do repositório
  [martj42/international_results](https://github.com/martj42/international_results)
  no GitHub. Licença **CC0 1.0** (domínio público). Os arquivos brutos estão
  incluídos neste repositório em `data/raw/` para garantir reprodutibilidade.
- **Conteúdo:** 49.425 partidas internacionais masculinas disputadas entre
  30/11/1872 e 16/06/2026, com seleção mandante e visitante, placar, tipo de
  competição (`tournament`), cidade, país e se a partida foi em campo neutro.
  Inclui também `shootouts.csv` (vencedores de disputas de pênaltis) e
  `former_names.csv` (referência de nomes históricos de seleções).
- **Limpeza e transformações** (`src/data_prep.py`): remoção de partidas
  futuras/sem placar, conversão de tipos, ordenação cronológica, criação da
  variável-alvo (`outcome`) e de uma categorização do tipo de torneio.
- **Engenharia de features** (`src/features.py`): rating Elo calculado partida
  a partida (sem vazamento temporal), forma recente (últimos 5/10 jogos),
  histórico de confrontos diretos entre as duas seleções, dias de descanso.
- **Amostra de modelagem:** partidas a partir do ano 2000 (~25 mil partidas),
  usando o histórico completo desde 1872 apenas para calcular as features de
  forma e rating com mais precisão.

## 3. Estrutura do repositório

```
├── data/
│   ├── raw/                  # dados brutos baixados da fonte (CC0)
│   └── processed/            # dataset com as features já calculadas
├── notebooks/
│   ├── 01_EDA.ipynb          # análise exploratória, hipóteses e validação
│   └── 02_modeling.ipynb     # treino, avaliação e discussão dos modelos
├── src/
│   ├── data_prep.py          # carregamento e limpeza
│   ├── features.py           # Elo, forma recente, head-to-head
│   └── modeling.py           # split temporal, modelos, métricas
├── dashboard/
│   └── app.py                # dashboard interativo (Streamlit)
├── outputs/
│   ├── figures/               # gráficos gerados (EDA + avaliação)
│   ├── metrics.json           # métricas de todos os modelos testados
│   └── models/                # modelos treinados (.pkl)
├── run_pipeline.py            # roda o pipeline completo (dados → modelos)
├── generate_figures.py        # gera os gráficos a partir dos dados processados
├── requirements.txt
└── README.md
```

## 4. Como executar

```bash
# 1. Criar ambiente e instalar dependências
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 2. Rodar o pipeline completo (limpeza → features → treino → avaliação)
python run_pipeline.py

# 3. Gerar os gráficos de EDA / avaliação
python generate_figures.py

# 4. Abrir o dashboard interativo
streamlit run dashboard/app.py
```

Os notebooks em `notebooks/` já estão com as saídas salvas e podem ser
visualizados diretamente no GitHub, sem precisar reexecutar nada. Para
reexecutá-los: `jupyter nbconvert --to notebook --execute --inplace notebooks/NOME.ipynb`
(executar a partir da raiz do projeto, ou ajustar o `sys.path` se executar de
dentro de `notebooks/`).

## 5. Modelos e resultados

Foram comparados 4 algoritmos de classificação (3 classes: vitória do mandante,
empate, vitória do visitante) contra um baseline (sempre prever a classe
majoritária), com **divisão treino/teste temporal** (treino até 2022, teste a
partir de 2023) — essencial para não vazar informação do futuro para o passado.

| Modelo | Acurácia | F1-macro |
|---|---|---|
| Baseline (classe majoritária) | 47,3% | 0,214 |
| Regressão Logística | 58,0% | 0,528 |
| Árvore de Decisão | 56,8% | 0,526 |
| Random Forest | 57,8% | 0,523 |
| **Gradient Boosting (HistGB)** | **57,6%** | **0,532** |

O Gradient Boosting foi escolhido como modelo de produção (usado no dashboard).
Detalhes, matriz de confusão e discussão de importância de features estão em
`notebooks/02_modeling.ipynb`.

## 6. Dashboard interativo

`streamlit run dashboard/app.py` abre um dashboard com três seções:

- **Explorar dados:** filtros por período, tipo de torneio e seleção, com
  gráficos de distribuição de resultados e evolução do rating Elo.
- **Prever uma partida:** escolha duas seleções (ou um confronto real ainda não
  disputado da Copa do Mundo 2026) e veja a probabilidade prevista para cada
  resultado.
- **Desempenho dos modelos:** comparação de métricas e matriz de confusão do
  melhor modelo.

## 7. Limitações e trabalhos futuros

O modelo não incorpora informações de escalação, lesões de jogadores-chave nem
odds de mercado, as quais são extensões para um trabalho futuro. Outra direção seria
testar uma janela de modelagem ainda mais recente ou pesos de decaimento
temporal nas features de forma.
