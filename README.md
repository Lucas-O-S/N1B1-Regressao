# Previsão de Preço de Carros Usados

Pipeline de Análise Exploratória de Dados (EDA) e Machine Learning para prever o valor de revenda (`current price`) de carros usados, a partir de atributos como quilometragem, idade e condição do veículo.

Projeto desenvolvido como exercício da disciplina de Inteligência Artificial / Aprendizado de Máquina, aplicando a metodologia de pipeline completo (EDA → limpeza → preparação → modelagem → avaliação) sobre uma base própria de dados.

## Sobre o dataset

`train.csv` contém **1000 registros** de carros usados, com as seguintes colunas:

| Coluna | Descrição |
|---|---|
| `v.id` | Identificador sequencial do registro (não é atributo preditivo) |
| `on road old` | Preço de tabela original (fábrica) do veículo |
| `on road now` | Preço de tabela atual do modelo |
| `years` | Idade do carro, em anos |
| `km` | Quilometragem rodada |
| `rating` | Avaliação do veículo (escala 1–5) |
| `condition` | Estado de conservação (escala 1–10) |
| `economy` | Economia de combustível |
| `top speed` | Velocidade máxima |
| `hp` | Potência (cavalos) |
| `torque` | Torque do motor |
| `current price` | **Alvo** — valor atual de revenda do carro |

O dataset não possui valores ausentes, duplicatas ou colunas categóricas — todas as 12 colunas já são numéricas.

## Metodologia

O pipeline segue as boas práticas para evitar vazamento de dados (*data leakage*):

1. **EDA** — dimensão, tipos de dados, estatísticas descritivas, distribuições (histogramas e boxplots), matriz de correlação com o alvo.
2. **Separação treino/teste** (80/20, `random_state=42`) — feita **antes** de qualquer preparação dos dados.
3. **Engenharia de atributos** — testados três atributos derivados (`depreciacao`, `km_por_ano`, `hp_por_torque`), avaliados por correlação com o alvo.
4. **Escalonamento** (`StandardScaler`) — necessário pela diferença de escala entre colunas (ex.: preços na casa de centenas de milhares vs. `rating` de 1 a 5).
5. **Pipeline scikit-learn** (`Pipeline` + transformador customizado) — encapsula engenharia de atributos e escalonamento em um único objeto, reaplicável ao conjunto de teste sem repetir código.
6. **Treinamento e comparação de modelos** — Regressão Linear, Árvore de Decisão e Random Forest, avaliados por validação cruzada (10 folds).
7. **Avaliação final no conjunto de teste** — métricas RMSE e MAE, e gráfico de dispersão real x previsto.

## Principais achados

- `km` é, isoladamente, o atributo mais correlacionado com o preço (**-0,94**) — quanto maior a quilometragem, menor o valor do carro, numa relação fortemente linear.
- Os atributos derivados testados (`depreciacao`, `km_por_ano`, `hp_por_torque`) **não melhoraram** a correlação com o alvo em relação às colunas originais; `km_por_ano`, em particular, diluiu o sinal forte que `km` já tinha sozinho.
- Como a relação entre os atributos e o preço é predominantemente linear, a **Regressão Linear superou os modelos baseados em árvore** — resultado oposto ao que costuma ocorrer em datasets com relações não lineares (como o de imóveis usado em aula).

### Comparação de modelos (validação cruzada, 10 folds)

| Modelo | RMSE médio | Desvio-padrão |
|---|---|---|
| **Regressão Linear** | **8.675** | 543 |
| Random Forest | 18.763 | 1.377 |
| Árvore de Decisão | 29.115 | 1.767 |

A Árvore de Decisão apresentou RMSE = 0 quando avaliada no próprio conjunto de treino — sintoma clássico de *overfitting* (decorou os dados de treino em vez de generalizar), confirmado pelo erro alto na validação cruzada.

### Avaliação final no conjunto de teste (Regressão Linear)

| Métrica | Valor |
|---|---|
| RMSE | 8.948 |
| MAE | 7.497 |

## Como executar

O projeto foi desenvolvido para rodar no [Google Colab](https://colab.research.google.com), sem necessidade de configuração de ambiente local.

1. Abra o notebook `Aula_03_Pipeline_Completo_EDA_e_Preparacao_de_Dados.ipynb` no Colab.
2. Faça upload do arquivo `train.csv` (via `files.upload()` ou montando o Google Drive — instruções na primeira célula do notebook).
3. Execute as células em ordem, de cima para baixo.

### Dependências

```
numpy
pandas
matplotlib
seaborn
scikit-learn
```

Todas já vêm pré-instaladas no ambiente padrão do Google Colab.

## Estrutura do repositório

```
.
├── train.csv    # dataset de carros usados
├── notebook.ipynb   # pipeline completo de EDA, preparação e modelagem
└── README.md
```
