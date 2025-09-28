# MVP — Bank Marketing (UCI): Predição de Adesão a Depósito a Prazo

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1Tihy55C9EVroexsQ4am-uXghDv3XHsg7?usp=sharing)

Projeto de **Machine Learning** para prever a **adesão** (`yes`/`no`) de clientes a um **depósito a prazo** (term deposit) em campanhas de **telemarketing bancário**. O objetivo prático é **ranquear clientes** por probabilidade de conversão e **priorizar ligações**.

---

## 🧠 Visão geral

- **Problema:** classificação binária desbalanceada (`y = yes/no`).
- **Métrica principal:** **PR-AUC (Average Precision)**; apoio: **ROC-AUC**, **F1-macro**, **Accuracy**.
- **Cuidado crítico:** a coluna `duration` (duração da ligação) é conhecida **após** a chamada; incluir no treino causaria **vazamento**. Neste projeto, **mostramos sua correlação** apenas para fins didáticos e **excluímos `duration`** da modelagem.

---

## 🗂️ Dados

- **Dataset:** Bank Marketing (UCI).
- **Arquivo:** `bank-full.csv` hospedado no GitHub (separador `;`).


- **Alvo:** `y` (yes/no) → `y_bin` (1/0).
- **Observação:** `duration` **removida** antes do treino/validação.

---

## 🧪 Metodologia

1. **EDA e preparação**
 - Carga via **URL raw** do GitHub.
 - Tipos, ausências, correlação com o alvo.
 - **Remoção de `duration`** para evitar vazamento.
 - Visualizações com `matplotlib` (1 gráfico/figura; sem cores forçadas).

2. **Split + Pipeline**
 - **Split 80/20** estratificado (`random_state=42`).
 - **`ColumnTransformer`**:
   - Numéricas: imputação **mediana** + **StandardScaler**.
   - Categóricas: imputação **moda** + **OneHotEncoder(handle_unknown="ignore")**.

3. **Modelagem**
 - **Baselines:** `LogisticRegression(class_weight='balanced')` e `RandomForestClassifier(class_weight='balanced_subsample')`.
 - **Tuning (GridSearchCV, cv=5; `scoring='average_precision'`):**
   grade em `n_estimators`, `max_depth`, `min_samples_split`, `min_samples_leaf`.

4. **Avaliação**
 - Métricas no **teste**: **PR-AUC** (principal), **ROC-AUC**, **F1-macro**, **Accuracy**.
 - **Curvas ROC e Precisão–Recall**.
 - **Importância de variáveis** (global) + agregação por variável original.

5. **Artefatos**
 - `best_model_etapa2.pkl` — pipeline completo (preprocess + modelo).
 - `run_metadata_etapa2.json` — versões, seed e `best_params`.

---

## 📊 Resultados (resumo)

**Baselines (teste):**
- **Random Forest:** PR-AUC **0.4288**, ROC-AUC 0.7916, F1-macro 0.6306, Acc 0.8937  
- **Logistic Regression:** PR-AUC **0.4093**, ROC-AUC 0.7722, F1-macro 0.6104, Acc 0.7548

**Tuning (GridSearchCV, PR-AUC):**
- **Melhor RF:** `n_estimators=600`, `max_depth=None`, `min_samples_leaf=4`, `min_samples_split=2`  
- **CV best (PR-AUC):** 0.4418  
- **Tempo:** ~1.658,9 s

**Teste final (vencedor — RF tunado):**
- **PR-AUC:** **0.4545**
- ROC-AUC: 0.8035
- **F1-macro:** 0.705
- Accuracy: 0.873

**Conclusão:** o **RF tunado** superou os baselines em **PR-AUC** (~+0,026 vs. RF baseline), confirmando **melhor ranqueamento** para priorização de ligações.

---

## 🚀 Uso prático

- **Priorizar ligações:** use `predict_proba` para **ranquear** clientes e aplique **limiar/percentil** conforme a **capacidade de chamadas** (ex.: top 15%).
- **Métricas operacionais:** monitore **precision@k**, conversões por lote e **custo por conversão**.
- **Ajustes recomendados:** calibração de probabilidades (Platt/Isotônica), ajuste de **threshold** (mais recall vs. mais precision) e (se necessário) técnicas de **balanceamento** (SMOTE).

---

## 🧰 Bibliotecas utilizadas

- **Python** ≥ 3.9  
- `pandas`, `numpy`  
- `scikit-learn` (ColumnTransformer, Pipeline, modelos, métricas, GridSearchCV)  
- `matplotlib` (gráficos)  
- `joblib` (persistência do pipeline)

---

## 🧭 Como executar

### ▶️ Colab (recomendado)
1. Clique no badge **Open in Colab** (topo do README).
2. Execute as células de cima para baixo.

### 💻 Ambiente local
```bash
# 1) Ambiente virtual
python -m venv .venv
# Linux/macOS:
source .venv/bin/activate
# Windows (PowerShell):
.venv\Scripts\Activate.ps1

# 2) Dependências
pip install -r requirements.txt

# 3) Jupyter
jupyter notebook  # ou jupyter lab
