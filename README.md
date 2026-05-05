# 🧥 Sistema de Recomendação de Tamanho para Vestuário Superior

> **TCC** — Desenvolvimento de um Sistema de Recomendação de Tamanho para Vestuário Superior em E-Commerce Visando a Redução de Incerteza na Escolha de Peças

[![Python](https://img.shields.io/badge/Python-3.10+-blue)](https://python.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0-orange)](https://xgboost.readthedocs.io)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111-green)](https://fastapi.tiangolo.com)
[![SQLite](https://img.shields.io/badge/SQLite-3-lightblue)](https://sqlite.org)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-purple)](https://ultralytics.com)

---

## 📌 Sobre o Projeto

O e-commerce de moda enfrenta altas taxas de devolução causadas por erro de tamanho — o cliente não consegue saber, antes da compra, se a peça vai servir. Este projeto desenvolve um sistema inteligente que recebe as medidas corporais do cliente e recomenda o tamanho correto de camiseta, por marca, com uma estimativa de confiança.

**Escopo atual:** vestuário superior masculino (camisetas). O sistema foi projetado para expansão futura ao público feminino após coleta de dados representativos.

O sistema é composto por quatro módulos:

| Módulo | Tecnologia | Status |
|---|---|---|
| Banco de Dados Antropométrico | SQLite | ✅ Concluído |
| Modelo de Machine Learning | XGBoost | ✅ Concluído |
| Visão Computacional | YOLOv8 + DeepFashion | 🔄 Em desenvolvimento |
| API REST | FastAPI | ⏳ Planejado |

---

## 🗂️ Estrutura do Projeto

```
recomenda-o-de-tamanho/
│
├── 📁 database/
│   └── create_db.py            # Cria o banco SQLite com todas as tabelas
│
├── 📁 data/
│   ├── size_charts/
│   │   └── size_charts_data.py # Size charts de 5 marcas (Hering, Renner, Reserva, C&A, Zara)
│   └── ansur/
│       ├── load_ansur.py       # Processa e carrega o ANSUR II no banco
│       └── ANSUR_II_MALE_Public.csv  # Dataset de medidas corporais (4.082 homens)
│
├── 📁 model/
│   ├── preparar_dados.py       # Engenharia de features + análise exploratória
│   └── treinar_modelo.py       # Treina, compara e exporta o melhor modelo
│
├── 📁 docs/
│   └── plots/                  # Gráficos gerados para a monografia (7 imagens)
│
├── 📁 api/                     # API REST — FastAPI (Fase 4, em breve)
├── 📁 notebooks/               # Análises exploratórias em Jupyter
├── 📁 scripts/                 # Scripts auxiliares
│
├── requirements.txt            # Dependências Python
├── setup_ambiente.ps1          # Configuração automática do ambiente (Windows)
└── verificar_fase1.py          # Verifica integridade do banco de dados
```

---

## 🚀 Como Executar

### Pré-requisitos
- Python 3.10 ou superior → [python.org](https://python.org)
- Git → [git-scm.com](https://git-scm.com/download/win)
- GPU NVIDIA com CUDA instalado *(necessário apenas para a Fase 3 — YOLO)*

### 1. Clonar o repositório

```powershell
git clone https://github.com/emanuelriceto/recomenda-o-de-tamanho.git
cd recomenda-o-de-tamanho
```

### 2. Configurar o ambiente virtual (Windows — PowerShell)

Executar apenas **uma vez**:

```powershell
# Liberar execução de scripts no PowerShell (necessário uma única vez)
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

# Criar ambiente virtual, ativá-lo e instalar todas as dependências
.\setup_ambiente.ps1
```

Para ativar o ambiente nas **próximas vezes** que abrir o terminal:

```powershell
.\venv\Scripts\Activate.ps1
```

---

### 3. Fase 1 — Banco de Dados Antropométrico

Cria o banco SQLite, popula as size charts das marcas e carrega as medidas corporais reais do ANSUR II.

```powershell
python database\create_db.py
python data\size_charts\size_charts_data.py
python data\ansur\load_ansur.py
python verificar_fase1.py
```

**O que cada script faz:**

| Script | O que faz | Resultado |
|---|---|---|
| `create_db.py` | Cria as 5 tabelas do banco | `database/antropometrico.db` |
| `size_charts_data.py` | Insere 28 tamanhos de 5 marcas | Banco populado com size charts |
| `load_ansur.py` | Processa o ANSUR II e insere no banco | 4.082 registros corporais reais |
| `verificar_fase1.py` | Confere integridade e mostra estatísticas | Relatório no terminal |

---

### 4. Fase 2 — Modelo de Machine Learning

Prepara o dataset de treinamento, gera gráficos de análise e treina os modelos.

```powershell
python model\preparar_dados.py
python model\treinar_modelo.py
```

**O que cada script faz:**

| Script | O que faz | Resultado |
|---|---|---|
| `preparar_dados.py` | Une banco + size charts, cria features derivadas, gera 4 gráficos | `model/dataset_treinamento.csv` + gráficos 01 a 04 |
| `treinar_modelo.py` | Treina 3 modelos, compara, salva o melhor, gera 3 gráficos | `model/modelo_recomendacao.joblib` + gráficos 05 a 07 |

---

## 📊 Resultados — Fase 2 (Modelo ML)

### Como as métricas foram calculadas

Cada modelo foi avaliado com duas estratégias combinadas:

**Acurácia CV (Cross-Validation 5-fold):**
O dataset foi dividido em 5 partes iguais. O modelo foi treinado 5 vezes, cada vez usando 4 partes para treino e 1 parte diferente para teste. A acurácia final é a **média das 5 rodadas** e o `± X%` é o **desvio padrão** — quanto menor, mais estável é o modelo entre diferentes subconjuntos de dados.

**F1-Score Teste:**
Após o cross-validation, o modelo foi treinado no conjunto completo de treino (80% dos dados) e avaliado em um conjunto de teste separado (20% dos dados, nunca visto durante o treino). O F1-Score combina **precisão** (dos que o modelo disse ser tamanho M, quantos realmente eram M?) e **recall** (de todos os que eram tamanho M, quantos o modelo identificou corretamente?). É mais justo que a acurácia simples quando há classes com quantidades diferentes de amostras.

### Comparativo dos modelos

| Modelo | Acurácia CV | F1-Score Teste | Observação |
|---|---|---|---|
| **XGBoost ⭐** | **99.6% ± 0.3%** | **99.8%** | Modelo selecionado |
| Random Forest | 99.7% ± 0.2% | 99.7% | Robusto, ligeiramente inferior |
| Regressão Logística | 95.3% ± 0.5% | 95.5% | Simples, usado como baseline |

> **Nota:** Os resultados foram obtidos com o dataset ANSUR II (militares masculinos americanos). A acurácia pode variar quando aplicada à população brasileira — coleta de dados locais está planejada como validação.

### Features utilizadas pelo modelo

| Feature | Tipo | Descrição |
|---|---|---|
| `busto_circunf` | Medida direta | Circunferência do busto/tórax em cm — **feature mais importante** |
| `largura_ombro` | Medida direta | Distância biacromial (ombro a ombro) em cm |
| `altura` | Medida direta | Altura total em cm |
| `peso_kg` | Medida direta | Peso corporal em kg |
| `imc` | Derivada | Peso / (Altura²) — índice de massa corporal |
| `ratio_busto_ombro` | Derivada | Busto ÷ Ombro — proporcionalidade corporal |
| `genero_num` | Codificada | 1 = Masculino, 0 = Feminino |
| `cintura_circunf` | Opcional | Circunferência da cintura em cm |
| `comprimento_braco` | Opcional | Comprimento do braço em cm |

---

## 🗺️ Roadmap

- [x] **Fase 1** — Banco de Dados Antropométrico
  - [x] Estrutura do banco SQLite (5 tabelas)
  - [x] Size charts de 5 marcas (Hering, Renner, Reserva, C&A, Zara)
  - [x] Integração com ANSUR II masculino (4.082 registros reais)
- [x] **Fase 2** — Modelo de Machine Learning
  - [x] Engenharia de features e análise exploratória
  - [x] Comparação de modelos (XGBoost, Random Forest, Regressão Logística)
  - [x] Exportação do modelo treinado
- [ ] **Fase 3** — Visão Computacional
  - [ ] Configuração do ambiente CUDA (GPU NVIDIA)
  - [ ] Fine-tuning do YOLOv8 com DeepFashion
  - [ ] Extração de keypoints de camisetas em imagens
- [ ] **Fase 4** — API REST (FastAPI)
  - [ ] Endpoint de recomendação por medidas
  - [ ] Endpoint de recomendação por foto
  - [ ] Documentação automática (Swagger)
- [ ] **Fase 5** — Interface do Cliente
  - [ ] Formulário web simples (HTML + JavaScript)
  - [ ] App mobile (implementação futura)

---

## 📚 Datasets e Fontes

| Fonte | Uso no Projeto | Acesso |
|---|---|---|
| [ANSUR II — US Army](https://www.openicpsr.org/openicpsr/project/116564) | Medidas corporais masculinas reais (4.082 pessoas) | Público e gratuito |
| [DeepFashion — CUHK](http://mmlab.ie.cuhk.edu.hk/projects/DeepFashion.html) | Imagens anotadas para treino do YOLOv8 | Acadêmico (requer acordo) |
| Hering, Renner, Reserva, C&A, Zara | Size charts coletadas dos sites oficiais | Público |
| [ABNT NBR 15800](https://www.abnt.org.br) | Norma brasileira de tabelas de tamanho | Pago |

> **Limitação conhecida:** Não existe dataset antropométrico brasileiro público equivalente ao ANSUR II. Os dados masculinos do ANSUR II (militares americanos) foram usados como proxy. Coleta de dados com a população brasileira está planejada como etapa de validação do TCC.

---

## 🛠️ Tecnologias

| Tecnologia | Versão | Uso |
|---|---|---|
| Python | 3.10+ | Linguagem principal |
| SQLite | 3 | Banco de dados |
| XGBoost | 2.0 | Modelo de recomendação |
| scikit-learn | 1.4 | Pipeline ML e avaliação |
| pandas / numpy | — | Manipulação de dados |
| matplotlib / seaborn | — | Visualizações |
| FastAPI | 0.111 | API REST *(Fase 4)* |
| YOLOv8 (Ultralytics) | — | Detecção de camisetas *(Fase 3)* |
| CUDA (NVIDIA) | — | Aceleração GPU para YOLO *(Fase 3)* |

---

## 👤 Autor

**Emanuel Riceto** — [@emanuelriceto](https://github.com/emanuelriceto)

---

*Projeto acadêmico (TCC). Os datasets utilizados possuem licenças próprias — consulte os links acima antes de qualquer uso comercial.*
