# 🧥 Sistema de Recomendação de Tamanho para Vestuário Superior

> **TCC — PUCPR · Escola Politécnica · Engenharia de Computação**
>
> Desenvolvimento de um Sistema de Recomendação de Tamanho para Vestuário Superior em E-Commerce Visando a Redução de Incerteza na Escolha de Peças

[![Python](https://img.shields.io/badge/Python-3.11%20%7C%203.13-blue)](https://python.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0-orange)](https://xgboost.readthedocs.io)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics%208.4-purple)](https://ultralytics.com)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111-green)](https://fastapi.tiangolo.com)
[![SQLite](https://img.shields.io/badge/SQLite-3-lightblue)](https://sqlite.org)
[![License](https://img.shields.io/badge/Dataset-CC%20BY%204.0-yellow)](https://creativecommons.org/licenses/by/4.0/)

---

## 📌 Sobre o Projeto

O e-commerce de moda enfrenta altas taxas de devolução causadas por erro de tamanho — estima-se que entre **25% e 40%** das compras online de roupas sejam devolvidas, sendo **31%** por problemas de ajuste e caimento (Shopify, 2024). Em 2022, apenas nos EUA, isso representou **US$ 212 bilhões** em devoluções (BBC News, 2023).

Este projeto desenvolve um sistema inteligente que resolve esse problema combinando dois modelos de IA:

1. **YOLOv8** — detecta a modelagem da camiseta na foto enviada pelo cliente (oversized, regular ou slim)
2. **XGBoost** — recomenda o tamanho correto cruzando as medidas corporais do cliente com a modelagem detectada e as size charts das marcas

### Escopo atual
- **Vestuário:** camisetas masculinas de manga curta
- **Modelagens:** oversized · regular · slim
- **Marcas:** Hering · Renner · Reserva · C&A · Zara
- **Público:** masculino adulto

---

## 🔄 Como o Sistema Funciona

```
Cliente informa:
  ├── Medidas corporais (busto, ombro, altura, peso)
  └── Foto da camiseta que pretende comprar
           │
           ▼
    ┌─────────────┐
    │   YOLOv8    │  → Detecta a modelagem
    │  (Fase 3)   │    (oversized / regular / slim)
    └──────┬──────┘
           │ modelagem detectada
           ▼
    ┌─────────────┐     ┌──────────────────────┐
    │   XGBoost   │ ←── │  Banco de Dados       │
    │  (Fase 2)   │     │  Size Charts + ANSUR  │
    └──────┬──────┘     └──────────────────────┘
           │
           ▼
    Tamanho recomendado
    + Confiança (%)
    + Alternativas
    + Medidas reais da peça
```

**Exemplo de resposta:**
```json
{
  "tamanho_recomendado": "M",
  "confianca": 0.94,
  "modelagem_detectada": "regular",
  "confianca_yolo": 0.95,
  "marca_referencia": "Hering",
  "alternativas": [
    {"tamanho": "M",  "probabilidade": 0.940},
    {"tamanho": "G",  "probabilidade": 0.058},
    {"tamanho": "P",  "probabilidade": 0.002}
  ]
}
```

---

## 🗂️ Estrutura do Projeto

```
recomenda-o-de-tamanho/
│
├── 📁 database/
│   └── create_db.py              # Cria o banco SQLite com 5 tabelas
│
├── 📁 data/
│   ├── size_charts/
│   │   └── size_charts_data.py   # Size charts: 5 marcas × 3 modelagens = 90 entradas
│   └── ansur/
│       ├── load_ansur.py         # Processa e carrega o ANSUR II no banco
│       └── ANSUR_II_MALE_Public.csv  # Dataset de medidas corporais (4.082 homens)
│
├── 📁 model/
│   ├── preparar_dados.py         # Engenharia de features + análise exploratória
│   └── treinar_modelo.py         # Treina XGBoost, Random Forest e Regressão Logística
│
├── 📁 yolo/
│   ├── treinar_yolo.py           # Fine-tuning do YOLOv8s com dataset próprio
│   ├── avaliar_yolo.py           # Avalia mAP, precisão e recall
│   └── weights/
│       └── best.pt               # Modelo YOLOv8 treinado (mAP50 = 0.953)
│
├── 📁 api/
│   ├── main.py                   # API REST — FastAPI com 5 endpoints
│   └── testar_api.py             # Script de testes automáticos
│
├── 📁 docs/plots/                # Gráficos gerados para a monografia
├── 📁 notebooks/                 # Análises exploratórias
│
├── requirements.txt              # Dependências Python (venv principal)
├── setup_ambiente.ps1            # Configuração automática do ambiente (Windows)
└── verificar_fase1.py            # Verifica integridade do banco de dados
```

---

## 🚀 Como Executar

### Pré-requisitos

- Python 3.11 e Python 3.13 instalados lado a lado
- Git
- GPU NVIDIA com CUDA (para treino do YOLO) — testado com RTX 4060 Ti 8GB

### 1. Clonar o repositório

```bash
git clone https://github.com/emanuelriceto/recomenda-o-de-tamanho.git
cd recomenda-o-de-tamanho
```

### 2. Configurar o ambiente principal (Python 3.13)

```powershell
# Windows — PowerShell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
.\setup_ambiente.ps1
```

Ou manualmente:

```bash
python -m venv venv
source venv/Scripts/activate          # Git Bash
pip install -r requirements.txt
```

### 3. Configurar o ambiente YOLO (Python 3.11)

```bash
py -3.11 -m venv venv_yolo
source venv_yolo/Scripts/activate
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 --no-cache-dir
pip install ultralytics pyyaml --prefer-binary --no-cache-dir
```

> **Por que dois ambientes?** O PyTorch com suporte CUDA não disponibiliza wheels para Python 3.13 ainda. O venv principal usa Python 3.13 para banco de dados, ML tabular e API. O venv_yolo usa Python 3.11 exclusivamente para o módulo de visão computacional.

---

## 📋 Execução por Fase

### Fase 1 — Banco de Dados Antropométrico

```bash
source venv/Scripts/activate

python database/create_db.py           # Cria as 5 tabelas no SQLite
python data/size_charts/size_charts_data.py  # Popula size charts (90 entradas)
python data/ansur/load_ansur.py        # Carrega ANSUR II (4.082 registros)
python verificar_fase1.py              # Verifica integridade e exibe estatísticas
```

**O que cada script faz:**

| Script | Resultado |
|---|---|
| `create_db.py` | `database/antropometrico.db` com 5 tabelas |
| `size_charts_data.py` | 90 entradas (5 marcas × 6 tam médios × 3 modelagens) |
| `load_ansur.py` | 4.082 registros corporais masculinos reais |
| `verificar_fase1.py` | Relatório de integridade no terminal |

**Para obter o ANSUR II:**
1. Acesse: https://www.openicpsr.org/openicpsr/project/116564
2. Baixe `ANSUR_II_MALE_Public.csv`
3. Coloque em `data/ansur/`

### Fase 2 — Modelo de Machine Learning (XGBoost)

```bash
source venv/Scripts/activate

python model/preparar_dados.py   # Prepara dataset e gera gráficos de análise
python model/treinar_modelo.py   # Treina, compara e exporta o melhor modelo
```

### Fase 3 — Visão Computacional (YOLOv8)

```bash
source venv_yolo/Scripts/activate

# Coloque o dataset em data/deepfashion/ antes de rodar
python yolo/treinar_yolo.py    # Fine-tuning do YOLOv8s
python yolo/avaliar_yolo.py    # Avaliação e testes visuais
```

**Estrutura esperada do dataset:**
```
data/deepfashion/
├── data.yaml
├── train/images/   # ~677 imagens (70%)
├── valid/images/   # ~81  imagens (20%)
└── test/images/    # ~40  imagens (10%)
```

### Fase 4 — API REST (FastAPI)

```bash
source venv/Scripts/activate

uvicorn api.main:app --reload --port 8000
```

Documentação interativa: **http://localhost:8000/docs**

---

## 📊 Resultados

### Fase 2 — Modelo XGBoost

Dataset de treinamento: ANSUR II masculino (4.082 pessoas × 3 modelagens = **12.246 amostras**).
Avaliação: cross-validation estratificado 5-fold + holdout de 20%.

| Modelo | Acurácia CV | F1-Score Teste |
|---|---|---|
| **XGBoost ⭐ (selecionado)** | **99,6% ± 0,3%** | **99,8%** |
| Random Forest | 99,7% ± 0,2% | 99,7% |
| Regressão Logística | 95,3% ± 0,5% | 95,5% |

**Features utilizadas (7):**

| Feature | Tipo | Descrição |
|---|---|---|
| `busto_circunf` | Medida direta | Circunferência do busto/tórax (cm) — mais importante |
| `largura_ombro` | Medida direta | Distância biacromial (cm) |
| `altura` | Medida direta | Altura total (cm) |
| `peso_kg` | Medida direta | Peso corporal (kg) |
| `imc` | Derivada | peso / (altura/100)² |
| `ratio_busto_ombro` | Derivada | busto ÷ ombro — proporcionalidade corporal |
| `modelagem_num` | Saída do YOLO | 0=oversized · 1=regular · 2=slim |

> **Nota:** `genero_num` foi removida — o sistema é exclusivamente masculino.

### Fase 3 — YOLOv8

**Dataset próprio** coletado nos sites das marcas cadastradas, anotado no Roboflow.
- 403 imagens originais → **967 imagens** após data augmentation
- Distribuição: 70% treino / 20% validação / 10% teste
- Classes: `oversized` · `regular` · `slim`

| Métrica | Resultado | Critério |
|---|---|---|
| **mAP50** | **0.953** | ≥ 0.50 ✅ |
| mAP50-95 | 0.946 | — |
| Precisão | 0.971 | — |
| Recall | 0.931 | — |

**Por classe:**

| Classe | mAP50 | Precisão | Recall |
|---|---|---|---|
| regular | 0.995 | 0.971 | 1.000 |
| oversized | 0.978 | 0.943 | 1.000 |
| slim | 0.886 | 1.000 | 0.792 |

> O recall menor na classe `slim` é esperado pelo menor número de amostras nessa classe. Coleta adicional de imagens slim está planejada.

---

## 🗺️ Roadmap

- [x] **Fase 1** — Banco de Dados Antropométrico
  - [x] Banco SQLite com 5 tabelas
  - [x] Size charts: 5 marcas × 3 modelagens (oversized/regular/slim)
  - [x] Integração com ANSUR II masculino (4.082 registros reais)
- [x] **Fase 2** — Modelo de Machine Learning
  - [x] Pipeline completo de treino e avaliação
  - [x] XGBoost selecionado — F1-Score: 99,8%
  - [x] Modelagem como feature (integração YOLO → XGBoost)
- [x] **Fase 3** — Visão Computacional
  - [x] Dataset próprio coletado e anotado no Roboflow (967 imagens)
  - [x] Fine-tuning YOLOv8s — mAP50: 0.953 ✅
  - [x] Modelo exportado: `yolo/weights/best.pt`
- [ ] **Fase 4** — API REST (FastAPI)
  - [x] Estrutura dos endpoints implementada
  - [ ] Integração YOLO + XGBoost no endpoint `/recomendar-por-foto`
- [ ] **Fase 5** — Interface do Cliente (trabalho futuro)

---

## 📚 Datasets e Fontes

| Fonte | Uso | Acesso |
|---|---|---|
| [ANSUR II — US Army](https://www.openicpsr.org/openicpsr/project/116564) | Medidas corporais masculinas reais (4.082 pessoas) | Público e gratuito |
| [Roboflow — camisetas-modelagem](https://universe.roboflow.com/emanuel-riceto-pucpr-edu-br/camisetas-modelagem) | Dataset próprio para YOLOv8 (967 imagens, CC BY 4.0) | Público |
| Hering, Renner, Reserva, C&A, Zara | Size charts coletadas dos sites oficiais | Público |

> **Limitação conhecida:** O ANSUR II é composto por militares americanos adultos. A generalização para a população brasileira será validada com coleta de dados locais com voluntários (etapa planejada).

---

## 🛠️ Tecnologias

| Tecnologia | Versão | Ambiente | Uso |
|---|---|---|---|
| Python | 3.13 | venv | Banco de dados, ML tabular, API |
| Python | 3.11 | venv_yolo | YOLOv8 (compatibilidade PyTorch+CUDA) |
| SQLite | 3 | — | Banco de dados antropométrico |
| XGBoost | 2.0 | venv | Modelo de recomendação de tamanho |
| scikit-learn | 1.4 | venv | Pipeline ML e avaliação |
| YOLOv8 (Ultralytics) | 8.4 | venv_yolo | Detecção de modelagem em imagens |
| PyTorch | 2.5.1+cu121 | venv_yolo | Backend CUDA para YOLOv8 |
| FastAPI | 0.111 | venv | API REST |
| Uvicorn | 0.29 | venv | Servidor ASGI |
| pandas / numpy | — | venv | Manipulação de dados |
| matplotlib / seaborn | — | venv | Visualizações |
| Roboflow | — | — | Anotação e export do dataset |
| Git / GitHub | — | — | Controle de versão |

**GPU testada:** NVIDIA GeForce RTX 4060 Ti (8 GB VRAM) · CUDA 12.1 · Driver 591.86

---

## 👥 Equipe

**Emanuel Riceto da Silva** — [@emanuelriceto](https://github.com/emanuelriceto)
**Frederico Virmond Fruet**

**Orientador:** Prof. Dr. Julio Cesar Nievola — PUCPR

---

## 📄 Licença

Projeto acadêmico (TCC). Os datasets utilizados possuem licenças próprias:
- ANSUR II: uso público para pesquisa acadêmica
- Dataset Roboflow (camisetas-modelagem): CC BY 4.0
- Size charts: coletadas dos sites oficiais das marcas para fins acadêmicos