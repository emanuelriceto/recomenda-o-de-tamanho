# 🧥 Sistema de Recomendação de Tamanho para Vestuário Superior

> **TCC — PUCPR · Escola Politécnica · Engenharia de Computação**
>
> Desenvolvimento de um Sistema de Recomendação de Tamanho para Vestuário Superior em E-Commerce Visando a Redução de Incerteza na Escolha de Peças

[![Python](https://img.shields.io/badge/Python-3.11%20%7C%203.13-blue)](https://python.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0-orange)](https://xgboost.readthedocs.io)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics%208.4-purple)](https://ultralytics.com)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111-green)](https://fastapi.tiangolo.com)
[![SQLite](https://img.shields.io/badge/SQLite-3-lightblue)](https://sqlite.org)
[![Tests](https://img.shields.io/badge/pytest-12%20passed-brightgreen)](tests/test_api.py)

---

## 📌 Sobre o Projeto

O e-commerce de moda enfrenta altas taxas de devolução causadas por erro de tamanho — estima-se que entre **25% e 40%** das compras online de roupas sejam devolvidas, sendo **31%** por problemas de ajuste e caimento (Shopify, 2024). Em 2022, apenas nos EUA, isso representou **US$ 212 bilhões** em devoluções (BBC News, 2023).

O sistema combina dois modelos de IA e uma verificação pelas medidas reais da peça:

1. **YOLOv8** — detecta a modelagem da camiseta na foto enviada pelo cliente (oversized, regular ou slim)
2. **XGBoost** — prevê o tamanho a partir das medidas corporais do cliente e da modelagem detectada
3. **Regra de folga** — confere na tabela de medidas da marca se o tamanho previsto não fica justo nem folgado demais, e corrige quando necessário

### Escopo atual
- **Vestuário:** camisetas masculinas de manga curta
- **Modelagens:** oversized · regular · slim
- **Marcas:** Hering · Renner · Reserva · C&A · Zara
- **Público:** masculino adulto
- **Imagens:** foto do produto ou da camiseta vestida por um modelo

---

## 🔄 Como o Sistema Funciona

```
Cliente envia:
  ├── Medidas corporais (busto, ombro, altura, peso; cintura e braço opcionais)
  └── Foto da camiseta que pretende comprar
           │
           ▼
    ┌─────────────┐   confiança ≥ 50% → usa a modelagem detectada
    │   YOLOv8    │   nada detectado ou confiança < 50% → "regular" + aviso
    └──────┬──────┘
           │ modelagem
           ▼
    ┌─────────────┐
    │   XGBoost   │  → tamanho de referência (Hering) + probabilidades
    └──────┬──────┘
           │ convertido para a marca pedida (pela circunferência da peça)
           ▼
    ┌──────────────────────┐     ┌────────────────────────┐
    │ Verificação de folga │ ←── │  SQLite: size charts   │
    │  (peça − corpo)      │     │  5 marcas × 3 modelag. │
    └──────┬───────────────┘     └────────────────────────┘
           │  justo/folgado demais → corrige · nenhum tamanho serve → SEM_TAMANHO
           ▼
    Tamanho + nível de confiança + alternativas
    + medidas e folga da peça + avisos   (registrado em log_recomendacoes)
```

**Exemplo ilustrativo de resposta (`POST /recomendar-por-foto`):**
```json
{
  "tamanho_recomendado": "G",
  "marca": "Hering",
  "tamanho_referencia_modelo": "G",
  "confianca_modelo": 0.998,
  "nivel_confianca": "alta",
  "ajustado_pela_regra": false,
  "modelagem": {"usada": "slim", "origem": "yolo", "detectada": "slim", "confianca_yolo": 0.93},
  "alternativas": [{"tamanho": "G", "probabilidade": 0.998}, {"tamanho": "M", "probabilidade": 0.002}],
  "medidas_peca": {"tamanho": "G", "largura_busto_cm": 52.0, "circunferencia_peca_cm": 104.0,
                   "largura_ombro_cm": 45.5, "comprimento_cm": 72.0, "folga_cm": 7.0},
  "imc": 26.5,
  "avisos": [],
  "tempo_ms": 38.2
}
```

---

## 🗂️ Estrutura do Projeto

```
recomenda-o-de-tamanho/
│
├── 📁 database/
│   └── create_db.py                     # Cria o banco SQLite (5 tabelas)
│
├── 📁 data/
│   ├── size_charts/size_charts_data.py  # 5 marcas × 6 tamanhos × 3 modelagens = 90 entradas
│   ├── ansur/load_ansur.py              # Carrega o ANSUR II (4.082 homens)
│   ├── sinteticos/gerar_perfis_extremos.py  # Perfis sintéticos IMC ≥ 40 e IMC < 16
│   └── deepfashion/                     # Dataset do YOLO (não versionado — ver Fase 3)
│
├── 📁 model/
│   ├── regra_folga.py                   # Regra de folga: gera o rótulo e verifica a recomendação
│   ├── preparar_dados.py                # Features + rótulo por folga + grupos de IMC (OMS)
│   └── treinar_modelo.py                # XGBoost × Random Forest × Regressão Logística
│
├── 📁 yolo/
│   ├── treinar_yolo.py                  # Fine-tuning do YOLOv8s
│   ├── avaliar_yolo.py                  # mAP, precisão, recall + testes visuais
│   ├── testar_foto.py                   # Detecta a modelagem de uma foto avulsa
│   ├── mesclar_datasets.py              # Junta exports do Roboflow (remapeia classes pelo nome)
│   ├── classificar_modelagem.py         # Converte export de 1 classe em oversized/regular/slim
│   └── weights/
│       ├── best.pt                      # YOLOv8 v2 em uso (1.491 imagens)
│       └── best_v1.pt                   # YOLOv8 v1 (967 imagens), mantido para comparação
│
├── 📁 api/
│   ├── main.py                          # FastAPI — 5 endpoints
│   ├── config.py · schemas.py           # Configurações e validação (Pydantic)
│   └── servicos/
│       ├── yolo_servico.py              # Carrega o YOLO e detecta a modelagem
│       ├── ml_servico.py                # Carrega o XGBoost e prevê o tamanho
│       ├── recomendador.py              # Orquestra: XGBoost propõe → regra de folga verifica
│       └── banco_servico.py             # Marcas, size charts e log de recomendações
│
├── 📁 tests/
│   ├── test_api.py                      # 12 testes automatizados (pytest)
│   └── avaliar_confiabilidade.py        # Avaliação ponta a ponta: foto → YOLO → XGBoost
│
├── 📁 docs/plots/                       # Gráficos para a monografia
├── requirements.txt                     # venv (Python 3.13)
├── requirements_api.txt                 # venv_yolo (Python 3.11): API + modelos
├── setup_ambiente.ps1
└── verificar_fase1.py                   # Integridade do banco
```

---

## 🚀 Como Executar

### Pré-requisitos

- Python 3.11 (venv_yolo) e Python 3.13 (venv)
- Git
- GPU NVIDIA com CUDA para treinar o YOLO (testado com RTX 4060 Ti 8 GB)

### 1. Clonar

```bash
git clone https://github.com/emanuelriceto/recomenda-o-de-tamanho.git
cd recomenda-o-de-tamanho
```

### 2. Ambientes

```bash
# venv (Python 3.13) — banco de dados
python -m venv venv
source venv/Scripts/activate
pip install -r requirements.txt

# venv_yolo (Python 3.11) — YOLO, XGBoost e API
py -3.11 -m venv venv_yolo
source venv_yolo/Scripts/activate
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 --no-cache-dir
pip install ultralytics --prefer-binary --no-cache-dir
pip install -r requirements_api.txt
```

> **Por que dois ambientes?** O PyTorch com CUDA não tem versão para Python 3.13. Como a API carrega o YOLO e o XGBoost juntos, ela roda no **venv_yolo**. Treine o XGBoost nesse mesmo ambiente, para as versões de scikit-learn/xgboost baterem com os arquivos `.joblib`.

---

## 📋 Execução por Fase

### Fase 1 — Banco de Dados

```bash
python database/create_db.py
python data/size_charts/size_charts_data.py   # 90 entradas
python data/ansur/load_ansur.py               # 4.082 registros (requer o CSV abaixo)
python verificar_fase1.py
```

ANSUR II: baixe `ANSUR_II_MALE_Public.csv` em https://www.openicpsr.org/openicpsr/project/116564 e coloque em `data/ansur/`.

> A tabela `tabela_tamanhos` precisa ter a coluna `modelagem`. Bancos criados por versões antigas devem ser recriados com os comandos acima.

**Opcional — perfis extremos (IMC ≥ 40 e IMC < 16):**
```bash
python data/sinteticos/gerar_perfis_extremos.py --obesos 600 --magros 400
python data/sinteticos/gerar_perfis_extremos.py --remover     # desfaz
```

### Fase 2 — XGBoost (no venv_yolo)

```bash
source venv_yolo/Scripts/activate
python model/preparar_dados.py
python model/treinar_modelo.py
```

### Fase 3 — YOLOv8 (no venv_yolo)

```bash
python yolo/treinar_yolo.py
python yolo/avaliar_yolo.py
python yolo/testar_foto.py                    # teste com uma foto avulsa
```

**Adicionar imagens novas do Roboflow** (export no formato YOLOv8):
```bash
mv data/deepfashion data/deepfashion_v1
mkdir data/novas_roboflow && unzip NOVO_EXPORT.zip -d data/novas_roboflow
python yolo/mesclar_datasets.py --fontes data/deepfashion_v1 data/novas_roboflow --prefixos v1 novo --saida data/deepfashion
```
Se o export vier com uma classe genérica (ex.: `Camiseta`), rode antes o `yolo/classificar_modelagem.py` (`preparar` → classificar as fotos em pastas → `aplicar`).

### Fase 4 — API REST (no venv_yolo)

```bash
uvicorn api.main:app --reload --port 8000
```

Documentação interativa: **http://localhost:8000/docs**

| Método | Endpoint | Entrada | Módulos |
|---|---|---|---|
| GET | `/` | — | Status da API e dos modelos |
| GET | `/marcas` | — | Banco |
| GET | `/marcas/{marca}/tamanhos` | `modelagem` (opcional) | Banco |
| POST | `/recomendar-tamanho` | JSON: medidas + modelagem + marca | XGBoost + regra + banco |
| POST | `/recomendar-por-foto` | Formulário: foto (upload) + medidas + marca | YOLO + XGBoost + regra + banco |

```bash
curl -X POST http://localhost:8000/recomendar-por-foto \
  -F "foto=@camiseta.jpg" \
  -F busto_circunf=94 -F largura_ombro=43 -F altura=175 -F peso_kg=78 -F cintura_circunf=84
```

### Testes

```bash
python -m pytest tests -v                              # 12 testes da API
python tests/avaliar_confiabilidade.py --pessoas 40    # confiabilidade ponta a ponta
```

---

## 📊 Resultados

### XGBoost v2 (modelo em uso)

- Dataset: ANSUR II masculino, 4.082 pessoas × 3 modelagens = **12.246 amostras**
- Rótulo: **regra de folga** (folga = circunferência da peça − max(tórax, cintura))
- Avaliação: holdout de 20% e validação cruzada 5-fold, **ambos agrupados por pessoa** (a mesma pessoa nunca fica no treino e no teste)

| Métrica (teste) | Resultado |
|---|---|
| Acurácia (tamanho exato) | **99,3%** |
| F1-Score ponderado | **99,3%** |
| Acerto ±1 tamanho | **100%** |
| F1 na validação cruzada | 99,3% ± 0,1% |
| Pessoas cujo tamanho muda conforme a modelagem | **49,6%** |

**Regra de folga** (`model/regra_folga.py`):

| Modelagem | Folga mínima | Folga-alvo |
|---|---|---|
| slim | 2 cm | 5 cm |
| regular | 4 cm | 8 cm |
| oversized | 12 cm | 18 cm |

Classes de saída: `PP · P · M · G · GG · XGG · SEM_TAMANHO`.

**Features (10):** `busto_circunf`, `largura_ombro`, `altura`, `peso_kg`, `imc`, `ratio_busto_ombro`, `modelagem_num` (saída do YOLO: 0 = oversized, 1 = regular, 2 = slim), `cintura_circunf`, `comprimento_braco`, `ratio_busto_cintura`.

| Grupo de IMC (OMS) | Amostras no teste | Acerto exato | ±1 tamanho |
|---|---|---|---|
| Magreza grau III (< 16) | 3 | 100% | 100% |
| Magreza (16 a 18,5) | 9 | 100% | 100% |
| Eutrofia | 585 | 98,8% | 100% |
| Sobrepeso | 1.164 | 99,7% | 100% |
| Obesidade I e II | 672 | 99,1% | 100% |
| Obesidade grau III (≥ 40) | 18 | 94,4% | 100% |

> **Mudanças em relação à v1:** na v1, o rótulo vinha só da faixa de busto corporal, igual nas três modelagens, e o modelo recomendava o mesmo tamanho para slim, regular e oversized. Além disso, a mesma pessoa podia cair no treino e no teste, o que inflava o F1 de 99,8% reportado.

### YOLOv8 v2 (modelo em uso)

Dataset próprio (Roboflow), com fotos de produto e de modelos vestindo a camiseta:

| | v1 | **v2** |
|---|---|---|
| Fotos originais | 403 | **621** (403 + 218) |
| Imagens após augmentation | 967 | **1.491** |
| Split (train / valid / test) | 846 / 81 / 40 | **1.305 / 125 / 61** |
| Caixas slim no treino | 66 | **135** |

**Comparação no mesmo conjunto de teste (61 imagens):**

| Métrica | v1 | **v2** |
|---|---|---|
| mAP50 | 0,620 | **0,884** |
| mAP50-95 | 0,589 | **0,873** |
| Precisão | 0,494 | **0,868** |
| Recall | 0,800 | **0,832** |
| mAP50 oversized (23) | 0,902 | **0,984** |
| mAP50 regular (33) | 0,860 | **0,979** |
| mAP50 slim (5) | 0,099 | **0,688** |

**Validação da v2 (125 imagens):** mAP50 0,964 · mAP50-95 0,942 · Precisão 0,895 · Recall 0,967 · critério de aceite (mAP50 ≥ 0,50) ✅

> **Limitações:** o conjunto de teste tem só 5 imagens slim, então cada uma vale 20% da métrica da classe. A v1 usou pré-processamento *Stretch 640×640* e augmentation 3x (flip, rotação ±15°, brilho ±25%). As imagens novas usaram *Fit (white edges) 640×640* e augmentation 3x (flip, rotação ±10°, brilho e exposição ±20%, blur 1 px, ruído 2%).

### API

- 12/12 testes automatizados aprovados (`tests/test_api.py`)
- Inferência do YOLO: ~5 ms por imagem na GPU

---

## 🗺️ Roadmap

- [x] **Fase 1** — Banco de dados (SQLite, 90 entradas de size chart, ANSUR II)
- [x] **Fase 2** — XGBoost
  - [x] v1: pipeline de treino e avaliação
  - [x] v2: rótulo por regra de folga, split por pessoa, 10 features, classe `SEM_TAMANHO`
- [x] **Fase 3** — YOLOv8
  - [x] v1: 967 imagens, mAP50 0,953 na validação
  - [x] v2: 1.491 imagens, mAP50 0,884 no teste comum (v1: 0,620)
- [x] **Fase 4** — API REST
  - [x] Integração YOLO → XGBoost no `/recomendar-por-foto`
  - [x] Fallback por confiança, verificação de folga, conversão entre marcas e log
  - [x] 12 testes automatizados
- [ ] Incorporar perfis extremos (IMC ≥ 40 e < 16) ao treino — script pronto em `data/sinteticos/`
- [ ] Validação com voluntários (≥ 20 homens, medidas reais)
- [ ] Interface do cliente (trabalho futuro)

---

## 📚 Datasets e Fontes

| Fonte | Uso | Acesso |
|---|---|---|
| [ANSUR II — US Army](https://www.openicpsr.org/openicpsr/project/116564) | Medidas corporais masculinas (4.082 pessoas) | Público |
| [Roboflow — camisetas-modelagem](https://universe.roboflow.com/emanuel-riceto-pucpr-edu-br/camisetas-modelagem) | Dataset do YOLO v1 (967 imagens, CC BY 4.0) | Público |
| Roboflow — My First Project (v2) | 218 fotos novas (524 imagens com augmentation) | Privado |
| Hering, Renner, Reserva, C&A, Zara | Size charts dos sites oficiais | Público |

> **Limitações conhecidas:** o ANSUR II é composto por militares americanos adultos, e a generalização para a população brasileira depende da validação com voluntários. As recomendações valem apenas para as 5 marcas cadastradas. Os parâmetros da regra de folga são uma hipótese de projeto, a calibrar com essa validação.

---

## 🛠️ Tecnologias

| Tecnologia | Versão | Ambiente | Uso |
|---|---|---|---|
| Python | 3.13 | venv | Banco de dados |
| Python | 3.11 | venv_yolo | YOLO, XGBoost e API |
| SQLite | 3 | — | Banco antropométrico e log |
| XGBoost | 2.0+ | venv_yolo | Recomendação de tamanho |
| scikit-learn | 1.3+ | venv_yolo | Pipeline de ML e avaliação |
| YOLOv8 (Ultralytics) | 8.4 | venv_yolo | Detecção de modelagem |
| PyTorch | 2.5.1+cu121 | venv_yolo | Backend CUDA |
| FastAPI · Uvicorn · Pydantic | 0.111+ · 0.29+ · 2.7+ | venv_yolo | API REST |
| pytest | 9 | venv_yolo | Testes automatizados |
| Roboflow | — | — | Anotação e augmentation |

**GPU:** NVIDIA GeForce RTX 4060 Ti (8 GB) · CUDA 12.1

---

## 👥 Equipe

**Emanuel Riceto da Silva** — [@emanuelriceto](https://github.com/emanuelriceto)
**Frederico Virmond Fruet**

**Orientador:** Prof. Dr. Julio Cesar Nievola — PUCPR

---

## 📄 Licença

Projeto acadêmico (TCC). Os datasets têm licenças próprias:
- ANSUR II: uso público para pesquisa acadêmica
- Dataset Roboflow camisetas-modelagem: CC BY 4.0
- Size charts: coletadas dos sites oficiais das marcas para fins acadêmicos
