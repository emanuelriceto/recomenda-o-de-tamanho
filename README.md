# 🧥 Sistema de Recomendação de Tamanho para Vestuário Superior

> **TCC** — Desenvolvimento de um Sistema de Recomendação de Tamanho para Vestuário Superior em E-Commerce Visando a Redução de Incerteza na Escolha de Peças

---

## 📌 Sobre o Projeto

Este projeto desenvolve um sistema inteligente capaz de recomendar o tamanho correto de camisetas para usuários de e-commerce com base em suas medidas antropométricas (busto, ombro, altura, peso). O objetivo principal é reduzir a taxa de devolução causada por erro de tamanho.

O sistema é composto por:
- **Banco de dados antropométrico** com size charts de marcas brasileiras e internacionais
- **Modelo de Machine Learning** (XGBoost) treinado com dados corporais reais (ANSUR II)
- **API REST** (FastAPI) para integração com e-commerces
- **Módulo de visão computacional** (YOLOv8) para detecção de camisetas *(Fase 3)*

---

## 🗂️ Estrutura do Projeto

```
recomenda-o-de-tamanho/
├── database/
│   └── create_db.py
├── data/
│   ├── size_charts/
│   │   └── size_charts_data.py
│   └── ansur/
│       └── load_ansur.py
├── model/
│   ├── preparar_dados.py
│   └── treinar_modelo.py
├── docs/plots/
├── api/
├── notebooks/
├── requirements.txt
├── setup_ambiente.ps1
└── verificar_fase1.py
```

---

## 🚀 Como Executar

### 1. Clonar e configurar ambiente (Windows/PowerShell)
```powershell
git clone https://github.com/emanuelriceto/recomenda-o-de-tamanho.git
cd recomenda-o-de-tamanho

Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
.\setup_ambiente.ps1
```

### 2. Fase 1 — Banco de Dados
```powershell
python database\create_db.py
python data\size_charts\size_charts_data.py
python data\ansur\load_ansur.py
python verificar_fase1.py
```

### 3. Fase 2 — Modelo ML
```powershell
python model\preparar_dados.py
python model\treinar_modelo.py
```

### 4. Obter ANSUR II (recomendado)
1. Acesse: https://www.openicpsr.org/openicpsr/project/116564
2. Baixe `ANSUR_II_MALE_Public.csv` e `ANSUR_II_FEMALE_Public.csv`
3. Coloque em `data/ansur/`

---

## 📊 Resultados — Fase 2

| Modelo | Acurácia CV | F1-Score Teste |
|---|---|---|
| XGBoost ⭐ | 99.2% ± 0.8% | **99.4%** |
| Random Forest | 98.2% ± 1.3% | 98.9% |
| Regressão Logística | 89.9% ± 2.4% | 95.0% |

---

## 🗺️ Roadmap

- [x] Fase 1 — Banco de Dados Antropométrico
- [x] Fase 2 — Modelo de Machine Learning
- [ ] Fase 3 — Visão Computacional (YOLOv8 + DeepFashion)
- [ ] Fase 4 — API REST (FastAPI)
- [ ] Fase 5 — Aplicativo cliente

---

## 📚 Datasets

| Fonte | Uso |
|---|---|
| [ANSUR II](https://www.openicpsr.org/openicpsr/project/116564) | Medidas corporais reais |
| [DeepFashion — CUHK](http://mmlab.ie.cuhk.edu.hk/projects/DeepFashion.html) | Imagens para YOLO |
| Hering, Renner, Reserva, C&A, Zara | Size charts oficiais |

---

## 🛠️ Tecnologias

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0-orange)
![FastAPI](https://img.shields.io/badge/FastAPI-0.111-green)
![SQLite](https://img.shields.io/badge/SQLite-3-lightblue)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-purple)

---

**Autor:** Emanuel Riceto — [@emanuelriceto](https://github.com/emanuelriceto)

*Projeto acadêmico (TCC). Datasets possuem licenças próprias.*
