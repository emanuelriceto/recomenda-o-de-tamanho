# TCC — Sistema de Recomendação de Tamanho para Vestuário Superior
## FASE 1 — Banco de Dados Antropométrico

---

## O que foi construído nesta fase

Um banco de dados SQLite com 5 tabelas que formam o núcleo do sistema:

| Tabela | Conteúdo | Registros |
|---|---|---|
| `marcas` | Cadastro das marcas | 5 |
| `tabela_tamanhos` | Size charts (medidas de roupa × tamanho × marca) | 28 |
| `medidas_corporais` | Medidas corporais reais/sintéticas | 900 |
| `clientes` | Inputs do usuário final | (populado pela API) |
| `recomendacoes` | Histórico de recomendações | (populado pela API) |

---

## Como executar (ordem obrigatória)

```bash
# 1. Criar o banco e as tabelas
python database/create_db.py

# 2. Inserir as size charts das marcas
python data/size_charts/size_charts_data.py

# 3. Carregar medidas corporais
#    (coloque os CSVs do ANSUR II em data/ansur/ antes, se tiver)
python data/ansur/load_ansur.py

# 4. Verificar tudo
python verificar_fase1.py
```

---

## Como obter o ANSUR II real (recomendado para TCC)

1. Acesse: https://www.openicpsr.org/openicpsr/project/116564
2. Crie conta gratuita e aceite os termos de uso
3. Baixe os arquivos:
   - `ANSUR_II_MALE_Public.csv`
   - `ANSUR_II_FEMALE_Public.csv`
4. Coloque ambos em `data/ansur/`
5. Execute `python data/ansur/load_ansur.py` novamente

O ANSUR II contém **4.082 homens** e **1.986 mulheres** com ~100 medidas corporais cada.

---

## Como adicionar novas marcas

Edite `data/size_charts/size_charts_data.py` e adicione à lista `MARCAS`:

```python
{
    "nome": "SuaMarca",
    "pais_origem": "Brasil",
    "sistema_tamanho": "P/M/G",
    "tamanhos": [
        ("P", 1, larg_bust_min, larg_bust_max, larg_ombro, comprimento,
         busto_corpo_min, busto_corpo_max, altura_min, altura_max, "URL_fonte"),
        ...
    ]
}
```

---

## Estrutura de pastas

```
tcc_tamanho/
├── database/
│   ├── create_db.py          ← Cria o banco e as tabelas
│   └── antropometrico.db     ← Banco SQLite (gerado automaticamente)
├── data/
│   ├── size_charts/
│   │   └── size_charts_data.py  ← Size charts das marcas
│   └── ansur/
│       └── load_ansur.py        ← Carga do ANSUR II
├── verificar_fase1.py        ← Verificação de integridade
└── README.md
```

---

## Próximas fases

- **Fase 2** — Treinamento do modelo ML de recomendação (Random Forest / XGBoost)
- **Fase 3** — Visão computacional: YOLO + DeepFashion para detecção de camisetas
- **Fase 4** — API REST (FastAPI) expondo os dois módulos
- **Fase 5** — App de entrada do cliente
