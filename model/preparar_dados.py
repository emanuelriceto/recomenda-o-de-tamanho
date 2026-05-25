"""
FASE 2B — Preparação dos Dados para o Modelo de ML
TCC: Sistema de Recomendação de Tamanho para Vestuário Superior

ATUALIZAÇÃO: feature 'modelagem_num' adicionada ao dataset de treino
A modelagem é detectada pelo YOLO e passada como feature ao XGBoost.

Mapeamento numérico das modelagens:
  regular   → 0
  slim      → 1
  oversized → 2
  longline  → 3
  henley    → 4
"""

import sqlite3
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
DB_PATH   = os.path.join(BASE_DIR, "../database/antropometrico.db")
OUT_DIR   = os.path.join(BASE_DIR, "../model")
PLOTS_DIR = os.path.join(BASE_DIR, "../docs/plots")
os.makedirs(OUT_DIR,   exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

sns.set_theme(style="whitegrid", palette="Set2", font_scale=1.1)

# ── Mapeamento de modelagem → número ─────────────────────────
MODELAGEM_MAP = {
    "regular":   0,
    "slim":      1,
    "oversized": 2,
    "longline":  3,
    "henley":    4,
}

CORES_TAMANHO = {
    "PP": "#4e9af1", "P": "#6cbe6c", "M": "#f5c242",
    "G":  "#f0855a", "GG": "#c06ab3", "XGG": "#888888",
    "XS": "#4e9af1", "S": "#6cbe6c", "L": "#f0855a",
    "XL": "#c06ab3", "XXL": "#888888", "EGG": "#555555",
    "XG": "#999999",
}


def carregar_dados():
    conn = sqlite3.connect(DB_PATH)

    df = pd.read_sql_query("""
        SELECT id, fonte_dataset, genero, idade,
               altura, peso_kg, busto_circunf,
               cintura_circunf, largura_ombro,
               comprimento_braco, tamanho_inferido
        FROM medidas_corporais
        WHERE busto_circunf IS NOT NULL
          AND altura IS NOT NULL
    """, conn)

    size_charts = pd.read_sql_query("""
        SELECT m.nome as marca, t.tamanho_label,
               t.tamanho_ordem, t.modelagem,
               t.busto_corpo_min, t.busto_corpo_max,
               t.largura_ombro as ombro_peca,
               t.comprimento_total,
               t.altura_min, t.altura_max
        FROM tabela_tamanhos t
        JOIN marcas m ON m.id = t.marca_id
        ORDER BY m.nome, t.modelagem, t.tamanho_ordem
    """, conn)

    conn.close()
    return df, size_charts


def atribuir_tamanho_por_marca_modelagem(df, size_charts,
                                          marca="Hering",
                                          modelagem="regular"):
    """
    Atribui tamanho para cada pessoa considerando marca E modelagem.
    """
    sc = size_charts[
        (size_charts["marca"] == marca) &
        (size_charts["modelagem"] == modelagem)
    ].copy().sort_values("tamanho_ordem")

    resultados = []
    for _, pessoa in df.iterrows():
        busto  = pessoa["busto_circunf"]
        altura = pessoa["altura"]

        matches = sc[
            (sc["busto_corpo_min"] <= busto) &
            (sc["busto_corpo_max"] >= busto)
        ]

        if altura and len(matches) > 1:
            matches_alt = matches[
                (matches["altura_min"] <= altura) &
                (matches["altura_max"] >= altura)
            ]
            if len(matches_alt) > 0:
                matches = matches_alt

        if len(matches) == 0:
            row = sc.iloc[0] if busto < sc["busto_corpo_min"].min() else sc.iloc[-1]
        else:
            row = matches.iloc[-1]

        resultados.append({
            "id": pessoa["id"],
            f"tamanho_{marca}_{modelagem}": row["tamanho_label"],
            f"ordem_{marca}_{modelagem}":   int(row["tamanho_ordem"]),
        })

    return pd.DataFrame(resultados)


def criar_features(df):
    df = df.copy()
    df["imc"] = (df["peso_kg"] / ((df["altura"] / 100) ** 2)).round(1)
    df["ratio_busto_ombro"] = (df["busto_circunf"] / df["largura_ombro"]).round(2)
    df["ratio_busto_cintura"] = np.where(
        df["cintura_circunf"].notna() & (df["cintura_circunf"] > 0),
        (df["busto_circunf"] / df["cintura_circunf"]).round(2),
        np.nan
    )
    df["genero_num"] = df["genero"].map({"M": 1, "F": 0})
    return df


def expandir_por_modelagem(df, size_charts):
    """
    NOVO: Para cada pessoa, cria uma linha por modelagem.
    Isso permite treinar o modelo com a modelagem como feature.
    """
    linhas = []

    for modelagem, mod_num in MODELAGEM_MAP.items():
        df_mod = df.copy()
        df_mod["modelagem"]     = modelagem
        df_mod["modelagem_num"] = mod_num

        # Atribuir tamanho para esta modelagem (usando Hering como referência)
        tamanhos = atribuir_tamanho_por_marca_modelagem(
            df, size_charts, marca="Hering", modelagem=modelagem
        )
        col_tam   = f"tamanho_Hering_{modelagem}"
        col_ordem = f"ordem_Hering_{modelagem}"

        df_mod = df_mod.merge(tamanhos, on="id", how="left")
        df_mod = df_mod.rename(columns={
            col_tam:   "tamanho_Hering",
            col_ordem: "ordem_Hering",
        })
        linhas.append(df_mod)

    return pd.concat(linhas, ignore_index=True)


def gerar_graficos(df):
    print("   Gerando gráficos...")

    # Gráfico 1: Distribuição de tamanhos por modelagem
    fig, axes = plt.subplots(1, 5, figsize=(20, 4))
    fig.suptitle("Distribuição de Tamanhos por Modelagem (Hering)",
                 fontsize=13, fontweight="bold")

    for ax, modelagem in zip(axes, MODELAGEM_MAP.keys()):
        sub = df[df["modelagem"] == modelagem]["tamanho_Hering"].value_counts()
        ordem = ["PP","P","XS","S","M","G","L","GG","XL","XGG","XXL","EGG","XG"]
        sub = sub.reindex([t for t in ordem if t in sub.index])
        cores = [CORES_TAMANHO.get(t, "#aaa") for t in sub.index]
        ax.bar(sub.index, sub.values, color=cores, edgecolor="white")
        ax.set_title(modelagem.capitalize())
        ax.set_xlabel("Tamanho")
        ax.set_ylabel("N")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "01_distribuicao_por_modelagem.png"), dpi=150)
    plt.close()

    # Gráfico 2: Busto vs Tamanho por modelagem
    fig, ax = plt.subplots(figsize=(10, 6))
    for modelagem in MODELAGEM_MAP.keys():
        sub = df[df["modelagem"] == modelagem]
        ax.scatter(sub["busto_circunf"], sub["ordem_Hering"],
                   label=modelagem, alpha=0.3, s=15)
    ax.set_xlabel("Circunferência do Busto (cm)")
    ax.set_ylabel("Ordem do Tamanho (1=menor)")
    ax.set_title("Busto × Tamanho por Modelagem", fontweight="bold")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "02_busto_vs_tamanho_modelagem.png"), dpi=150)
    plt.close()

    # Gráfico 3: Correlação
    cols_corr = ["busto_circunf", "largura_ombro", "altura",
                 "peso_kg", "imc", "modelagem_num", "ordem_Hering"]
    corr = df[cols_corr].corr()
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm",
                center=0, ax=ax, linewidths=0.5,
                xticklabels=["Busto","Ombro","Altura","Peso",
                              "IMC","Modelagem","Tamanho(num)"],
                yticklabels=["Busto","Ombro","Altura","Peso",
                              "IMC","Modelagem","Tamanho(num)"])
    ax.set_title("Correlação — Variáveis + Modelagem", fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "03_correlacao_com_modelagem.png"), dpi=150)
    plt.close()

    print("   ✓ 3 gráficos salvos em docs/plots/")


def preparar_dados():
    print("\n" + "="*60)
    print("  FASE 2B — Preparação dos Dados (com Modelagem)")
    print("="*60)

    print("\n[1/5] Carregando dados do banco...")
    df, size_charts = carregar_dados()
    print(f"      ✓ {len(df)} pessoas | "
          f"{len(size_charts)} entradas de size charts")

    print("\n[2/5] Criando features derivadas...")
    df = criar_features(df)

    print("\n[3/5] Expandindo dataset por modelagem...")
    df_expandido = expandir_por_modelagem(df, size_charts)
    print(f"      ✓ {len(df_expandido)} linhas "
          f"({len(df)} pessoas × {len(MODELAGEM_MAP)} modelagens)")

    print("\n[4/5] Gerando gráficos de análise exploratória...")
    gerar_graficos(df_expandido)

    print("\n[5/5] Salvando dataset de treinamento...")
    path = os.path.join(OUT_DIR, "dataset_treinamento.csv")
    df_expandido.to_csv(path, index=False)
    print(f"      ✓ {path}")
    print(f"      ✓ {len(df_expandido)} linhas × {len(df_expandido.columns)} colunas")

    print("\n📋 Features disponíveis:")
    print("   Medidas:   busto_circunf, largura_ombro, altura, peso_kg")
    print("   Derivadas: imc, ratio_busto_ombro, genero_num")
    print("   NOVO:      modelagem_num (0=regular…4=henley)")
    print("   Label:     tamanho_Hering")

    print("\n✅ Dados prontos para o treino!")
    print("="*60)

    return df_expandido


if __name__ == "__main__":
    preparar_dados()
