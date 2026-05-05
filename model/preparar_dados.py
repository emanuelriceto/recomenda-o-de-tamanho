"""
FASE 2B — Preparação dos Dados para o Modelo de ML
TCC: Sistema de Recomendação de Tamanho para Vestuário Superior

O que este script faz:
  1. Carrega os dados do banco SQLite (medidas corporais + size charts)
  2. Cria o dataset de treinamento unindo as duas fontes
  3. Gera análise exploratória com gráficos
  4. Salva o dataset preparado em CSV para uso no treino
"""

import sqlite3
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

# ── Caminhos ────────────────────────────────────────────────
BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
DB_PATH   = os.path.join(BASE_DIR, "../database/antropometrico.db")
OUT_DIR   = os.path.join(BASE_DIR, "../model")
PLOTS_DIR = os.path.join(BASE_DIR, "../docs/plots")
os.makedirs(OUT_DIR,   exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

# Paleta e estilo padrão dos gráficos
sns.set_theme(style="whitegrid", palette="Set2", font_scale=1.1)
CORES_TAMANHO = {
    "PP": "#4e9af1", "P": "#6cbe6c", "M": "#f5c242",
    "G":  "#f0855a", "GG": "#c06ab3", "XGG": "#888888"
}


# ═══════════════════════════════════════════════════════════
# 1. CARREGAR DADOS DO BANCO
# ═══════════════════════════════════════════════════════════

def carregar_dados() -> pd.DataFrame:
    """
    Carrega medidas corporais e cria o dataset de treinamento.
    
    Lógica:
      - Para cada pessoa nas medidas_corporais, calculamos a qual
        tamanho ela pertence em cada marca (usando as size charts).
      - Usamos a marca Hering como referência padrão (mais representativa
        do mercado brasileiro). Você pode mudar isso.
    """
    conn = sqlite3.connect(DB_PATH)

    # Carregar medidas corporais
    df = pd.read_sql_query("""
        SELECT
            id, fonte_dataset, genero, idade,
            altura, peso_kg, busto_circunf,
            cintura_circunf, largura_ombro,
            comprimento_braco, tamanho_inferido
        FROM medidas_corporais
        WHERE busto_circunf IS NOT NULL
          AND altura IS NOT NULL
    """, conn)

    # Carregar size charts (todas as marcas)
    size_charts = pd.read_sql_query("""
        SELECT m.nome as marca, t.tamanho_label, t.tamanho_ordem,
               t.busto_corpo_min, t.busto_corpo_max,
               t.largura_ombro as ombro_peca,
               t.comprimento_total,
               t.altura_min, t.altura_max
        FROM tabela_tamanhos t
        JOIN marcas m ON m.id = t.marca_id
        ORDER BY m.nome, t.tamanho_ordem
    """, conn)

    conn.close()
    return df, size_charts


def atribuir_tamanho_por_marca(df: pd.DataFrame, size_charts: pd.DataFrame,
                                marca: str = "Hering") -> pd.DataFrame:
    """
    Para cada pessoa no dataset, encontra o tamanho correto segundo
    a size chart de uma marca específica.
    
    Estratégia de desempate quando o busto cai em dois tamanhos:
      - Prioriza o maior tamanho (mais conservador = evita roupa apertada)
    """
    sc_marca = size_charts[size_charts["marca"] == marca].copy()
    sc_marca = sc_marca.sort_values("tamanho_ordem")

    resultados = []
    for _, pessoa in df.iterrows():
        busto  = pessoa["busto_circunf"]
        altura = pessoa["altura"]

        # Filtrar tamanhos compatíveis com busto
        matches = sc_marca[
            (sc_marca["busto_corpo_min"] <= busto) &
            (sc_marca["busto_corpo_max"] >= busto)
        ]

        # Se tiver filtro de altura, aplicar também
        if altura and len(matches) > 1:
            matches_alt = matches[
                (matches["altura_min"] <= altura) &
                (matches["altura_max"] >= altura)
            ]
            if len(matches_alt) > 0:
                matches = matches_alt

        if len(matches) == 0:
            # Fora do range: atribuir o mais próximo
            if busto < sc_marca["busto_corpo_min"].min():
                tamanho = sc_marca.iloc[0]["tamanho_label"]
                ordem   = sc_marca.iloc[0]["tamanho_ordem"]
            else:
                tamanho = sc_marca.iloc[-1]["tamanho_label"]
                ordem   = sc_marca.iloc[-1]["tamanho_ordem"]
        else:
            # Pegar o maior tamanho do intervalo (mais conservador)
            row     = matches.iloc[-1]
            tamanho = row["tamanho_label"]
            ordem   = row["tamanho_ordem"]

        resultados.append({
            "id":               pessoa["id"],
            f"tamanho_{marca}": tamanho,
            f"ordem_{marca}":   int(ordem),
        })

    return pd.DataFrame(resultados)


# ═══════════════════════════════════════════════════════════
# 2. ENGENHARIA DE FEATURES
# ═══════════════════════════════════════════════════════════

def criar_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cria features derivadas que ajudam o modelo a generalizar melhor.
    Todas as medidas originais estão em cm.
    """
    df = df.copy()

    # IMC (Índice de Massa Corporal)
    df["imc"] = df["peso_kg"] / ((df["altura"] / 100) ** 2)
    df["imc"] = df["imc"].round(1)

    # Relação busto/ombro (identifica proporcionalidade)
    df["ratio_busto_ombro"] = (df["busto_circunf"] / df["largura_ombro"]).round(2)

    # Relação busto/cintura (identifica tipo corporal)
    df["ratio_busto_cintura"] = np.where(
        df["cintura_circunf"].notna() & (df["cintura_circunf"] > 0),
        (df["busto_circunf"] / df["cintura_circunf"]).round(2),
        np.nan
    )

    # Gênero numérico (para o modelo)
    df["genero_num"] = df["genero"].map({"M": 1, "F": 0})

    return df


# ═══════════════════════════════════════════════════════════
# 3. ANÁLISE EXPLORATÓRIA (gráficos para o TCC)
# ═══════════════════════════════════════════════════════════

def gerar_graficos(df: pd.DataFrame):
    """Gera 4 gráficos úteis para a seção de metodologia do TCC."""

    print("   Gerando gráficos da análise exploratória...")

    # ── Gráfico 1: Distribuição de tamanhos por gênero ──────
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Distribuição de Tamanhos por Gênero\n(Dataset de Treinamento)",
                 fontsize=13, fontweight="bold")

    for ax, genero, titulo in zip(axes, ["M", "F"], ["Masculino", "Feminino"]):
        sub = df[df["genero"] == genero]["tamanho_Hering"].value_counts()
        ordem = ["PP", "P", "M", "G", "GG", "XGG"]
        sub = sub.reindex([t for t in ordem if t in sub.index])
        cores = [CORES_TAMANHO.get(t, "#aaa") for t in sub.index]
        ax.bar(sub.index, sub.values, color=cores, edgecolor="white", linewidth=0.8)
        ax.set_title(titulo, fontsize=12)
        ax.set_xlabel("Tamanho (Hering)")
        ax.set_ylabel("Quantidade de pessoas")
        for i, v in enumerate(sub.values):
            ax.text(i, v + 1, str(v), ha="center", fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "01_distribuicao_tamanhos.png"), dpi=150)
    plt.close()

    # ── Gráfico 2: Busto vs Altura com tamanho colorido ─────
    fig, ax = plt.subplots(figsize=(10, 6))
    for tam, grupo in df.groupby("tamanho_Hering"):
        cor = CORES_TAMANHO.get(tam, "#aaa")
        ax.scatter(grupo["busto_circunf"], grupo["altura"],
                   label=tam, color=cor, alpha=0.55, s=25, edgecolors="none")

    ax.set_xlabel("Circunferência do Busto (cm)")
    ax.set_ylabel("Altura (cm)")
    ax.set_title("Relação Busto × Altura por Tamanho (Hering)",
                 fontsize=13, fontweight="bold")
    ax.legend(title="Tamanho", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "02_busto_vs_altura.png"), dpi=150)
    plt.close()

    # ── Gráfico 3: Boxplot do busto por tamanho ─────────────
    fig, ax = plt.subplots(figsize=(10, 5))
    ordem = [t for t in ["PP","P","M","G","GG","XGG"]
             if t in df["tamanho_Hering"].values]
    cores = [CORES_TAMANHO.get(t, "#aaa") for t in ordem]

    bp = ax.boxplot(
        [df[df["tamanho_Hering"] == t]["busto_circunf"].dropna() for t in ordem],
        labels=ordem, patch_artist=True, notch=False,
        medianprops=dict(color="black", linewidth=2)
    )
    for patch, cor in zip(bp["boxes"], cores):
        patch.set_facecolor(cor)
        patch.set_alpha(0.75)

    ax.set_xlabel("Tamanho (Hering)")
    ax.set_ylabel("Circunferência do Busto (cm)")
    ax.set_title("Distribuição do Busto por Tamanho",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "03_boxplot_busto.png"), dpi=150)
    plt.close()

    # ── Gráfico 4: Mapa de calor de correlação ───────────────
    cols_corr = ["busto_circunf", "largura_ombro", "altura",
                 "peso_kg", "imc", "ratio_busto_ombro", "ordem_Hering"]
    corr = df[cols_corr].corr()

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm",
                center=0, ax=ax, linewidths=0.5,
                xticklabels=["Busto","Ombro","Altura","Peso","IMC","Busto/Ombro","Tamanho(num)"],
                yticklabels=["Busto","Ombro","Altura","Peso","IMC","Busto/Ombro","Tamanho(num)"])
    ax.set_title("Correlação entre Variáveis Antropométricas e Tamanho",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "04_correlacao.png"), dpi=150)
    plt.close()

    print(f"   ✓ 4 gráficos salvos em: docs/plots/")


# ═══════════════════════════════════════════════════════════
# 4. EXECUTAR PREPARAÇÃO COMPLETA
# ═══════════════════════════════════════════════════════════

def preparar_dados():
    print("\n" + "="*60)
    print("  FASE 2B — Preparação dos Dados para ML")
    print("="*60)

    # 1. Carregar
    print("\n[1/5] Carregando dados do banco...")
    df, size_charts = carregar_dados()
    print(f"      ✓ {len(df)} pessoas carregadas")
    print(f"      ✓ {len(size_charts)} entradas de size charts")

    # 2. Atribuir tamanho por marca de referência
    print("\n[2/5] Atribuindo tamanhos por marca...")
    for marca in size_charts["marca"].unique():
        tamanhos_df = atribuir_tamanho_por_marca(df, size_charts, marca)
        df = df.merge(tamanhos_df, on="id", how="left")
        print(f"      ✓ {marca}")

    # 3. Criar features derivadas
    print("\n[3/5] Criando features derivadas...")
    df = criar_features(df)
    print("      ✓ IMC, ratio busto/ombro, ratio busto/cintura, genero_num")

    # 4. Gráficos de análise exploratória
    print("\n[4/5] Gerando análise exploratória...")
    gerar_graficos(df)

    # 5. Salvar dataset final
    print("\n[5/5] Salvando dataset de treinamento...")
    dataset_path = os.path.join(OUT_DIR, "dataset_treinamento.csv")
    df.to_csv(dataset_path, index=False)
    print(f"      ✓ {dataset_path}")
    print(f"      ✓ {len(df)} linhas × {len(df.columns)} colunas")

    # Resumo das colunas disponíveis
    print("\n📋 Colunas do dataset final:")
    grupos = {
        "Identificação": ["id", "fonte_dataset", "genero", "idade"],
        "Medidas brutas": ["altura", "peso_kg", "busto_circunf",
                           "cintura_circunf", "largura_ombro", "comprimento_braco"],
        "Features criadas": ["imc", "ratio_busto_ombro", "ratio_busto_cintura", "genero_num"],
        "Labels (tamanhos)": [c for c in df.columns if c.startswith("tamanho_")],
    }
    for grupo, cols in grupos.items():
        existentes = [c for c in cols if c in df.columns]
        print(f"   {grupo}: {', '.join(existentes)}")

    # Estatísticas rápidas
    print("\n📊 Estatísticas das features principais:")
    stats_cols = ["busto_circunf", "largura_ombro", "altura", "imc"]
    stats = df[stats_cols].describe().round(1)
    stats.index = ["contagem", "média", "std", "mín", "Q1", "mediana", "Q3", "máx"]
    stats.columns = ["Busto (cm)", "Ombro (cm)", "Altura (cm)", "IMC"]
    print(stats.to_string())

    print("\n✅ Dados preparados! Pronto para a Fase 2C (treinamento do modelo).")
    print("="*60)

    return df


if __name__ == "__main__":
    preparar_dados()
