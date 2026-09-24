"""
FASE 2B (v2) — Preparação dos dados para o XGBoost
TCC: Sistema de Recomendação de Tamanho para Vestuário Superior (masculino)

MUDANÇAS EM RELAÇÃO À v1
  1. Rótulo gerado pela REGRA DE FOLGA (model/regra_folga.py), que depende da
     modelagem. Na v1 o rótulo vinha só da faixa de busto corporal, idêntica
     nas 3 modelagens, e o modelo devolvia o mesmo tamanho para slim, regular
     e oversized (modelagem_num sem efeito).
  2. Coluna grupo_imc (faixas da OMS) para avaliar o modelo por perfil corporal.
  3. Classe SEM_TAMANHO: nenhum tamanho da marca de referência atende à folga
     mínima (ex.: obesidade grau III).

Execução:  python model/preparar_dados.py
"""
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))
from model.regra_folga import FOLGA, SEM_TAMANHO, escolher_tamanho  # noqa: E402

DB_PATH   = RAIZ / "database" / "antropometrico.db"
SAIDA_CSV = RAIZ / "model" / "dataset_treinamento.csv"
PLOTS_DIR = RAIZ / "docs" / "plots"
MARCA_REF = "Hering"
MODELAGEM_MAP = {"oversized": 0, "regular": 1, "slim": 2}  # ordem do data.yaml
GRUPOS = ["magreza_extrema", "magreza", "eutrofia", "sobrepeso",
          "obesidade_I_II", "obesidade_extrema"]


def grupo_imc(imc: float) -> str:
    """Faixas de IMC da OMS, agrupadas para análise."""
    if imc < 16:
        return "magreza_extrema"      # magreza grau III
    if imc < 18.5:
        return "magreza"              # graus I e II
    if imc < 25:
        return "eutrofia"
    if imc < 30:
        return "sobrepeso"
    if imc < 40:
        return "obesidade_I_II"
    return "obesidade_extrema"        # obesidade grau III


def carregar_dados():
    if not DB_PATH.exists():
        sys.exit(f"Banco não encontrado: {DB_PATH}. Execute a Fase 1 antes.")
    with sqlite3.connect(DB_PATH) as con:
        pessoas = pd.read_sql_query("""
            SELECT id, fonte_dataset, idade, altura, peso_kg, busto_circunf,
                   cintura_circunf, largura_ombro, comprimento_braco
            FROM medidas_corporais
            WHERE genero = 'M'
              AND busto_circunf IS NOT NULL AND altura IS NOT NULL
              AND peso_kg IS NOT NULL AND largura_ombro IS NOT NULL
        """, con)
        tabela = pd.read_sql_query("""
            SELECT t.tamanho_label, t.tamanho_ordem, t.modelagem,
                   t.largura_busto_min, t.largura_busto_max
            FROM tabela_tamanhos t JOIN marcas m ON m.id = t.marca_id
            WHERE m.nome = ? AND t.modelagem IN ('oversized', 'regular', 'slim')
            ORDER BY t.modelagem, t.tamanho_ordem
        """, con, params=(MARCA_REF,))
    if pessoas.empty:
        sys.exit("Nenhuma medida corporal masculina no banco. Execute load_ansur.py.")
    if tabela.empty:
        sys.exit(f"Sem tabela de tamanhos para {MARCA_REF}. Execute size_charts_data.py.")
    return pessoas, tabela


def criar_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["imc"] = (df["peso_kg"] / (df["altura"] / 100) ** 2).round(1)
    df["ratio_busto_ombro"] = (df["busto_circunf"] / df["largura_ombro"]).round(3)
    df["ratio_busto_cintura"] = np.where(
        df["cintura_circunf"] > 0,
        (df["busto_circunf"] / df["cintura_circunf"]).round(3), np.nan)
    df["grupo_imc"] = df["imc"].apply(grupo_imc)
    return df


def rotular(df: pd.DataFrame, tabela: pd.DataFrame) -> pd.DataFrame:
    """Uma linha por pessoa × modelagem, com o tamanho dado pela regra de folga."""
    ordem_sem_tamanho = int(tabela["tamanho_ordem"].max()) + 1
    blocos = []
    for modelagem, num in MODELAGEM_MAP.items():
        tam = tabela[tabela["modelagem"] == modelagem].to_dict("records")
        ordem = {t["tamanho_label"]: int(t["tamanho_ordem"]) for t in tam}
        ordem[SEM_TAMANHO] = ordem_sem_tamanho
        escolhas = [escolher_tamanho(b, c, tam, modelagem)
                    for b, c in zip(df["busto_circunf"], df["cintura_circunf"])]
        bloco = df.copy()
        bloco["modelagem"] = modelagem
        bloco["modelagem_num"] = num
        bloco["tamanho_Hering"] = [e.tamanho for e in escolhas]
        bloco["ordem_Hering"] = bloco["tamanho_Hering"].map(ordem)
        bloco["folga_cm"] = [e.folga_cm for e in escolhas]
        bloco["status_folga"] = [e.status for e in escolhas]
        blocos.append(bloco)
    return pd.concat(blocos, ignore_index=True)


def resumir(df: pd.DataFrame):
    print("\n📊 Tamanho × modelagem (amostras):")
    print(pd.crosstab(df["tamanho_Hering"], df["modelagem"]).to_string())

    piv = df.pivot_table(index="id", columns="modelagem", values="ordem_Hering")
    muda_slim = (piv["slim"] != piv["regular"]).mean()
    muda_over = (piv["oversized"] != piv["regular"]).mean()
    print(f"\n🔎 A modelagem altera o tamanho de {muda_slim:.1%} das pessoas (slim × regular)"
          f" e de {muda_over:.1%} (oversized × regular).")
    if muda_slim == 0 and muda_over == 0:
        print("   ⚠️  Modelagem sem efeito: revise FOLGA em model/regra_folga.py.")

    print("\n👥 Pessoas por grupo de IMC (OMS):")
    cont = df.drop_duplicates("id")["grupo_imc"].value_counts()
    for g in GRUPOS:
        print(f"   {g:<18} {int(cont.get(g, 0))}")


def gerar_grafico(df: pd.DataFrame):
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    tab = pd.crosstab(df["tamanho_Hering"], df["modelagem"])
    ordem = df.groupby("tamanho_Hering")["ordem_Hering"].min().sort_values().index
    tab = tab.reindex(ordem)
    ax = tab.plot(kind="bar", figsize=(9, 4.5), color=["#f0855a", "#2E75B6", "#c06ab3"])
    ax.set_xlabel("Tamanho (referência Hering)")
    ax.set_ylabel("Amostras")
    ax.set_title("Distribuição de tamanhos por modelagem (regra de folga)")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "01_tamanho_por_modelagem_v2.png", dpi=150)
    plt.close()


def preparar_dados() -> pd.DataFrame:
    print("=" * 62)
    print("  FASE 2B (v2) — Preparação dos dados · rótulo por folga")
    print("=" * 62)
    pessoas, tabela = carregar_dados()
    fontes = pessoas["fonte_dataset"].value_counts().to_dict()
    print(f"\n[1/4] {len(pessoas)} pessoas carregadas · fontes: {fontes}")

    df = criar_features(pessoas)
    print("[2/4] Features: IMC, ratio busto/ombro, ratio busto/cintura, grupo_imc")

    df = rotular(df, tabela)
    print(f"[3/4] {len(df)} amostras rotuladas ({df['id'].nunique()} pessoas × 3 modelagens)")
    print(f"      Parâmetros de folga (cm): {FOLGA}")
    resumir(df)
    gerar_grafico(df)

    df.to_csv(SAIDA_CSV, index=False)
    print(f"\n[4/4] Dataset salvo em {SAIDA_CSV.relative_to(RAIZ)}")
    return df


if __name__ == "__main__":
    preparar_dados()
