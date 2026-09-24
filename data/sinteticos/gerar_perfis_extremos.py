"""
Geração de perfis MASCULINOS sintéticos fora da distribuição do ANSUR II:
  • obesidade grau III  (IMC ≥ 40)   — classificação OMS
  • magreza grau III    (IMC < 16)   — classificação OMS

Método
  1. Ajusta regressões lineares no próprio banco (ANSUR II):
       medida = a + b·altura + c·peso   (tórax, cintura, ombro, braço)
  2. Sorteia altura ~ N(173; 7,5) cm e IMC ~ U(faixa); peso = IMC · altura²
  3. Prevê as medidas + ruído residual do ajuste; aplica limites plausíveis.

LIMITAÇÃO (declarar no TCC): é extrapolação estatística. Os perfis devem ser
validados com medições reais antes de conclusões fortes.

Uso (depois de load_ansur.py, que apaga a tabela medidas_corporais):
    python data/sinteticos/gerar_perfis_extremos.py --obesos 600 --magros 400
    python data/sinteticos/gerar_perfis_extremos.py --remover
"""
import argparse
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parent.parent.parent
DB_PATH = RAIZ / "database" / "antropometrico.db"
FONTE_OBESO = "SINTETICO_OBESIDADE_III"
FONTE_MAGRO = "SINTETICO_MAGREZA_III"
MEDIDAS = {  # coluna: limites plausíveis (cm)
    "busto_circunf": (70, 175),
    "cintura_circunf": (55, 185),
    "largura_ombro": (34, 54),       # estrutura óssea: varia pouco com o peso
    "comprimento_braco": (70, 100),
}


def inferir_tamanho(busto: float) -> str:  # mesma regra do load_ansur.py
    for t, (lo, hi) in {"PP": (0, 86), "P": (86, 92), "M": (92, 98), "G": (98, 104),
                        "GG": (104, 112), "XGG": (112, 999)}.items():
        if lo <= busto < hi:
            return t
    return "XGG"


R2_MINIMO = 0.40   # tórax e cintura precisam depender do peso para a extrapolação valer


def ajustar_regressoes(base: pd.DataFrame) -> dict:
    modelos = {}
    for col in MEDIDAS:
        d = base.dropna(subset=["altura", "peso_kg", col])
        X = np.column_stack([np.ones(len(d)), d["altura"], d["peso_kg"]])
        y = d[col].values
        coef, *_ = np.linalg.lstsq(X, y, rcond=None)
        resid = y - X @ coef
        r2 = 1 - resid.var() / y.var()
        modelos[col] = (coef, float(resid.std()))
        print(f"   {col:<18} = {coef[0]:7.2f} + {coef[1]:.3f}·altura + {coef[2]:.3f}·peso"
              f"   (R² = {r2:.2f} · σ resíduo = {resid.std():.1f} cm)")
        if col in ("busto_circunf", "cintura_circunf") and r2 < R2_MINIMO:
            sys.exit(f"\n❌ R² de {col} = {r2:.2f} (< {R2_MINIMO}): a base não relaciona medidas e peso.\n"
                     "   Isso ocorre com o fallback sintético IBGE (valores sorteados de forma\n"
                     "   independente). Carregue o ANSUR_II_MALE_Public.csv real e rode de novo.")
    return modelos


def gerar(n, imc_min, imc_max, fonte, modelos, rng) -> list[tuple]:
    altura = np.clip(rng.normal(173, 7.5, n), 155, 200)
    imc = rng.uniform(imc_min, imc_max, n)
    peso = imc * (altura / 100) ** 2
    X = np.column_stack([np.ones(n), altura, peso])
    med = {}
    for col, (lo, hi) in MEDIDAS.items():
        coef, sd = modelos[col]
        med[col] = np.clip(X @ coef + rng.normal(0, sd, n), lo, hi)
    return [(fonte, f"{fonte}_{i:04d}", round(altura[i], 1), round(peso[i], 1),
             round(med["busto_circunf"][i], 1), round(med["cintura_circunf"][i], 1),
             round(med["largura_ombro"][i], 1), None, round(med["comprimento_braco"][i], 1),
             "M", int(rng.integers(18, 66)), inferir_tamanho(med["busto_circunf"][i]))
            for i in range(n)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--obesos", type=int, default=600)
    ap.add_argument("--magros", type=int, default=400)
    ap.add_argument("--remover", action="store_true", help="remove os perfis sintéticos extremos")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if not DB_PATH.exists():
        sys.exit(f"Banco não encontrado: {DB_PATH}")
    con = sqlite3.connect(DB_PATH)
    con.execute("DELETE FROM medidas_corporais WHERE fonte_dataset IN (?, ?)", (FONTE_OBESO, FONTE_MAGRO))
    con.commit()
    if args.remover:
        print("✅ Perfis sintéticos extremos removidos.")
        return

    base = pd.read_sql_query("SELECT * FROM medidas_corporais WHERE genero = 'M'", con)
    if len(base) < 100:
        sys.exit("Poucos registros de base. Execute data/ansur/load_ansur.py antes.")
    print(f"Base: {len(base)} registros ({base['fonte_dataset'].unique().tolist()})")
    print("\nRegressões ajustadas:")
    modelos = ajustar_regressoes(base)

    rng = np.random.default_rng(args.seed)
    novos = (gerar(args.obesos, 40.0, 55.0, FONTE_OBESO, modelos, rng)
             + gerar(args.magros, 14.0, 15.99, FONTE_MAGRO, modelos, rng))
    con.executemany("""
        INSERT INTO medidas_corporais (fonte_dataset, id_original, altura, peso_kg,
            busto_circunf, cintura_circunf, largura_ombro, comprimento_torso,
            comprimento_braco, genero, idade, tamanho_inferido)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""", novos)
    con.commit()

    resumo = pd.read_sql_query("""
        SELECT fonte_dataset AS fonte, COUNT(*) AS n,
               ROUND(AVG(peso_kg / ((altura/100.0)*(altura/100.0))), 1) AS imc_medio,
               ROUND(AVG(busto_circunf), 1) AS busto_medio,
               ROUND(AVG(cintura_circunf), 1) AS cintura_media,
               ROUND(AVG(largura_ombro), 1) AS ombro_medio
        FROM medidas_corporais GROUP BY fonte_dataset""", con)
    con.close()
    print("\n" + resumo.to_string(index=False))
    print("\n✅ Perfis extremos inseridos. Próximo: python model/preparar_dados.py")


if __name__ == "__main__":
    main()
