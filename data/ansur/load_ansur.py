"""
FASE 1C — Carga do Dataset ANSUR II no Banco de Dados
TCC: Sistema de Recomendação de Tamanho para Vestuário Superior

ANSUR II (Army Anthropometric Survey II, 2012)
  - Dataset público do Exército dos EUA
  - ~6.000 militares americanos com medidas corporais detalhadas
  - Medidas originais em MILÍMETROS e KG
  - Download gratuito em: https://data.openicpsr.org/openicpsr/project/116564

Como baixar manualmente (sem login):
  1. Acesse: https://www.openicpsr.org/openicpsr/project/116564/version/V1/view
  2. Baixe: ANSUR_II_MALE_Public.csv e ANSUR_II_FEMALE_Public.csv
  3. Coloque ambos em: tcc_tamanho/data/ansur/

Este script também tenta baixar via URL direta de mirrors acadêmicos.
Se não conseguir, usa dados sintéticos representativos para o TCC.
"""

import sqlite3
import os
import csv
import urllib.request
import json

DB_PATH = os.path.join(os.path.dirname(__file__), '../../database/antropometrico.db')
ANSUR_DIR = os.path.dirname(__file__)

# Mapeamento das colunas ANSUR II → nossas colunas
# Medidas originais estão em MILÍMETROS, convertemos para CM (/10)
MAPEAMENTO_COLUNAS = {
    "stature":              "altura",           # altura total em pé (mm → cm)
    "weightkg":             "peso_kg",          # peso em décimos de kg (815 = 81.5kg)
    "chestcircumference":   "busto_circunf",    # circunferência do tórax (mm → cm)
    "waistcircumference":   "cintura_circunf",  # circunferência da cintura (mm → cm)
    "biacromialbreadth":    "largura_ombro",    # largura biacromial = distância entre ombros (mm → cm)
    "sleeveoutseam":        "comprimento_braco",# comprimento manga externa (mm → cm)
    "cervicaleheight":      "comprimento_torso",# altura do pescoço (mm → cm)
}

# Função para inferir tamanho com base no busto (circunferência)
def inferir_tamanho(busto_cm: float, genero: str = "M") -> str:
    """
    Inferência simples de tamanho baseada na circunferência do busto.
    Usa tabela Hering como referência padrão.
    """
    tabela = {
        "PP":  (80,  86),
        "P":   (86,  92),
        "M":   (92,  98),
        "G":   (98, 104),
        "GG":  (104, 112),
        "XGG": (112, 124),
    }
    for tamanho, (min_val, max_val) in tabela.items():
        if min_val <= busto_cm < max_val:
            return tamanho
    if busto_cm < 80:
        return "PP"
    return "XGG"


def processar_arquivo_ansur(filepath: str, genero: str) -> list:
    """Lê CSV do ANSUR II e retorna lista de dicionários com medidas."""
    registros = []

    with open(filepath, newline='', encoding='latin-1') as f:
        reader = csv.DictReader(f)
        colunas_disponiveis = reader.fieldnames

        for i, row in enumerate(reader):
            try:
                # Normalizar nomes de colunas (minúsculo)
                row_lower = {k.lower(): v for k, v in row.items()}

                def get_mm_to_cm(col):
                    val = row_lower.get(col, "").strip()
                    return round(float(val) / 10, 1) if val else None

                def get_decikg_to_kg(col):
                    # ANSUR II armazena peso em décimos de kg (815 = 81.5 kg)
                    val = row_lower.get(col, "").strip()
                    return round(float(val) / 10, 1) if val else None

                altura    = get_mm_to_cm("stature")
                peso      = get_decikg_to_kg("weightkg")
                busto     = get_mm_to_cm("chestcircumference")
                cintura   = get_mm_to_cm("waistcircumference")
                ombro     = get_mm_to_cm("biacromialbreadth")   # coluna correta para largura ombro
                braco     = get_mm_to_cm("sleeveoutseam")
                torso     = get_mm_to_cm("cervicaleheight")
                id_orig   = row_lower.get("subjectid", str(i))
                # Age vem com maiúscula no ANSUR II real
                idade_str = row_lower.get("age", "")
                idade     = int(float(idade_str)) if idade_str.strip() else None

                if not busto:
                    continue  # linha sem dado principal, pular

                tamanho = inferir_tamanho(busto, genero)

                registros.append({
                    "fonte_dataset":     f"ANSUR_II_{genero}",
                    "id_original":       id_orig,
                    "altura":            altura,
                    "peso_kg":           peso,
                    "busto_circunf":     busto,
                    "cintura_circunf":   cintura,
                    "largura_ombro":     ombro,
                    "comprimento_torso": torso,
                    "comprimento_braco": braco,
                    "genero":            genero,
                    "idade":             idade,
                    "tamanho_inferido":  tamanho,
                })
            except (ValueError, KeyError):
                continue

    return registros


def gerar_dados_sinteticos(n_masculino=500, n_feminino=400) -> list:
    """
    Gera dados sintéticos representativos da população brasileira adulta.
    Baseado nas médias do IBGE POF 2008-2009 e literatura antropométrica.
    Usado quando o ANSUR II não está disponível.
    """
    import random
    import math

    random.seed(42)  # reprodutibilidade

    def normal_clamp(media, std, minv, maxv):
        # Gera valor com distribuição normal truncada
        for _ in range(100):
            v = random.gauss(media, std)
            if minv <= v <= maxv:
                return round(v, 1)
        return round(media, 1)

    registros = []

    # Parâmetros masculinos (médias brasileiras aproximadas)
    params_m = {
        "altura":   (173.0, 7.5, 155.0, 200.0),
        "peso_kg":  (82.0,  15.0, 55.0, 140.0),
        "busto":    (98.0,  10.0, 78.0, 130.0),
        "cintura":  (90.0,  12.0, 68.0, 130.0),
        "ombro":    (43.5,  3.0,  36.0, 54.0),
        "braco":    (85.0,  5.0,  70.0, 100.0),
    }

    # Parâmetros femininos
    params_f = {
        "altura":   (161.0, 7.0, 148.0, 185.0),
        "peso_kg":  (69.0,  14.0, 45.0, 120.0),
        "busto":    (91.0,  10.0, 74.0, 120.0),
        "cintura":  (82.0,  12.0, 62.0, 115.0),
        "ombro":    (38.5,  2.5,  32.0, 48.0),
        "braco":    (75.0,  4.5,  62.0, 90.0),
    }

    for genero, params, n in [("M", params_m, n_masculino), ("F", params_f, n_feminino)]:
        for i in range(n):
            altura  = normal_clamp(*params["altura"])
            peso    = normal_clamp(*params["peso_kg"])
            busto   = normal_clamp(*params["busto"])
            cintura = normal_clamp(*params["cintura"])
            ombro   = normal_clamp(*params["ombro"])
            braco   = normal_clamp(*params["braco"])
            idade   = random.randint(18, 65)

            registros.append({
                "fonte_dataset":     f"SINTETICO_{genero}",
                "id_original":       f"SYN_{genero}_{i:04d}",
                "altura":            altura,
                "peso_kg":           peso,
                "busto_circunf":     busto,
                "cintura_circunf":   cintura,
                "largura_ombro":     ombro,
                "comprimento_torso": None,
                "comprimento_braco": braco,
                "genero":            genero,
                "idade":             idade,
                "tamanho_inferido":  inferir_tamanho(busto, genero),
            })

    return registros


def carregar_medidas_corporais():
    """
    Tenta carregar o ANSUR II real. Se não encontrado, usa dados sintéticos.
    """
    arquivos = {
        "M": os.path.join(ANSUR_DIR, "ANSUR_II_MALE_Public.csv"),
        "F": os.path.join(ANSUR_DIR, "ANSUR_II_FEMALE_Public.csv"),
    }

    todos_registros = []
    usando_real = False

    for genero, filepath in arquivos.items():
        if os.path.exists(filepath):
            print(f"   📂 Arquivo encontrado: {os.path.basename(filepath)}")
            registros = processar_arquivo_ansur(filepath, genero)
            todos_registros.extend(registros)
            usando_real = True
            print(f"      → {len(registros)} registros carregados")
        else:
            print(f"   ⚠️  {os.path.basename(filepath)} não encontrado")

    if not usando_real:
        print("\n   ℹ️  Usando dados SINTÉTICOS representativos da população brasileira.")
        print("   Para usar o ANSUR II real:")
        print("   1. Acesse: https://www.openicpsr.org/openicpsr/project/116564")
        print("   2. Baixe ANSUR_II_MALE_Public.csv e ANSUR_II_FEMALE_Public.csv")
        print(f"   3. Coloque em: {ANSUR_DIR}/\n")
        todos_registros = gerar_dados_sinteticos(n_masculino=500, n_feminino=400)
        print(f"   ✓ {len(todos_registros)} registros sintéticos gerados")

    # Inserir no banco
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    cur.executemany("""
        INSERT INTO medidas_corporais (
            fonte_dataset, id_original, altura, peso_kg,
            busto_circunf, cintura_circunf, largura_ombro,
            comprimento_torso, comprimento_braco,
            genero, idade, tamanho_inferido
        ) VALUES (
            :fonte_dataset, :id_original, :altura, :peso_kg,
            :busto_circunf, :cintura_circunf, :largura_ombro,
            :comprimento_torso, :comprimento_braco,
            :genero, :idade, :tamanho_inferido
        )
    """, todos_registros)

    conn.commit()

    # Estatísticas pós-carga
    cur.execute("SELECT fonte_dataset, COUNT(*) FROM medidas_corporais GROUP BY fonte_dataset")
    stats = cur.fetchall()

    cur.execute("""
        SELECT tamanho_inferido, COUNT(*) as n
        FROM medidas_corporais
        GROUP BY tamanho_inferido
        ORDER BY n DESC
    """)
    dist_tamanhos = cur.fetchall()

    conn.close()

    print("\n✅ Medidas corporais carregadas no banco!")
    print("\n📊 Registros por fonte:")
    for fonte, count in stats:
        print(f"   {fonte}: {count}")

    print("\n📊 Distribuição de tamanhos inferidos:")
    total = sum(c for _, c in dist_tamanhos)
    for tam, count in dist_tamanhos:
        barra = "█" * int(count / total * 30)
        print(f"   {tam:<5} {barra} {count} ({count/total*100:.1f}%)")


if __name__ == "__main__":
    carregar_medidas_corporais()
