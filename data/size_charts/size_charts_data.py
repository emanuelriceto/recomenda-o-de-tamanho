"""
FASE 1B — Size Charts com Modelagens
TCC: Sistema de Recomendação de Tamanho para Vestuário Superior

ATUALIZAÇÃO: cada tamanho agora tem 5 entradas (uma por modelagem):
  regular, slim, oversized, longline, henley

Critério de ajuste das medidas por modelagem:
  slim      → largura_busto -2cm, ombro -0.5cm (corte mais justo)
  oversized → largura_busto +6cm, ombro +2cm, comprimento +8cm
  longline  → comprimento +12cm (mais comprido, resto igual ao regular)
  henley    → ombro +0.5cm (gola estruturada, ombro levemente maior)

Fontes: sites oficiais das marcas (guias de tamanho)
"""

import sqlite3
import os

DB_PATH = os.path.join(os.path.dirname(__file__), "../../database/antropometrico.db")

# ── Modelagens suportadas ────────────────────────────────────
MODELAGENS = ["regular", "slim", "oversized", "longline", "henley"]

# ── Ajustes de medida por modelagem (relativos ao regular) ───
# (delta_largura_busto, delta_ombro, delta_comprimento)
AJUSTES_MODELAGEM = {
    "regular":  ( 0.0,  0.0,  0.0),
    "slim":     (-2.0, -0.5,  0.0),
    "oversized":(+6.0, +2.0, +8.0),
    "longline": ( 0.0,  0.0, +12.0),
    "henley":   ( 0.0, +0.5,  0.0),
}

# ── Dados base das marcas (modelagem regular) ────────────────
# Formato de cada tamanho:
# (label, ordem, larg_bust_min, larg_bust_max, larg_ombro,
#  compr_total, busto_corpo_min, busto_corpo_max,
#  altura_min, altura_max, fonte)

MARCAS = [
    {
        "nome": "Hering",
        "pais_origem": "Brasil",
        "sistema_tamanho": "PP/P/M/G/GG/XGG",
        "observacao": "Marca de referência nacional",
        "tamanhos_regular": [
            ("PP", 1, 44.0, 46.0, 40.0, 66.0, 80.0,  86.0, 155.0, 163.0,
             "https://www.hering.com.br/guia-de-tamanhos"),
            ("P",  2, 47.0, 49.0, 42.0, 68.0, 86.0,  92.0, 160.0, 168.0,
             "https://www.hering.com.br/guia-de-tamanhos"),
            ("M",  3, 50.0, 52.0, 44.0, 70.0, 92.0,  98.0, 163.0, 171.0,
             "https://www.hering.com.br/guia-de-tamanhos"),
            ("G",  4, 53.0, 55.0, 46.0, 72.0, 98.0, 104.0, 168.0, 176.0,
             "https://www.hering.com.br/guia-de-tamanhos"),
            ("GG", 5, 56.0, 59.0, 48.0, 74.0,104.0, 112.0, 170.0, 178.0,
             "https://www.hering.com.br/guia-de-tamanhos"),
            ("XGG",6, 60.0, 64.0, 51.0, 76.0,112.0, 122.0, 172.0, 180.0,
             "https://www.hering.com.br/guia-de-tamanhos"),
        ]
    },
    {
        "nome": "Renner",
        "pais_origem": "Brasil",
        "sistema_tamanho": "PP/P/M/G/GG",
        "observacao": "Rede varejista nacional",
        "tamanhos_regular": [
            ("PP", 1, 43.0, 46.0, 39.0, 65.0, 82.0,  88.0, 155.0, 163.0,
             "https://www.lojasrenner.com.br/guia-de-tamanhos"),
            ("P",  2, 46.0, 49.0, 41.0, 67.0, 88.0,  94.0, 160.0, 168.0,
             "https://www.lojasrenner.com.br/guia-de-tamanhos"),
            ("M",  3, 49.0, 52.0, 43.0, 69.0, 94.0, 100.0, 163.0, 171.0,
             "https://www.lojasrenner.com.br/guia-de-tamanhos"),
            ("G",  4, 52.0, 56.0, 46.0, 71.0,100.0, 106.0, 168.0, 176.0,
             "https://www.lojasrenner.com.br/guia-de-tamanhos"),
            ("GG", 5, 56.0, 60.0, 48.0, 73.0,106.0, 114.0, 170.0, 178.0,
             "https://www.lojasrenner.com.br/guia-de-tamanhos"),
        ]
    },
    {
        "nome": "Reserva",
        "pais_origem": "Brasil",
        "sistema_tamanho": "P/M/G/GG/XG",
        "observacao": "Marca premium brasileira, foco masculino",
        "tamanhos_regular": [
            ("P",  1, 50.0, 52.0, 43.5, 70.0, 88.0,  94.0, 168.0, 175.0,
             "https://www.reserva.ink/guia-de-tamanhos"),
            ("M",  2, 53.0, 55.0, 45.5, 72.0, 94.0, 100.0, 172.0, 178.0,
             "https://www.reserva.ink/guia-de-tamanhos"),
            ("G",  3, 56.0, 58.0, 47.5, 74.0,100.0, 106.0, 174.0, 181.0,
             "https://www.reserva.ink/guia-de-tamanhos"),
            ("GG", 4, 59.0, 62.0, 50.0, 76.0,106.0, 114.0, 176.0, 183.0,
             "https://www.reserva.ink/guia-de-tamanhos"),
            ("XG", 5, 63.0, 67.0, 53.0, 78.0,114.0, 124.0, 178.0, 185.0,
             "https://www.reserva.ink/guia-de-tamanhos"),
        ]
    },
    {
        "nome": "C&A",
        "pais_origem": "Brasil",
        "sistema_tamanho": "PP/P/M/G/GG/EGG",
        "observacao": "Rede varejista, tabela acessível",
        "tamanhos_regular": [
            ("PP",  1, 44.0, 47.0, 40.0, 65.0, 80.0,  86.0, 155.0, 163.0,
             "https://www.cea.com.br/guia-de-tamanhos"),
            ("P",   2, 47.0, 50.0, 42.0, 67.0, 86.0,  92.0, 160.0, 168.0,
             "https://www.cea.com.br/guia-de-tamanhos"),
            ("M",   3, 50.0, 53.0, 44.0, 69.0, 92.0,  98.0, 163.0, 171.0,
             "https://www.cea.com.br/guia-de-tamanhos"),
            ("G",   4, 53.0, 57.0, 47.0, 71.0, 98.0, 105.0, 168.0, 176.0,
             "https://www.cea.com.br/guia-de-tamanhos"),
            ("GG",  5, 57.0, 61.0, 49.0, 73.0,105.0, 113.0, 170.0, 178.0,
             "https://www.cea.com.br/guia-de-tamanhos"),
            ("EGG", 6, 61.0, 66.0, 52.0, 75.0,113.0, 124.0, 172.0, 180.0,
             "https://www.cea.com.br/guia-de-tamanhos"),
        ]
    },
    {
        "nome": "Zara",
        "pais_origem": "Espanha",
        "sistema_tamanho": "XS/S/M/L/XL/XXL",
        "observacao": "Marca internacional, tabela adaptada para Brasil",
        "tamanhos_regular": [
            ("XS",  1, 43.0, 46.0, 39.0, 64.0, 80.0,  86.0, 155.0, 163.0,
             "https://www.zara.com/br/pt/size-guide"),
            ("S",   2, 46.0, 49.0, 41.0, 66.0, 86.0,  92.0, 160.0, 168.0,
             "https://www.zara.com/br/pt/size-guide"),
            ("M",   3, 49.0, 52.0, 43.0, 68.0, 92.0,  98.0, 163.0, 171.0,
             "https://www.zara.com/br/pt/size-guide"),
            ("L",   4, 52.0, 56.0, 46.0, 70.0, 98.0, 105.0, 168.0, 176.0,
             "https://www.zara.com/br/pt/size-guide"),
            ("XL",  5, 56.0, 61.0, 49.0, 72.0,105.0, 113.0, 170.0, 178.0,
             "https://www.zara.com/br/pt/size-guide"),
            ("XXL", 6, 61.0, 66.0, 52.0, 74.0,113.0, 122.0, 172.0, 180.0,
             "https://www.zara.com/br/pt/size-guide"),
        ]
    },
]


def gerar_tamanhos_por_modelagem(tamanhos_regular: list) -> list:
    """
    Recebe os tamanhos base (regular) e gera automaticamente
    os tamanhos para todas as modelagens aplicando os ajustes.
    Retorna lista de tuplas prontas para inserção.
    """
    todas_entradas = []

    for modelagem, (d_bust, d_ombro, d_compr) in AJUSTES_MODELAGEM.items():
        for tam in tamanhos_regular:
            (label, ordem, lb_min, lb_max, l_ombro, compr,
             bc_min, bc_max, alt_min, alt_max, fonte) = tam

            todas_entradas.append((
                label, ordem, modelagem,
                round(lb_min + d_bust, 1),
                round(lb_max + d_bust, 1),
                round(l_ombro + d_ombro, 1),
                round(compr + d_compr, 1),
                bc_min, bc_max,
                alt_min, alt_max,
                fonte
            ))

    return todas_entradas


def popular_size_charts():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    total_marcas   = 0
    total_tamanhos = 0

    for marca_data in MARCAS:
        cur.execute("""
            INSERT OR IGNORE INTO marcas
            (nome, pais_origem, sistema_tamanho, observacao)
            VALUES (?, ?, ?, ?)
        """, (marca_data["nome"], marca_data["pais_origem"],
              marca_data["sistema_tamanho"], marca_data["observacao"]))

        cur.execute("SELECT id FROM marcas WHERE nome = ?", (marca_data["nome"],))
        marca_id = cur.fetchone()[0]

        entradas = gerar_tamanhos_por_modelagem(marca_data["tamanhos_regular"])

        for entrada in entradas:
            (label, ordem, modelagem, lb_min, lb_max,
             l_ombro, compr, bc_min, bc_max,
             alt_min, alt_max, fonte) = entrada

            cur.execute("""
                INSERT OR REPLACE INTO tabela_tamanhos (
                    marca_id, tamanho_label, tamanho_ordem, modelagem,
                    largura_busto_min, largura_busto_max,
                    largura_ombro, comprimento_total,
                    busto_corpo_min, busto_corpo_max,
                    altura_min, altura_max, fonte
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (marca_id, label, ordem, modelagem,
                  lb_min, lb_max, l_ombro, compr,
                  bc_min, bc_max, alt_min, alt_max, fonte))
            total_tamanhos += 1

        total_marcas += 1
        n_tam = len(marca_data["tamanhos_regular"])
        print(f"   ✓ {marca_data['nome']:<10} "
              f"{n_tam} tamanhos × {len(MODELAGENS)} modelagens "
              f"= {n_tam * len(MODELAGENS)} entradas")

    conn.commit()
    conn.close()
    print(f"\n✅ Size Charts inseridas:")
    print(f"   {total_marcas} marcas | {total_tamanhos} entradas totais")
    print(f"   Modelagens: {', '.join(MODELAGENS)}")


def consultar_tamanho(busto_corpo: float, modelagem: str = "regular",
                      marca_nome: str = None, altura: float = None):
    """
    Consulta qual tamanho corresponde ao busto do cliente
    para uma modelagem específica.
    """
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    query = """
        SELECT m.nome, t.tamanho_label, t.modelagem,
               t.busto_corpo_min, t.busto_corpo_max,
               t.largura_ombro, t.comprimento_total
        FROM tabela_tamanhos t
        JOIN marcas m ON m.id = t.marca_id
        WHERE ? BETWEEN t.busto_corpo_min AND t.busto_corpo_max
          AND t.modelagem = ?
    """
    params = [busto_corpo, modelagem]

    if marca_nome:
        query += " AND m.nome = ?"
        params.append(marca_nome)

    if altura:
        query += " AND ? BETWEEN t.altura_min AND t.altura_max"
        params.append(altura)

    query += " ORDER BY m.nome, t.tamanho_ordem"
    cur.execute(query, params)
    resultados = cur.fetchall()
    conn.close()

    print(f"\n📏 Recomendações: busto={busto_corpo}cm | "
          f"modelagem={modelagem}" +
          (f" | altura={altura}cm" if altura else ""))
    print(f"{'Marca':<12} {'Tamanho':<8} {'Modelagem':<12} "
          f"{'Busto aceito':<20} {'Ombro':<10} {'Compr.'}")
    print("-" * 72)

    for row in resultados:
        nome, label, mod, bc_min, bc_max, ombro, compr = row
        print(f"{nome:<12} {label:<8} {mod:<12} "
              f"{bc_min}–{bc_max} cm{'':<7} {ombro} cm{'':<3} {compr} cm")

    return resultados


if __name__ == "__main__":
    popular_size_charts()

    print("\n" + "="*72)
    print("TESTES DE CONSULTA")
    print("="*72)
    consultar_tamanho(94.0, modelagem="regular")
    consultar_tamanho(94.0, modelagem="slim")
    consultar_tamanho(94.0, modelagem="oversized")
