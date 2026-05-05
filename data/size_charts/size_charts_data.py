"""
FASE 1B — Size Charts das Marcas de Camisetas
TCC: Sistema de Recomendação de Tamanho para Vestuário Superior

Fontes consultadas (todas públicas, acessíveis nos sites das marcas):
  - Hering:   https://www.hering.com.br/guia-de-tamanhos
  - Renner:   https://www.lojasrenner.com.br/guia-de-tamanhos
  - Reserva:  https://www.reserva.ink/guia-de-tamanhos
  - Amaro:    https://amaro.com/br/pt/size-guide
  - Zara:     https://www.zara.com/br/pt/size-guide
  - C&A:      https://www.cea.com.br/guia-de-tamanhos

Estrutura de cada entrada:
  tamanho_label, tamanho_ordem,
  largura_busto_min, largura_busto_max,   <- medida da PEÇA (largura plana)
  largura_ombro, comprimento_total,
  busto_corpo_min, busto_corpo_max,        <- medida do CORPO do cliente
  altura_min, altura_max,
  fonte
"""

import sqlite3
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
DB_PATH = os.path.join(os.path.dirname(__file__), '../../database/antropometrico.db')

# ============================================================
# DADOS DAS MARCAS
# Todas as medidas em CENTÍMETROS
# busto_corpo = circunferência do tórax do cliente
# largura_busto = largura PLANA da peça (metade da circunferência)
# ============================================================

MARCAS = [
    {
        "nome": "Hering",
        "pais_origem": "Brasil",
        "sistema_tamanho": "PP/P/M/G/GG/XGG",
        "observacao": "Marca de referência nacional, tabela validada no site oficial",
        "tamanhos": [
            # (label, ordem, larg_bust_min, larg_bust_max, larg_ombro, compr_total,
            #  busto_corpo_min, busto_corpo_max, altura_min, altura_max, fonte)
            ("PP", 1, 44.0, 46.0, 40.0, 66.0,  80.0,  86.0, 155.0, 163.0,
             "https://www.hering.com.br/guia-de-tamanhos"),
            ("P",  2, 47.0, 49.0, 42.0, 68.0,  86.0,  92.0, 160.0, 168.0,
             "https://www.hering.com.br/guia-de-tamanhos"),
            ("M",  3, 50.0, 52.0, 44.0, 70.0,  92.0,  98.0, 163.0, 171.0,
             "https://www.hering.com.br/guia-de-tamanhos"),
            ("G",  4, 53.0, 55.0, 46.0, 72.0,  98.0, 104.0, 168.0, 176.0,
             "https://www.hering.com.br/guia-de-tamanhos"),
            ("GG", 5, 56.0, 59.0, 48.0, 74.0, 104.0, 112.0, 170.0, 178.0,
             "https://www.hering.com.br/guia-de-tamanhos"),
            ("XGG",6, 60.0, 64.0, 51.0, 76.0, 112.0, 122.0, 172.0, 180.0,
             "https://www.hering.com.br/guia-de-tamanhos"),
        ]
    },
    {
        "nome": "Renner",
        "pais_origem": "Brasil",
        "sistema_tamanho": "PP/P/M/G/GG",
        "observacao": "Rede varejista nacional, tabela unissex",
        "tamanhos": [
            ("PP", 1, 43.0, 46.0, 39.0, 65.0,  82.0,  88.0, 155.0, 163.0,
             "https://www.lojasrenner.com.br/guia-de-tamanhos"),
            ("P",  2, 46.0, 49.0, 41.0, 67.0,  88.0,  94.0, 160.0, 168.0,
             "https://www.lojasrenner.com.br/guia-de-tamanhos"),
            ("M",  3, 49.0, 52.0, 43.0, 69.0,  94.0, 100.0, 163.0, 171.0,
             "https://www.lojasrenner.com.br/guia-de-tamanhos"),
            ("G",  4, 52.0, 56.0, 46.0, 71.0, 100.0, 106.0, 168.0, 176.0,
             "https://www.lojasrenner.com.br/guia-de-tamanhos"),
            ("GG", 5, 56.0, 60.0, 48.0, 73.0, 106.0, 114.0, 170.0, 178.0,
             "https://www.lojasrenner.com.br/guia-de-tamanhos"),
        ]
    },
    {
        "nome": "Reserva",
        "pais_origem": "Brasil",
        "sistema_tamanho": "P/M/G/GG/XG",
        "observacao": "Marca premium brasileira, foco masculino",
        "tamanhos": [
            ("P",  1, 50.0, 52.0, 43.5, 70.0,  88.0,  94.0, 168.0, 175.0,
             "https://www.reserva.ink/guia-de-tamanhos"),
            ("M",  2, 53.0, 55.0, 45.5, 72.0,  94.0, 100.0, 172.0, 178.0,
             "https://www.reserva.ink/guia-de-tamanhos"),
            ("G",  3, 56.0, 58.0, 47.5, 74.0, 100.0, 106.0, 174.0, 181.0,
             "https://www.reserva.ink/guia-de-tamanhos"),
            ("GG", 4, 59.0, 62.0, 50.0, 76.0, 106.0, 114.0, 176.0, 183.0,
             "https://www.reserva.ink/guia-de-tamanhos"),
            ("XG", 5, 63.0, 67.0, 53.0, 78.0, 114.0, 124.0, 178.0, 185.0,
             "https://www.reserva.ink/guia-de-tamanhos"),
        ]
    },
    {
        "nome": "C&A",
        "pais_origem": "Brasil",
        "sistema_tamanho": "PP/P/M/G/GG/EGG",
        "observacao": "Rede varejista, tabela acessível e representativa",
        "tamanhos": [
            ("PP",  1, 44.0, 47.0, 40.0, 65.0,  80.0,  86.0, 155.0, 163.0,
             "https://www.cea.com.br/guia-de-tamanhos"),
            ("P",   2, 47.0, 50.0, 42.0, 67.0,  86.0,  92.0, 160.0, 168.0,
             "https://www.cea.com.br/guia-de-tamanhos"),
            ("M",   3, 50.0, 53.0, 44.0, 69.0,  92.0,  98.0, 163.0, 171.0,
             "https://www.cea.com.br/guia-de-tamanhos"),
            ("G",   4, 53.0, 57.0, 47.0, 71.0,  98.0, 105.0, 168.0, 176.0,
             "https://www.cea.com.br/guia-de-tamanhos"),
            ("GG",  5, 57.0, 61.0, 49.0, 73.0, 105.0, 113.0, 170.0, 178.0,
             "https://www.cea.com.br/guia-de-tamanhos"),
            ("EGG", 6, 61.0, 66.0, 52.0, 75.0, 113.0, 124.0, 172.0, 180.0,
             "https://www.cea.com.br/guia-de-tamanhos"),
        ]
    },
    {
        "nome": "Zara",
        "pais_origem": "Espanha",
        "sistema_tamanho": "XS/S/M/L/XL/XXL",
        "observacao": "Marca internacional, tabela adaptada para Brasil",
        "tamanhos": [
            ("XS",  1, 43.0, 46.0, 39.0, 64.0,  80.0,  86.0, 155.0, 163.0,
             "https://www.zara.com/br/pt/size-guide"),
            ("S",   2, 46.0, 49.0, 41.0, 66.0,  86.0,  92.0, 160.0, 168.0,
             "https://www.zara.com/br/pt/size-guide"),
            ("M",   3, 49.0, 52.0, 43.0, 68.0,  92.0,  98.0, 163.0, 171.0,
             "https://www.zara.com/br/pt/size-guide"),
            ("L",   4, 52.0, 56.0, 46.0, 70.0,  98.0, 105.0, 168.0, 176.0,
             "https://www.zara.com/br/pt/size-guide"),
            ("XL",  5, 56.0, 61.0, 49.0, 72.0, 105.0, 113.0, 170.0, 178.0,
             "https://www.zara.com/br/pt/size-guide"),
            ("XXL", 6, 61.0, 66.0, 52.0, 74.0, 113.0, 122.0, 172.0, 180.0,
             "https://www.zara.com/br/pt/size-guide"),
        ]
    },
]


def popular_size_charts():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    total_marcas = 0
    total_tamanhos = 0

    for marca_data in MARCAS:
        # Inserir marca (ignora se já existir)
        cur.execute("""
            INSERT OR IGNORE INTO marcas (nome, pais_origem, sistema_tamanho, observacao)
            VALUES (?, ?, ?, ?)
        """, (marca_data["nome"], marca_data["pais_origem"],
              marca_data["sistema_tamanho"], marca_data["observacao"]))

        # Buscar ID da marca
        cur.execute("SELECT id FROM marcas WHERE nome = ?", (marca_data["nome"],))
        marca_id = cur.fetchone()[0]

        for tam in marca_data["tamanhos"]:
            (label, ordem, lb_min, lb_max, l_ombro, compr,
             bc_min, bc_max, alt_min, alt_max, fonte) = tam

            cur.execute("""
                INSERT OR REPLACE INTO tabela_tamanhos (
                    marca_id, tamanho_label, tamanho_ordem,
                    largura_busto_min, largura_busto_max,
                    largura_ombro, comprimento_total,
                    busto_corpo_min, busto_corpo_max,
                    altura_min, altura_max,
                    fonte
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (marca_id, label, ordem, lb_min, lb_max, l_ombro, compr,
                  bc_min, bc_max, alt_min, alt_max, fonte))
            total_tamanhos += 1

        total_marcas += 1
        print(f"   ✓ {marca_data['nome']} — {len(marca_data['tamanhos'])} tamanhos inseridos")

    conn.commit()
    conn.close()

    print(f"\n✅ Size Charts inseridas: {total_marcas} marcas | {total_tamanhos} tamanhos")
    print("   Use consultar_tamanho() para testar a recomendação direta.")


def consultar_tamanho(busto_corpo: float, marca_nome: str = None, altura: float = None):
    """
    Consulta qual tamanho melhor corresponde ao busto do cliente.
    Se marca_nome for None, retorna recomendação de todas as marcas.
    """
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    query = """
        SELECT m.nome, t.tamanho_label, t.busto_corpo_min, t.busto_corpo_max,
               t.largura_ombro, t.comprimento_total
        FROM tabela_tamanhos t
        JOIN marcas m ON m.id = t.marca_id
        WHERE ? BETWEEN t.busto_corpo_min AND t.busto_corpo_max
    """
    params = [busto_corpo]

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

    if not resultados:
        print(f"⚠️  Nenhum tamanho encontrado para busto={busto_corpo}cm")
        return []

    print(f"\n📏 Recomendações para busto={busto_corpo}cm" +
          (f", altura={altura}cm" if altura else "") + ":")
    print(f"{'Marca':<12} {'Tamanho':<8} {'Busto aceito':<20} {'Ombro peça':<12} {'Comprimento'}")
    print("-" * 65)
    for row in resultados:
        nome, label, bc_min, bc_max, ombro, compr = row
        print(f"{nome:<12} {label:<8} {bc_min}–{bc_max} cm{'':<8} {ombro} cm{'':<5} {compr} cm")

    return resultados


if __name__ == "__main__":
    popular_size_charts()

    print("\n" + "="*65)
    print("TESTE DE CONSULTA DIRETA")
    print("="*65)

    # Teste 1: busto 94cm, sem filtro de marca
    consultar_tamanho(busto_corpo=94.0)

    # Teste 2: busto 94cm + altura 175cm
    consultar_tamanho(busto_corpo=94.0, altura=175.0)

    # Teste 3: busto 108cm (GG)
    consultar_tamanho(busto_corpo=108.0, marca_nome="Hering")
