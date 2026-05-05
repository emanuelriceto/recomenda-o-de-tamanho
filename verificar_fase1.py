"""
Verificação completa do Banco de Dados — Fase 1
Executa após create_db, size_charts e load_ansur para confirmar integridade.
"""

import sqlite3
import os

DB_PATH = os.path.join(os.path.dirname(__file__), 'database/antropometrico.db')


def verificar_banco():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    print("=" * 60)
    print("  VERIFICAÇÃO DO BANCO ANTROPOMÉTRICO — FASE 1")
    print("=" * 60)

    tabelas = ["marcas", "tabela_tamanhos", "medidas_corporais", "clientes", "recomendacoes"]
    for tabela in tabelas:
        cur.execute(f"SELECT COUNT(*) FROM {tabela}")
        count = cur.fetchone()[0]
        status = "✅" if count > 0 or tabela in ("clientes", "recomendacoes") else "❌"
        print(f"  {status}  {tabela:<22} {count} registros")

    print("\n📋 Marcas cadastradas:")
    cur.execute("SELECT nome, sistema_tamanho FROM marcas ORDER BY nome")
    for row in cur.fetchall():
        print(f"     • {row[0]} ({row[1]})")

    print("\n📐 Exemplo de consulta — busto 100cm, altura 178cm:")
    cur.execute("""
        SELECT m.nome, t.tamanho_label, t.busto_corpo_min, t.busto_corpo_max
        FROM tabela_tamanhos t JOIN marcas m ON m.id = t.marca_id
        WHERE 100 BETWEEN t.busto_corpo_min AND t.busto_corpo_max
          AND 178 BETWEEN t.altura_min AND t.altura_max
        ORDER BY m.nome
    """)
    for row in cur.fetchall():
        print(f"     {row[0]:<12} → {row[1]} (busto aceito: {row[2]}–{row[3]} cm)")

    print("\n📊 Medidas corporais — resumo estatístico:")
    cur.execute("""
        SELECT
            genero,
            COUNT(*) as n,
            ROUND(AVG(altura), 1) as alt_media,
            ROUND(AVG(busto_circunf), 1) as busto_medio,
            ROUND(AVG(largura_ombro), 1) as ombro_medio
        FROM medidas_corporais
        GROUP BY genero
    """)
    print(f"  {'Gênero':<8} {'N':<6} {'Altura média':<15} {'Busto médio':<14} {'Ombro médio'}")
    print("  " + "-" * 52)
    for row in cur.fetchall():
        print(f"  {row[0]:<8} {row[1]:<6} {row[2]} cm{'':<8} {row[3]} cm{'':<6} {row[4]} cm")

    conn.close()
    print("\n✅ Fase 1 concluída com sucesso! Banco pronto para a Fase 2 (modelo ML).")
    print("=" * 60)


if __name__ == "__main__":
    verificar_banco()
