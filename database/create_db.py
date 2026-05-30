"""
FASE 1A — Criação do Banco de Dados Antropométrico
TCC: Sistema de Recomendação de Tamanho para Vestuário Superior

ATUALIZAÇÃO: coluna 'modelagem' adicionada em tabela_tamanhos
Modelagens suportadas: regular, slim, oversized, longline, henley
"""

import sqlite3
import os

DB_PATH = os.path.join(os.path.dirname(__file__), "antropometrico.db")


def criar_banco():
    # Remove banco antigo se existir para recriar do zero
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)

    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    # ── Tabela 1: Marcas ─────────────────────────────────────
    cur.execute("""
        CREATE TABLE IF NOT EXISTS marcas (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            nome            TEXT NOT NULL UNIQUE,
            pais_origem     TEXT DEFAULT 'Brasil',
            sistema_tamanho TEXT DEFAULT 'PP/P/M/G/GG',
            observacao      TEXT,
            criado_em       TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # ── Tabela 2: Tabela de Tamanhos ─────────────────────────
    # ATUALIZAÇÃO: coluna 'modelagem' adicionada
    # Cada linha = um tamanho + uma modelagem de uma marca
    # Exemplo: Hering | M | slim tem medidas diferentes de Hering | M | oversized
    cur.execute("""
        CREATE TABLE IF NOT EXISTS tabela_tamanhos (
            id                  INTEGER PRIMARY KEY AUTOINCREMENT,
            marca_id            INTEGER NOT NULL REFERENCES marcas(id),
            tamanho_label       TEXT NOT NULL,
            tamanho_ordem       INTEGER,

            -- NOVO: modelagem da camiseta
            -- Valores: 'regular', 'slim', 'oversized', 'longline', 'henley'
            modelagem           TEXT NOT NULL DEFAULT 'regular',

            -- Medidas da PEÇA (em cm)
            largura_busto_min   REAL,
            largura_busto_max   REAL,
            largura_ombro       REAL,
            comprimento_total   REAL,

            -- Medidas do CORPO recomendadas (em cm)
            busto_corpo_min     REAL,
            busto_corpo_max     REAL,
            altura_min          REAL,
            altura_max          REAL,

            fonte               TEXT,
            criado_em           TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

            UNIQUE(marca_id, tamanho_label, modelagem)
        )
    """)

    # ── Tabela 3: Medidas Corporais ──────────────────────────
    cur.execute("""
        CREATE TABLE IF NOT EXISTS medidas_corporais (
            id                  INTEGER PRIMARY KEY AUTOINCREMENT,
            fonte_dataset       TEXT NOT NULL,
            id_original         TEXT,
            altura              REAL,
            peso_kg             REAL,
            busto_circunf       REAL,
            cintura_circunf     REAL,
            largura_ombro       REAL,
            comprimento_torso   REAL,
            comprimento_braco   REAL,
            genero              TEXT,
            idade               INTEGER,
            tamanho_inferido    TEXT,
            criado_em           TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # ── Tabela 4: Clientes ───────────────────────────────────
    cur.execute("""
        CREATE TABLE IF NOT EXISTS clientes (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            nome            TEXT,
            email           TEXT UNIQUE,
            genero          TEXT,
            altura          REAL,
            peso_kg         REAL,
            busto_circunf   REAL,
            cintura_circunf REAL,
            largura_ombro   REAL,
            criado_em       TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            atualizado_em   TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # ── Tabela 5: Recomendações ──────────────────────────────
    # ATUALIZAÇÃO: campo modelagem_detectada adicionado
    cur.execute("""
        CREATE TABLE IF NOT EXISTS recomendacoes (
            id                      INTEGER PRIMARY KEY AUTOINCREMENT,
            cliente_id              INTEGER REFERENCES clientes(id),
            marca_id                INTEGER REFERENCES marcas(id),
            tamanho_recomendado     TEXT NOT NULL,
            confianca               REAL,
            metodo                  TEXT,

            -- NOVO: modelagem detectada pelo YOLO
            modelagem_detectada     TEXT,

            -- Medidas usadas
            busto_usado             REAL,
            ombro_usado             REAL,
            altura_usada            REAL,

            -- Feedback
            feedback_correto        INTEGER,
            feedback_tamanho_real   TEXT,
            criado_em               TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # ── Índices ──────────────────────────────────────────────
    cur.execute("CREATE INDEX IF NOT EXISTS idx_tamanhos_marca     ON tabela_tamanhos(marca_id)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_tamanhos_modelagem ON tabela_tamanhos(modelagem)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_medidas_fonte      ON medidas_corporais(fonte_dataset)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_recomendacoes_cli  ON recomendacoes(cliente_id)")

    conn.commit()
    conn.close()
    print(f"✅ Banco criado em: {DB_PATH}")
    print("   Tabelas: marcas, tabela_tamanhos (com modelagem),")
    print("            medidas_corporais, clientes, recomendacoes")


if __name__ == "__main__":
    criar_banco()
