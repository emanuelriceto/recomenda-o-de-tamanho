"""
FASE 1A — Criação do Banco de Dados Antropométrico
TCC: Sistema de Recomendação de Tamanho para Vestuário Superior

Tabelas criadas:
  - marcas              : cadastro das marcas de roupa
  - tabela_tamanhos     : medidas físicas da ROUPA por tamanho e marca
  - medidas_corporais   : medidas do CORPO HUMANO (do ANSUR II)
  - clientes            : dados e medidas do cliente (input do sistema)
  - recomendacoes       : histórico de recomendações feitas
"""

import sqlite3
import os

DB_PATH = os.path.join(os.path.dirname(__file__), "antropometrico.db")


def criar_banco():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    # ------------------------------------------------------------------ #
    # TABELA 1 — Marcas cadastradas                                        #
    # ------------------------------------------------------------------ #
    cur.execute("""
        CREATE TABLE IF NOT EXISTS marcas (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            nome            TEXT NOT NULL UNIQUE,
            pais_origem     TEXT DEFAULT 'Brasil',
            sistema_tamanho TEXT DEFAULT 'PP/P/M/G/GG',  -- ou S/M/L/XL, 36/38/40...
            observacao      TEXT,
            criado_em       TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # ------------------------------------------------------------------ #
    # TABELA 2 — Tabela de tamanhos (medidas da ROUPA)                    #
    #                                                                      #
    # Todas as medidas são em CENTÍMETROS e referem-se à PEÇA em si,      #
    # não ao corpo. Cada linha = um tamanho de uma marca específica.      #
    # ------------------------------------------------------------------ #
    cur.execute("""
        CREATE TABLE IF NOT EXISTS tabela_tamanhos (
            id                  INTEGER PRIMARY KEY AUTOINCREMENT,
            marca_id            INTEGER NOT NULL REFERENCES marcas(id),
            tamanho_label       TEXT NOT NULL,   -- Ex: 'P', 'M', 'G', '38', 'S'
            tamanho_ordem       INTEGER,         -- Para ordenação: PP=1, P=2, M=3...

            -- Medidas da PEÇA (o que aparece na ficha técnica da roupa)
            largura_busto_min   REAL,            -- menor largura aceitável do busto do cliente
            largura_busto_max   REAL,            -- maior largura aceitável do busto do cliente
            largura_ombro       REAL,            -- largura entre costuras dos ombros
            comprimento_total   REAL,            -- do ombro até a barra
            largura_manga       REAL,            -- circunferência da manga (se houver)
            comprimento_manga   REAL,            -- do ombro até o punho

            -- Medidas do CORPO recomendadas para este tamanho (campo cliente)
            busto_corpo_min     REAL,            -- circunferência mínima do busto do cliente
            busto_corpo_max     REAL,            -- circunferência máxima do busto do cliente
            altura_min          REAL,            -- altura mínima recomendada
            altura_max          REAL,            -- altura máxima recomendada

            fonte               TEXT,            -- URL ou nome da fonte consultada
            criado_em           TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

            UNIQUE(marca_id, tamanho_label)
        )
    """)

    # ------------------------------------------------------------------ #
    # TABELA 3 — Medidas corporais (dados do ANSUR II / CAESAR)           #
    #                                                                      #
    # Cada linha é uma pessoa real do dataset com suas medidas.           #
    # Usada para treinar e validar o modelo de ML da Fase 3.              #
    # ------------------------------------------------------------------ #
    cur.execute("""
        CREATE TABLE IF NOT EXISTS medidas_corporais (
            id                  INTEGER PRIMARY KEY AUTOINCREMENT,
            fonte_dataset       TEXT NOT NULL,   -- 'ANSUR_II_MALE', 'ANSUR_II_FEMALE', 'CAESAR'
            id_original         TEXT,            -- ID do sujeito no dataset original

            -- Medidas principais (em cm, convertidas de mm se necessário)
            altura              REAL,            -- stature
            peso_kg             REAL,            -- weightkg
            busto_circunf       REAL,            -- chestcircumference (circunferência do tórax)
            cintura_circunf     REAL,            -- waistcircumference
            quadril_circunf     REAL,            -- hipbreadth * 2 (aproximação)
            largura_ombro       REAL,            -- shoulderbreadth
            comprimento_torso   REAL,            -- cervicaleheight - waistbacklength aprox.
            comprimento_braco   REAL,            -- sleeveoutseam

            -- Dados demográficos
            genero              TEXT,            -- 'M' ou 'F'
            idade               INTEGER,

            -- Tamanho inferido (calculado pelo script de carga)
            tamanho_inferido    TEXT,            -- 'P', 'M', 'G', etc.

            criado_em           TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # ------------------------------------------------------------------ #
    # TABELA 4 — Clientes (inputs do usuário final)                       #
    # ------------------------------------------------------------------ #
    cur.execute("""
        CREATE TABLE IF NOT EXISTS clientes (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            nome            TEXT,
            email           TEXT UNIQUE,
            genero          TEXT,                -- 'M', 'F', 'Outro'

            -- Medidas informadas pelo cliente
            altura          REAL,               -- em cm
            peso_kg         REAL,
            busto_circunf   REAL,               -- circunferência do busto/tórax em cm
            cintura_circunf REAL,
            largura_ombro   REAL,               -- distância entre ombros em cm

            criado_em       TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            atualizado_em   TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # ------------------------------------------------------------------ #
    # TABELA 5 — Histórico de recomendações                               #
    # ------------------------------------------------------------------ #
    cur.execute("""
        CREATE TABLE IF NOT EXISTS recomendacoes (
            id                  INTEGER PRIMARY KEY AUTOINCREMENT,
            cliente_id          INTEGER REFERENCES clientes(id),
            marca_id            INTEGER REFERENCES marcas(id),

            -- Resultado
            tamanho_recomendado TEXT NOT NULL,
            confianca           REAL,            -- 0.0 a 1.0
            metodo              TEXT,            -- 'tabela_direta', 'ml_model', 'yolo_foto'

            -- Medidas usadas no momento da recomendação (snapshot)
            busto_usado         REAL,
            ombro_usado         REAL,
            altura_usada        REAL,

            -- Feedback posterior do cliente (opcional)
            feedback_correto    INTEGER,         -- 1=correto, 0=errado, NULL=sem feedback
            feedback_tamanho_real TEXT,          -- o que o cliente realmente usou

            criado_em           TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # ------------------------------------------------------------------ #
    # ÍNDICES para performance de consulta                                 #
    # ------------------------------------------------------------------ #
    cur.execute("CREATE INDEX IF NOT EXISTS idx_tamanhos_marca ON tabela_tamanhos(marca_id)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_medidas_fonte ON medidas_corporais(fonte_dataset)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_recomendacoes_cliente ON recomendacoes(cliente_id)")

    conn.commit()
    conn.close()
    print(f"✅ Banco criado com sucesso em: {DB_PATH}")
    print("   Tabelas: marcas, tabela_tamanhos, medidas_corporais, clientes, recomendacoes")


if __name__ == "__main__":
    criar_banco()
