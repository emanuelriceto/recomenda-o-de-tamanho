"""
FASE 1C — Carga do Dataset ANSUR II no Banco de Dados
TCC: Sistema de Recomendação de Tamanho para Vestuário Superior

ESCOPO: apenas masculino. Dados femininos foram removidos do projeto.

ANSUR II (Army Anthropometric Survey II, 2012)
  - Dataset público do Exército dos EUA
  - 4.082 militares MASCULINOS com medidas corporais detalhadas
  - Medidas originais em MILÍMETROS e décimos de KG
  - Download: https://www.openicpsr.org/openicpsr/project/116564

Como baixar (sem login):
  1. Acesse: https://www.openicpsr.org/openicpsr/project/116564
  2. Baixe: ANSUR_II_MALE_Public.csv
  3. Coloque em: data/ansur/
"""

import sqlite3
import os
import csv

DB_PATH  = os.path.join(os.path.dirname(__file__), '../../database/antropometrico.db')
ANSUR_DIR = os.path.dirname(__file__)


def inferir_tamanho(busto_cm: float) -> str:
    """
    Inferência de tamanho com base na circunferência do busto.
    Referência: tabela Hering masculino.
    """
    tabela = {
        'PP':  (0,    86),
        'P':   (86,   92),
        'M':   (92,   98),
        'G':   (98,  104),
        'GG':  (104, 112),
        'XGG': (112, 999),
    }
    for tamanho, (minv, maxv) in tabela.items():
        if minv <= busto_cm < maxv:
            return tamanho
    return 'XGG'


def processar_arquivo_ansur(filepath: str) -> list:
    """
    Lê o CSV do ANSUR II masculino e retorna lista de dicionários.
    Conversões:
      - medidas em mm  → ÷ 10 → cm
      - weightkg       → ÷ 10 → kg  (armazenado em décimos de kg)
    """
    registros = []

    with open(filepath, newline='', encoding='latin-1') as f:
        reader = csv.DictReader(f)

        for i, row in enumerate(reader):
            row_lower = {k.lower(): v for k, v in row.items()}

            def mm2cm(col):
                val = row_lower.get(col, '').strip()
                return round(float(val) / 10, 1) if val else None

            def decikg(col):
                val = row_lower.get(col, '').strip()
                return round(float(val) / 10, 1) if val else None

            try:
                altura  = mm2cm('stature')
                peso    = decikg('weightkg')
                busto   = mm2cm('chestcircumference')
                cintura = mm2cm('waistcircumference')
                ombro   = mm2cm('biacromialbreadth')
                braco   = mm2cm('sleeveoutseam')
                torso   = mm2cm('cervicaleheight')
                id_orig = row_lower.get('subjectid', str(i))
                idade_s = row_lower.get('age', '').strip()
                idade   = int(float(idade_s)) if idade_s else None

                if not busto:
                    continue

                registros.append({
                    'fonte_dataset':     'ANSUR_II_M',
                    'id_original':       id_orig,
                    'altura':            altura,
                    'peso_kg':           peso,
                    'busto_circunf':     busto,
                    'cintura_circunf':   cintura,
                    'largura_ombro':     ombro,
                    'comprimento_torso': torso,
                    'comprimento_braco': braco,
                    'genero':            'M',
                    'idade':             idade,
                    'tamanho_inferido':  inferir_tamanho(busto),
                })
            except (ValueError, KeyError):
                continue

    return registros


def gerar_dados_sinteticos_masculinos(n: int = 600) -> list:
    """
    Gera dados sintéticos representativos da população MASCULINA brasileira.
    Baseado nas médias do IBGE POF e literatura antropométrica nacional.
    Usado como fallback quando o CSV do ANSUR II não está disponível.
    """
    import random
    random.seed(42)

    def normal_clamp(media, std, minv, maxv):
        for _ in range(100):
            v = random.gauss(media, std)
            if minv <= v <= maxv:
                return round(v, 1)
        return round(media, 1)

    # Parâmetros baseados em médias brasileiras masculinas (IBGE POF 2017-2018)
    params = {
        'altura':   (173.0, 7.5,  155.0, 200.0),
        'peso_kg':  (82.0,  15.0,  55.0, 140.0),
        'busto':    (98.0,  10.0,  78.0, 130.0),
        'cintura':  (90.0,  12.0,  68.0, 130.0),
        'ombro':    (43.5,   3.0,  36.0,  54.0),
        'braco':    (85.0,   5.0,  70.0, 100.0),
    }

    registros = []
    for i in range(n):
        altura  = normal_clamp(*params['altura'])
        peso    = normal_clamp(*params['peso_kg'])
        busto   = normal_clamp(*params['busto'])
        cintura = normal_clamp(*params['cintura'])
        ombro   = normal_clamp(*params['ombro'])
        braco   = normal_clamp(*params['braco'])
        idade   = random.randint(18, 65)

        registros.append({
            'fonte_dataset':     'SINTETICO_M_BR',
            'id_original':       f'SYN_M_{i:04d}',
            'altura':            altura,
            'peso_kg':           peso,
            'busto_circunf':     busto,
            'cintura_circunf':   cintura,
            'largura_ombro':     ombro,
            'comprimento_torso': None,
            'comprimento_braco': braco,
            'genero':            'M',
            'idade':             idade,
            'tamanho_inferido':  inferir_tamanho(busto),
        })

    return registros


def carregar_medidas_corporais():
    """
    Tenta carregar o ANSUR II masculino real.
    Se não encontrado, usa dados sintéticos masculinos brasileiros.
    """
    filepath = os.path.join(ANSUR_DIR, 'ANSUR_II_MALE_Public.csv')
    usando_real = False

    if os.path.exists(filepath):
        print('   📂 ANSUR_II_MALE_Public.csv encontrado — carregando dados reais...')
        registros = processar_arquivo_ansur(filepath)
        usando_real = True
        print(f'      → {len(registros)} registros masculinos carregados')
    else:
        print('   ⚠️  ANSUR_II_MALE_Public.csv não encontrado.')
        print('      Usando dados sintéticos representativos (masculino brasileiro).')
        print(f'      Para usar dados reais: https://www.openicpsr.org/openicpsr/project/116564\n')
        registros = gerar_dados_sinteticos_masculinos(n=600)
        print(f'      → {len(registros)} registros sintéticos gerados')

    # Inserir no banco
    conn = sqlite3.connect(DB_PATH)
    cur  = conn.cursor()

    # Limpar registros anteriores para evitar duplicatas
    cur.execute("DELETE FROM medidas_corporais")

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
    """, registros)

    conn.commit()

    # Estatísticas
    cur.execute("""
        SELECT tamanho_inferido, COUNT(*) as n
        FROM medidas_corporais
        GROUP BY tamanho_inferido
        ORDER BY n DESC
    """)
    dist = cur.fetchall()

    cur.execute("SELECT COUNT(*) FROM medidas_corporais")
    total = cur.fetchone()[0]

    conn.close()

    print(f'\n✅ Medidas corporais carregadas — {total} registros masculinos')
    print(f'   Fonte: {"ANSUR II real" if usando_real else "Sintético masculino brasileiro"}')
    print('\n📊 Distribuição de tamanhos:')
    for tam, cnt in dist:
        barra = '█' * int(cnt / total * 30)
        print(f'   {tam:<5} {barra} {cnt} ({cnt/total*100:.1f}%)')


if __name__ == '__main__':
    carregar_medidas_corporais()
