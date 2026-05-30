"""
FASE 2B — Preparação dos Dados para o Modelo de ML
TCC: Sistema de Recomendação de Tamanho para Vestuário Superior

ESCOPO: apenas masculino.
Modelagens YOLO suportadas: oversized (0), regular (1), slim (2)
  — conforme data.yaml do Roboflow: ['oversized', 'regular', 'slim']
"""

import sqlite3
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
DB_PATH   = os.path.join(BASE_DIR, '../database/antropometrico.db')
OUT_DIR   = BASE_DIR
PLOTS_DIR = os.path.join(BASE_DIR, '../docs/plots')
os.makedirs(OUT_DIR,   exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

sns.set_theme(style='whitegrid', palette='Set2', font_scale=1.1)

# ── Modelagens — ordem exata do data.yaml do Roboflow ─────────
# names: ['oversized', 'regular', 'slim']
MODELAGEM_MAP = {
    'oversized': 0,
    'regular':   1,
    'slim':      2,
}

CORES_TAMANHO = {
    'PP':  '#4e9af1', 'P':   '#6cbe6c', 'M':   '#f5c242',
    'G':   '#f0855a', 'GG':  '#c06ab3', 'XGG': '#888888',
    'XS':  '#4e9af1', 'S':   '#6cbe6c', 'L':   '#f0855a',
    'XL':  '#c06ab3', 'XXL': '#888888', 'EGG': '#555555',
    'XG':  '#999999',
}


def carregar_dados():
    conn = sqlite3.connect(DB_PATH)

    # Apenas masculino
    df = pd.read_sql_query("""
        SELECT id, fonte_dataset, idade,
               altura, peso_kg, busto_circunf,
               cintura_circunf, largura_ombro,
               comprimento_braco, tamanho_inferido
        FROM medidas_corporais
        WHERE busto_circunf IS NOT NULL
          AND altura IS NOT NULL
          AND genero = 'M'
    """, conn)

    size_charts = pd.read_sql_query("""
        SELECT m.nome as marca, t.tamanho_label,
               t.tamanho_ordem, t.modelagem,
               t.busto_corpo_min, t.busto_corpo_max,
               t.largura_ombro as ombro_peca,
               t.comprimento_total,
               t.altura_min, t.altura_max
        FROM tabela_tamanhos t
        JOIN marcas m ON m.id = t.marca_id
        WHERE t.modelagem IN ('oversized', 'regular', 'slim')
        ORDER BY m.nome, t.modelagem, t.tamanho_ordem
    """, conn)

    conn.close()
    return df, size_charts


def atribuir_tamanho(df: pd.DataFrame, size_charts: pd.DataFrame,
                     marca: str = 'Hering', modelagem: str = 'regular') -> pd.DataFrame:
    sc = size_charts[
        (size_charts['marca'] == marca) &
        (size_charts['modelagem'] == modelagem)
    ].copy().sort_values('tamanho_ordem')

    resultados = []
    for _, pessoa in df.iterrows():
        busto  = pessoa['busto_circunf']
        altura = pessoa['altura']

        matches = sc[
            (sc['busto_corpo_min'] <= busto) &
            (sc['busto_corpo_max'] >= busto)
        ]

        if altura and len(matches) > 1:
            alt_m = matches[
                (matches['altura_min'] <= altura) &
                (matches['altura_max'] >= altura)
            ]
            if len(alt_m) > 0:
                matches = alt_m

        if len(matches) == 0:
            row = sc.iloc[0] if busto < sc['busto_corpo_min'].min() else sc.iloc[-1]
        else:
            row = matches.iloc[-1]

        resultados.append({
            'id':                             pessoa['id'],
            f'tamanho_Hering_{modelagem}':    row['tamanho_label'],
            f'ordem_Hering_{modelagem}':      int(row['tamanho_ordem']),
        })

    return pd.DataFrame(resultados)


def criar_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['imc'] = (df['peso_kg'] / ((df['altura'] / 100) ** 2)).round(1)
    df['ratio_busto_ombro'] = (df['busto_circunf'] / df['largura_ombro']).round(2)
    df['ratio_busto_cintura'] = np.where(
        df['cintura_circunf'].notna() & (df['cintura_circunf'] > 0),
        (df['busto_circunf'] / df['cintura_circunf']).round(2),
        np.nan
    )
    # Sem genero_num — escopo exclusivamente masculino
    return df


def expandir_por_modelagem(df: pd.DataFrame,
                            size_charts: pd.DataFrame) -> pd.DataFrame:
    """
    Expande o dataset criando uma linha por modelagem para cada pessoa.
    Apenas oversized, regular, slim.
    """
    linhas = []

    for modelagem, mod_num in MODELAGEM_MAP.items():
        df_mod = df.copy()
        df_mod['modelagem']     = modelagem
        df_mod['modelagem_num'] = mod_num

        tamanhos = atribuir_tamanho(df, size_charts, marca='Hering',
                                     modelagem=modelagem)
        col_tam   = f'tamanho_Hering_{modelagem}'
        col_ordem = f'ordem_Hering_{modelagem}'

        df_mod = df_mod.merge(tamanhos, on='id', how='left')
        df_mod = df_mod.rename(columns={
            col_tam:   'tamanho_Hering',
            col_ordem: 'ordem_Hering',
        })
        linhas.append(df_mod)

    return pd.concat(linhas, ignore_index=True)


def gerar_graficos(df: pd.DataFrame):
    print('   Gerando gráficos...')

    # Gráfico 1: Distribuição de tamanhos por modelagem
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    fig.suptitle('Distribuição de Tamanhos por Modelagem — Masculino (Hering)',
                 fontsize=12, fontweight='bold')

    for ax, modelagem in zip(axes, ['oversized', 'regular', 'slim']):
        sub = df[df['modelagem'] == modelagem]['tamanho_Hering'].value_counts()
        ordem = ['PP', 'P', 'M', 'G', 'GG', 'XGG']
        sub = sub.reindex([t for t in ordem if t in sub.index])
        cores = [CORES_TAMANHO.get(t, '#aaa') for t in sub.index]
        ax.bar(sub.index, sub.values, color=cores, edgecolor='white', linewidth=0.8)
        ax.set_title(modelagem.capitalize(), fontsize=11)
        ax.set_xlabel('Tamanho')
        ax.set_ylabel('Nº de indivíduos')
        for i, v in enumerate(sub.values):
            ax.text(i, v + 1, str(v), ha='center', fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, '01_distribuicao_por_modelagem.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

    # Gráfico 2: Busto × Tamanho por modelagem
    fig, ax = plt.subplots(figsize=(10, 6))
    cores_mod = {'regular': '#2E75B6', 'slim': '#c06ab3', 'oversized': '#f0855a'}
    for modelagem in MODELAGEM_MAP:
        sub = df[df['modelagem'] == modelagem]
        ax.scatter(sub['busto_circunf'], sub['ordem_Hering'],
                   label=modelagem, color=cores_mod[modelagem],
                   alpha=0.3, s=12)
    ax.set_xlabel('Circunferência do Busto (cm)')
    ax.set_ylabel('Ordem do Tamanho (1=menor)')
    ax.set_title('Busto × Tamanho por Modelagem — Masculino', fontweight='bold')
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, '02_busto_vs_tamanho_modelagem.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

    # Gráfico 3: Correlação
    cols_corr = ['busto_circunf', 'largura_ombro', 'altura',
                 'peso_kg', 'imc', 'modelagem_num', 'ordem_Hering']
    corr = df[cols_corr].corr()
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm',
                center=0, ax=ax, linewidths=0.5,
                xticklabels=['Busto', 'Ombro', 'Altura', 'Peso',
                              'IMC', 'Modelagem', 'Tamanho(num)'],
                yticklabels=['Busto', 'Ombro', 'Altura', 'Peso',
                              'IMC', 'Modelagem', 'Tamanho(num)'])
    ax.set_title('Correlação entre Variáveis e Tamanho', fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, '03_correlacao.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

    print(f'   ✓ 3 gráficos salvos em docs/plots/')


def preparar_dados():
    print('\n' + '='*60)
    print('  FASE 2B — Preparação dos Dados (Masculino · 3 Modelagens)')
    print('='*60)

    print('\n[1/5] Carregando dados do banco (apenas masculino)...')
    df, size_charts = carregar_dados()
    print(f'      ✓ {len(df)} indivíduos masculinos')
    print(f'      ✓ {len(size_charts)} entradas de size charts (oversized/regular/slim)')

    print('\n[2/5] Criando features derivadas...')
    df = criar_features(df)
    print('      ✓ IMC, ratio busto/ombro, ratio busto/cintura')

    print('\n[3/5] Expandindo dataset por modelagem...')
    df_exp = expandir_por_modelagem(df, size_charts)
    print(f'      ✓ {len(df_exp)} amostras '
          f'({len(df)} pessoas × {len(MODELAGEM_MAP)} modelagens)')

    print('\n[4/5] Gerando gráficos...')
    gerar_graficos(df_exp)

    print('\n[5/5] Salvando dataset de treinamento...')
    path = os.path.join(OUT_DIR, 'dataset_treinamento.csv')
    df_exp.to_csv(path, index=False)
    print(f'      ✓ {path}')
    print(f'      ✓ {len(df_exp)} linhas × {len(df_exp.columns)} colunas')

    print('\n📋 Features disponíveis:')
    print('   Medidas:    busto_circunf, largura_ombro, altura, peso_kg')
    print('   Derivadas:  imc, ratio_busto_ombro')
    print('   Modelagem:  modelagem_num (0=oversized, 1=regular, 2=slim)')
    print('   Label:      tamanho_Hering')
    print('\n✅ Dados prontos para o treino!')
    print('='*60)

    return df_exp


if __name__ == '__main__':
    preparar_dados()
