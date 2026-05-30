"""
FASE 2C — Treinamento do Modelo ML
TCC: Sistema de Recomendação de Tamanho para Vestuário Superior

ESCOPO: apenas masculino.
Modelagens: oversized (0), regular (1), slim (2)
  — alinhado com data.yaml: ['oversized', 'regular', 'slim']
"""

import os
import json
import warnings
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
warnings.filterwarnings('ignore')

from sklearn.model_selection  import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble         import RandomForestClassifier
from sklearn.linear_model     import LogisticRegression
from sklearn.preprocessing    import LabelEncoder, StandardScaler
from sklearn.pipeline         import Pipeline
from sklearn.metrics          import (classification_report,
                                       confusion_matrix,
                                       accuracy_score, f1_score)
from xgboost                  import XGBClassifier

BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = BASE_DIR
PLOTS_DIR = os.path.join(BASE_DIR, '../docs/plots')
os.makedirs(PLOTS_DIR, exist_ok=True)

sns.set_theme(style='whitegrid', font_scale=1.1)

# ── Modelagens — ordem exata do data.yaml ────────────────────
# names: ['oversized', 'regular', 'slim']
MODELAGEM_MAP = {
    'oversized': 0,
    'regular':   1,
    'slim':      2,
}

# ── Features ──────────────────────────────────────────────────
# Sem genero_num — escopo exclusivamente masculino
FEATURES_PRINCIPAIS = [
    'busto_circunf',      # circunferência do busto — feature mais importante
    'largura_ombro',      # distância biacromial
    'altura',             # altura total
    'peso_kg',            # peso corporal
    'imc',                # índice de massa corporal
    'ratio_busto_ombro',  # proporção busto/ombro
    'modelagem_num',      # detectada pelo YOLO: 0=oversized, 1=regular, 2=slim
]

FEATURES_OPCIONAIS = [
    'cintura_circunf',
    'comprimento_braco',
    'ratio_busto_cintura',
]

MARCA_ALVO = 'Hering'


def carregar_dataset():
    path = os.path.join(MODEL_DIR, 'dataset_treinamento.csv')
    if not os.path.exists(path):
        raise FileNotFoundError(
            'Dataset não encontrado. Execute: python model/preparar_dados.py'
        )

    df = pd.read_csv(path)

    if 'tamanho_Hering' not in df.columns:
        raise ValueError("Coluna 'tamanho_Hering' não encontrada.")

    df = df.dropna(subset=['tamanho_Hering'])

    # Garantir apenas masculino (campo genero pode estar presente em dados antigos)
    if 'genero' in df.columns:
        df = df[df['genero'] == 'M'].copy()

    for col in FEATURES_OPCIONAIS:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())

    print(f'   ✓ {len(df)} amostras masculinas')
    print(f'   ✓ Modelagens: {df["modelagem"].unique().tolist()}')
    print(f'   ✓ Distribuição:\n{df["tamanho_Hering"].value_counts().to_string()}')
    return df


def preparar_xy(df: pd.DataFrame):
    features = FEATURES_PRINCIPAIS.copy()
    for f in FEATURES_OPCIONAIS:
        if f in df.columns:
            features.append(f)

    X = df[features].copy()

    # Ordem lógica de tamanhos
    ordem = ['PP', 'P', 'XS', 'S', 'M', 'G', 'L', 'GG', 'XL', 'XGG', 'XXL', 'EGG', 'XG']
    classes = [t for t in ordem if t in df['tamanho_Hering'].values]
    le = LabelEncoder()
    le.classes_ = np.array(classes)
    y = le.transform(df['tamanho_Hering'])

    print(f'\n   Features ({len(features)}): {features}')
    return X, y, le, features


def treinar_comparar(X, y, le):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f'\n   Train: {len(X_train)} | Test: {len(X_test)}')

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    modelos = {
        'Random Forest': Pipeline([
            ('clf', RandomForestClassifier(
                n_estimators=200, max_depth=12,
                min_samples_leaf=3, random_state=42, n_jobs=-1
            ))
        ]),
        'XGBoost': Pipeline([
            ('clf', XGBClassifier(
                n_estimators=200, max_depth=6,
                learning_rate=0.1, subsample=0.8,
                eval_metric='mlogloss',
                random_state=42, verbosity=0
            ))
        ]),
        'Regressão Logística': Pipeline([
            ('scaler', StandardScaler()),
            ('clf', LogisticRegression(max_iter=1000, random_state=42))
        ]),
    }

    resultados = {}
    melhor_nome = None
    melhor_f1   = -1

    print(f"\n   {'Modelo':<25} {'Acurácia CV':<16} {'F1 CV':<13} {'F1 Test'}")
    print('   ' + '-'*65)

    for nome, pipeline in modelos.items():
        cv_acc = cross_val_score(pipeline, X_train, y_train,
                                  cv=cv, scoring='accuracy', n_jobs=-1)
        cv_f1  = cross_val_score(pipeline, X_train, y_train,
                                  cv=cv, scoring='f1_weighted', n_jobs=-1)

        pipeline.fit(X_train, y_train)
        y_pred  = pipeline.predict(X_test)
        f1_test = f1_score(y_test, y_pred, average='weighted')
        acc_test= accuracy_score(y_test, y_pred)

        resultados[nome] = {
            'pipeline':    pipeline,
            'cv_acc':      cv_acc.mean(),
            'cv_acc_std':  cv_acc.std(),
            'cv_f1':       cv_f1.mean(),
            'cv_f1_std':   cv_f1.std(),
            'f1_test':     f1_test,
            'acc_test':    acc_test,
            'y_pred':      y_pred,
        }

        print(f'   {nome:<25} {cv_acc.mean():.3f} ± {cv_acc.std():.3f}   '
              f'{cv_f1.mean():.3f} ± {cv_f1.std():.3f}   {f1_test:.3f}')

        if f1_test > melhor_f1:
            melhor_f1   = f1_test
            melhor_nome = nome

    print(f'\n   🏆 Melhor: {melhor_nome} (F1={melhor_f1:.3f})')
    return modelos, resultados, melhor_nome, X_train, X_test, y_train, y_test


def gerar_graficos(resultados, melhor_nome, X_test, y_test, le, features):
    print('\n   Gerando gráficos de avaliação...')

    # Comparação de modelos
    fig, ax = plt.subplots(figsize=(9, 5))
    nomes = list(resultados.keys())
    f1s   = [resultados[n]['f1_test']  for n in nomes]
    accs  = [resultados[n]['acc_test'] for n in nomes]
    x     = np.arange(len(nomes)); w = 0.35
    b1 = ax.bar(x-w/2, accs, w, label='Acurácia', color='#4e9af1', alpha=0.85)
    b2 = ax.bar(x+w/2, f1s,  w, label='F1-Score',  color='#6cbe6c', alpha=0.85)
    for bar in list(b1)+list(b2):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.005,
                f'{bar.get_height():.3f}', ha='center', fontsize=9)
    ax.set_ylim(0, 1.1); ax.set_xticks(x); ax.set_xticklabels(nomes)
    ax.set_title('Comparação de Modelos — Masculino (ANSUR II)', fontweight='bold')
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, '04_comparacao_modelos.png'), dpi=150)
    plt.close()

    # Matriz de confusão
    y_pred  = resultados[melhor_nome]['y_pred']
    classes = le.classes_
    cm      = confusion_matrix(y_test, y_pred)
    cm_pct  = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100
    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(cm_pct, annot=True, fmt='.1f', cmap='Blues',
                xticklabels=classes, yticklabels=classes, ax=ax)
    ax.set_xlabel('Previsto'); ax.set_ylabel('Real')
    ax.set_title(f'Matriz de Confusão — {melhor_nome} (%)', fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, '05_matriz_confusao.png'), dpi=150)
    plt.close()

    # Importância das features
    rf  = resultados['Random Forest']['pipeline'].named_steps['clf']
    imp = rf.feature_importances_
    idx = np.argsort(imp)[::-1]
    fig, ax = plt.subplots(figsize=(10, 5))
    cores = ['#4e9af1' if i == idx[0] else
             '#f5c242' if features[i] == 'modelagem_num' else
             '#aaaaaa' for i in range(len(features))]
    ax.bar([features[i] for i in idx], imp[idx], color=cores, edgecolor='white')
    ax.set_title('Importância das Features (Random Forest)\nAmarelo = modelagem_num (saída do YOLO)',
                 fontweight='bold')
    ax.tick_params(axis='x', rotation=30)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, '06_importancia_features.png'), dpi=150)
    plt.close()

    print('   ✓ 3 gráficos salvos')


def salvar_modelo(pipeline, le, features, nome_modelo, resultados):
    joblib.dump(pipeline, os.path.join(MODEL_DIR, 'modelo_recomendacao.joblib'))
    joblib.dump(le,       os.path.join(MODEL_DIR, 'label_encoder.joblib'))

    info = {
        'nome_modelo':       nome_modelo,
        'marca_referencia':  MARCA_ALVO,
        'escopo':            'masculino',
        'features':          features,
        'classes':           list(le.classes_),
        'modelagem_map':     MODELAGEM_MAP,
        'modelagens_yolo':   ['oversized', 'regular', 'slim'],
        'metricas': {
            'f1_test':  round(resultados[nome_modelo]['f1_test'],  4),
            'acc_test': round(resultados[nome_modelo]['acc_test'], 4),
            'cv_f1':    round(resultados[nome_modelo]['cv_f1'],    4),
        },
        'features_principais': FEATURES_PRINCIPAIS,
        'features_opcionais':  FEATURES_OPCIONAIS,
    }

    with open(os.path.join(MODEL_DIR, 'modelo_info.json'), 'w',
              encoding='utf-8') as f:
        json.dump(info, f, ensure_ascii=False, indent=2)

    print('\n   ✓ modelo_recomendacao.joblib')
    print('   ✓ label_encoder.joblib')
    print('   ✓ modelo_info.json')


def prever_tamanho(busto_circunf: float, largura_ombro: float,
                   altura: float, peso_kg: float,
                   modelagem: str = 'regular',
                   cintura_circunf: float = None,
                   comprimento_braco: float = None) -> dict:
    """
    Predição de tamanho para cliente MASCULINO.

    Parâmetros:
      busto_circunf  : circunferência do busto/tórax em cm
      largura_ombro  : largura entre ombros em cm
      altura         : altura em cm
      peso_kg        : peso em kg
      modelagem      : 'oversized', 'regular' ou 'slim'
                       (detectada automaticamente pelo YOLO)
    """
    pipeline = joblib.load(os.path.join(MODEL_DIR, 'modelo_recomendacao.joblib'))
    le       = joblib.load(os.path.join(MODEL_DIR, 'label_encoder.joblib'))
    with open(os.path.join(MODEL_DIR, 'modelo_info.json'),
              encoding='utf-8') as f:
        info = json.load(f)

    if modelagem not in MODELAGEM_MAP:
        raise ValueError(
            f"Modelagem '{modelagem}' inválida. Use: {list(MODELAGEM_MAP.keys())}"
        )

    imc      = round(peso_kg / ((altura / 100) ** 2), 1)
    ratio_bo = round(busto_circunf / largura_ombro, 2)
    cin_val  = cintura_circunf or (busto_circunf * 0.9)
    ratio_bc = round(busto_circunf / cin_val, 2)
    bra_val  = comprimento_braco or (altura * 0.49)
    mod_num  = MODELAGEM_MAP[modelagem]

    row = {
        'busto_circunf':      busto_circunf,
        'largura_ombro':      largura_ombro,
        'altura':             altura,
        'peso_kg':            peso_kg,
        'imc':                imc,
        'ratio_busto_ombro':  ratio_bo,
        'modelagem_num':      mod_num,
        'cintura_circunf':    cin_val,
        'comprimento_braco':  bra_val,
        'ratio_busto_cintura':ratio_bc,
    }

    X     = pd.DataFrame([{f: row[f] for f in info['features']}])
    proba = pipeline.predict_proba(X)[0]
    idx   = int(np.argmax(proba))

    top3 = np.argsort(proba)[::-1][:3]
    return {
        'tamanho_recomendado': le.classes_[idx],
        'confianca':           round(float(proba[idx]), 3),
        'modelagem_usada':     modelagem,
        'marca_referencia':    info['marca_referencia'],
        'escopo':              'masculino',
        'alternativas': [
            {'tamanho': le.classes_[i],
             'probabilidade': round(float(proba[i]), 3)}
            for i in top3
        ],
    }


def treinar_modelo():
    print('\n' + '='*60)
    print('  FASE 2C — Treinamento ML (Masculino · oversized/regular/slim)')
    print('='*60)

    print('\n[1/5] Carregando dataset...')
    df = carregar_dataset()

    print('\n[2/5] Preparando features e labels...')
    X, y, le, features = preparar_xy(df)

    print('\n[3/5] Treinando e comparando modelos...')
    modelos, resultados, melhor, X_tr, X_te, y_tr, y_te = \
        treinar_comparar(X, y, le)

    print('\n[4/5] Gerando gráficos...')
    gerar_graficos(resultados, melhor, X_te, y_te, le, features)

    print('\n[5/5] Salvando modelo...')
    salvar_modelo(resultados[melhor]['pipeline'], le,
                  features, melhor, resultados)

    m = resultados[melhor]
    print(f'\n{"="*60}')
    print(f'  RELATÓRIO FINAL — {melhor}')
    print(f'{"="*60}')
    print(f'  Acurácia : {m["acc_test"]:.1%}')
    print(f'  F1-Score : {m["f1_test"]:.1%}')
    print(f'  CV F1    : {m["cv_f1"]:.1%} ± {m["cv_f1_std"]:.1%}')
    print(f'\n  Classification Report:')
    print(classification_report(y_te, m['y_pred'],
                                 target_names=le.classes_,
                                 zero_division=0))

    # Testes ao vivo — sem genero, apenas masculino
    print('='*60)
    print('  TESTES AO VIVO')
    print('='*60)
    for mod in ['regular', 'slim', 'oversized']:
        r = prever_tamanho(94.0, 43.0, 175.0, 78.0, modelagem=mod)
        print(f'\n  busto=94 | ombro=43 | h=175 | p=78 | {mod}')
        print(f'  → {r["tamanho_recomendado"]} (confiança: {r["confianca"]:.1%})')

    print(f'\n✅ Fase 2 concluída!')
    print('='*60)


if __name__ == '__main__':
    treinar_modelo()
