"""
FASE 2C — Treinamento, Avaliação e Exportação do Modelo ML
TCC: Sistema de Recomendação de Tamanho para Vestuário Superior

Modelos treinados:
  1. Random Forest  — robusto, interpretável, bom baseline
  2. XGBoost        — geralmente o mais preciso para dados tabulares
  3. Regressão Logística — simples, para comparação

O melhor modelo é salvo em model/modelo_recomendacao.joblib
Um arquivo de metadados model/modelo_info.json é gerado junto.
"""

import os
import json
import warnings
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
warnings.filterwarnings("ignore")

from sklearn.model_selection    import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble           import RandomForestClassifier
from sklearn.linear_model       import LogisticRegression
from sklearn.preprocessing      import LabelEncoder, StandardScaler
from sklearn.pipeline           import Pipeline
from sklearn.metrics            import (classification_report, confusion_matrix,
                                         accuracy_score, f1_score)
from xgboost                    import XGBClassifier

# ── Caminhos ────────────────────────────────────────────────
BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = BASE_DIR
PLOTS_DIR = os.path.join(BASE_DIR, "../docs/plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

sns.set_theme(style="whitegrid", font_scale=1.1)

# ── Features usadas no treinamento ──────────────────────────
# Estas são as medidas que o cliente vai informar no app.
# Mantenha esta lista IGUAL no treinamento e na API (Fase 4).
FEATURES_PRINCIPAIS = [
    "busto_circunf",    # circunferência do busto/tórax — mais importante
    "largura_ombro",    # largura biacromial
    "altura",           # altura total
    "peso_kg",          # peso corporal
    "imc",              # índice de massa corporal (derivado)
    "ratio_busto_ombro",# proporção busto/ombro (derivado)
    "genero_num",       # 1=Masculino, 0=Feminino
]

# Features opcionais (nem todo cliente vai ter)
FEATURES_OPCIONAIS = [
    "cintura_circunf",
    "comprimento_braco",
    "ratio_busto_cintura",
]

MARCA_ALVO = "Hering"  # marca de referência para o label principal


# ═══════════════════════════════════════════════════════════
# 1. CARREGAR E PREPARAR DADOS
# ═══════════════════════════════════════════════════════════

def carregar_dataset():
    dataset_path = os.path.join(MODEL_DIR, "dataset_treinamento.csv")
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(
            "Dataset não encontrado. Execute primeiro: python model/preparar_dados.py"
        )

    df = pd.read_csv(dataset_path)
    label_col = f"tamanho_{MARCA_ALVO}"

    if label_col not in df.columns:
        raise ValueError(f"Coluna '{label_col}' não encontrada. Verifique MARCA_ALVO.")

    # Remover linhas sem label
    df = df.dropna(subset=[label_col])

    # Preencher features opcionais faltantes com a mediana
    for col in FEATURES_OPCIONAIS:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())

    print(f"   ✓ Dataset: {len(df)} amostras")
    print(f"   ✓ Label: {label_col}")
    print(f"   ✓ Distribuição:\n{df[label_col].value_counts().to_string()}")

    return df, label_col


# ═══════════════════════════════════════════════════════════
# 2. PREPARAR X e y
# ═══════════════════════════════════════════════════════════

def preparar_xy(df: pd.DataFrame, label_col: str):
    # Features disponíveis (combina principais + opcionais que existem)
    features_usar = FEATURES_PRINCIPAIS.copy()
    for f in FEATURES_OPCIONAIS:
        if f in df.columns:
            features_usar.append(f)

    X = df[features_usar].copy()
    y_raw = df[label_col].copy()

    # Codificar label como número (o modelo precisa de int)
    le = LabelEncoder()
    # Forçar ordem lógica de tamanhos
    ordem_tamanhos = ["PP", "P", "XS", "S", "M", "G", "L", "GG", "XL", "XGG", "XXL", "EGG", "XG"]
    classes_presentes = [t for t in ordem_tamanhos if t in y_raw.values]
    le.classes_ = np.array(classes_presentes)
    y = le.transform(y_raw)

    print(f"\n   Features usadas ({len(features_usar)}):")
    for f in features_usar:
        nulos = X[f].isna().sum()
        print(f"     • {f:<25} ({nulos} nulos)")

    return X, y, le, features_usar


# ═══════════════════════════════════════════════════════════
# 3. TREINAR MODELOS E COMPARAR
# ═══════════════════════════════════════════════════════════

def treinar_comparar(X: pd.DataFrame, y: np.ndarray, le: LabelEncoder):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"\n   Train: {len(X_train)} | Test: {len(X_test)}")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # ── Definir modelos ──────────────────────────────────────
    modelos = {
        "Random Forest": Pipeline([
            ("clf", RandomForestClassifier(
                n_estimators=200, max_depth=12,
                min_samples_leaf=3, random_state=42, n_jobs=-1
            ))
        ]),
        "XGBoost": Pipeline([
            ("clf", XGBClassifier(
                n_estimators=200, max_depth=6,
                learning_rate=0.1, subsample=0.8,
                use_label_encoder=False,
                eval_metric="mlogloss",
                random_state=42, verbosity=0
            ))
        ]),
        "Regressão Logística": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                max_iter=1000, random_state=42, C=1.0
            ))
        ]),
    }

    resultados = {}
    melhor_nome = None
    melhor_f1 = -1

    print("\n   Treinando e avaliando modelos (cross-validation 5-fold):")
    print(f"   {'Modelo':<25} {'Acurácia CV':<15} {'F1 CV':<12} {'F1 Test'}")
    print("   " + "-"*62)

    for nome, pipeline in modelos.items():
        # Cross-validation no conjunto de treino
        cv_acc = cross_val_score(pipeline, X_train, y_train, cv=cv,
                                  scoring="accuracy", n_jobs=-1)
        cv_f1  = cross_val_score(pipeline, X_train, y_train, cv=cv,
                                  scoring="f1_weighted", n_jobs=-1)

        # Treinar no treino completo e avaliar no teste
        pipeline.fit(X_train, y_train)
        y_pred   = pipeline.predict(X_test)
        f1_test  = f1_score(y_test, y_pred, average="weighted")
        acc_test = accuracy_score(y_test, y_pred)

        resultados[nome] = {
            "pipeline":   pipeline,
            "cv_acc":     cv_acc.mean(),
            "cv_acc_std": cv_acc.std(),
            "cv_f1":      cv_f1.mean(),
            "cv_f1_std":  cv_f1.std(),
            "f1_test":    f1_test,
            "acc_test":   acc_test,
            "y_pred":     y_pred,
        }

        print(f"   {nome:<25} {cv_acc.mean():.3f} ± {cv_acc.std():.3f}   "
              f"{cv_f1.mean():.3f} ± {cv_f1.std():.3f}   {f1_test:.3f}")

        if f1_test > melhor_f1:
            melhor_f1   = f1_test
            melhor_nome = nome

    print(f"\n   🏆 Melhor modelo: {melhor_nome} (F1 test = {melhor_f1:.3f})")
    return modelos, resultados, melhor_nome, X_train, X_test, y_train, y_test


# ═══════════════════════════════════════════════════════════
# 4. GRÁFICOS DE AVALIAÇÃO
# ═══════════════════════════════════════════════════════════

def gerar_graficos_avaliacao(resultados: dict, melhor_nome: str,
                              X_test, y_test, le: LabelEncoder, features: list):

    print("\n   Gerando gráficos de avaliação...")

    # ── Gráfico 5: Comparação de modelos ────────────────────
    fig, ax = plt.subplots(figsize=(9, 5))
    nomes  = list(resultados.keys())
    f1s    = [resultados[n]["f1_test"] for n in nomes]
    accs   = [resultados[n]["acc_test"] for n in nomes]
    x      = np.arange(len(nomes))
    largura = 0.35

    bars1 = ax.bar(x - largura/2, accs, largura, label="Acurácia", color="#4e9af1", alpha=0.85)
    bars2 = ax.bar(x + largura/2, f1s,  largura, label="F1-Score",  color="#6cbe6c", alpha=0.85)

    for bar in bars1 + bars2:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=9)

    ax.set_ylim(0, 1.1)
    ax.set_xticks(x)
    ax.set_xticklabels(nomes, fontsize=10)
    ax.set_ylabel("Score")
    ax.set_title("Comparação de Modelos — Conjunto de Teste", fontsize=13, fontweight="bold")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "05_comparacao_modelos.png"), dpi=150)
    plt.close()

    # ── Gráfico 6: Matriz de confusão do melhor modelo ──────
    melhor_pipeline = resultados[melhor_nome]["pipeline"]
    y_pred          = resultados[melhor_nome]["y_pred"]
    classes         = le.classes_

    cm = confusion_matrix(y_test, y_pred)
    cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm_pct, annot=True, fmt=".1f", cmap="Blues",
                xticklabels=classes, yticklabels=classes, ax=ax,
                linewidths=0.5, cbar_kws={"label": "%"})
    ax.set_xlabel("Tamanho Previsto")
    ax.set_ylabel("Tamanho Real")
    ax.set_title(f"Matriz de Confusão — {melhor_nome} (%)\n"
                 f"(cada linha soma 100% — diagonal = acertos)",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "06_matriz_confusao.png"), dpi=150)
    plt.close()

    # ── Gráfico 7: Importância das features (Random Forest) ─
    rf_pipeline = resultados["Random Forest"]["pipeline"]
    importancias = rf_pipeline.named_steps["clf"].feature_importances_

    fig, ax = plt.subplots(figsize=(9, 5))
    idx = np.argsort(importancias)[::-1]
    cores = ["#4e9af1" if i == idx[0] else "#aaaaaa" for i in range(len(features))]
    ax.bar([features[i] for i in idx], importancias[idx], color=cores, edgecolor="white")
    ax.set_xlabel("Feature")
    ax.set_ylabel("Importância (Gini)")
    ax.set_title("Importância das Features — Random Forest",
                 fontsize=13, fontweight="bold")
    ax.tick_params(axis="x", rotation=30)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "07_importancia_features.png"), dpi=150)
    plt.close()

    print(f"   ✓ 3 gráficos salvos em docs/plots/")


# ═══════════════════════════════════════════════════════════
# 5. SALVAR MODELO E METADADOS
# ═══════════════════════════════════════════════════════════

def salvar_modelo(pipeline, le: LabelEncoder, features: list,
                   nome_modelo: str, resultados: dict):

    # Salvar pipeline completo
    model_path = os.path.join(MODEL_DIR, "modelo_recomendacao.joblib")
    encoder_path = os.path.join(MODEL_DIR, "label_encoder.joblib")
    joblib.dump(pipeline, model_path)
    joblib.dump(le, encoder_path)

    # Salvar metadados em JSON (usados pela API na Fase 4)
    info = {
        "nome_modelo":    nome_modelo,
        "marca_referencia": MARCA_ALVO,
        "features":       features,
        "classes":        list(le.classes_),
        "metricas": {
            "f1_test":  round(resultados[nome_modelo]["f1_test"], 4),
            "acc_test": round(resultados[nome_modelo]["acc_test"], 4),
            "cv_f1":    round(resultados[nome_modelo]["cv_f1"], 4),
        },
        "features_principais": FEATURES_PRINCIPAIS,
        "features_opcionais":  FEATURES_OPCIONAIS,
    }

    info_path = os.path.join(MODEL_DIR, "modelo_info.json")
    with open(info_path, "w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=2)

    print(f"\n   ✓ Modelo salvo:    model/modelo_recomendacao.joblib")
    print(f"   ✓ Encoder salvo:   model/label_encoder.joblib")
    print(f"   ✓ Metadados:       model/modelo_info.json")
    return model_path, info_path


# ═══════════════════════════════════════════════════════════
# 6. FUNÇÃO DE PREDIÇÃO (usada também pela API)
# ═══════════════════════════════════════════════════════════

def prever_tamanho(busto_circunf: float, largura_ombro: float,
                   altura: float, peso_kg: float, genero: str,
                   cintura_circunf: float = None,
                   comprimento_braco: float = None) -> dict:
    """
    Faz a predição de tamanho dado as medidas do cliente.
    Retorna o tamanho recomendado e a probabilidade de confiança.

    Parâmetros:
      busto_circunf    : circunferência do busto/tórax em cm
      largura_ombro    : largura entre os ombros em cm
      altura           : altura em cm
      peso_kg          : peso em kg
      genero           : 'M' (masculino) ou 'F' (feminino)
      cintura_circunf  : opcional, circunferência da cintura em cm
      comprimento_braco: opcional, comprimento do braço em cm
    """
    model_path   = os.path.join(MODEL_DIR, "modelo_recomendacao.joblib")
    encoder_path = os.path.join(MODEL_DIR, "label_encoder.joblib")
    info_path    = os.path.join(MODEL_DIR, "modelo_info.json")

    if not os.path.exists(model_path):
        raise FileNotFoundError("Modelo não encontrado. Execute treinar_modelo() primeiro.")

    pipeline = joblib.load(model_path)
    le       = joblib.load(encoder_path)

    with open(info_path, encoding="utf-8") as f:
        info = json.load(f)

    # Montar vetor de features na mesma ordem do treinamento
    imc              = round(peso_kg / ((altura / 100) ** 2), 1)
    ratio_bo         = round(busto_circunf / largura_ombro, 2)
    cintura_val      = cintura_circunf or (busto_circunf * 0.9)
    ratio_bc         = round(busto_circunf / cintura_val, 2)
    braco_val        = comprimento_braco or (altura * 0.49)

    row = {
        "busto_circunf":     busto_circunf,
        "largura_ombro":     largura_ombro,
        "altura":            altura,
        "peso_kg":           peso_kg,
        "imc":               imc,
        "ratio_busto_ombro": ratio_bo,
        "genero_num":        1 if genero.upper() == "M" else 0,
        "cintura_circunf":   cintura_val,
        "comprimento_braco": braco_val,
        "ratio_busto_cintura": ratio_bc,
    }

    # Filtrar apenas as features que o modelo conhece
    X = pd.DataFrame([{f: row[f] for f in info["features"]}])

    proba     = pipeline.predict_proba(X)[0]
    pred_idx  = int(np.argmax(proba))
    tamanho   = le.classes_[pred_idx]
    confianca = float(proba[pred_idx])

    # Top-3 alternativas
    top3_idx = np.argsort(proba)[::-1][:3]
    alternativas = [
        {"tamanho": le.classes_[i], "probabilidade": round(float(proba[i]), 3)}
        for i in top3_idx
    ]

    return {
        "tamanho_recomendado": tamanho,
        "confianca":           round(confianca, 3),
        "marca_referencia":    info["marca_referencia"],
        "alternativas":        alternativas,
        "medidas_usadas": {
            "busto": busto_circunf, "ombro": largura_ombro,
            "altura": altura, "imc": imc
        }
    }


# ═══════════════════════════════════════════════════════════
# MAIN — executa tudo
# ═══════════════════════════════════════════════════════════

def treinar_modelo():
    print("\n" + "="*60)
    print("  FASE 2C — Treinamento do Modelo ML")
    print("="*60)

    print("\n[1/5] Carregando dataset...")
    df, label_col = carregar_dataset()

    print("\n[2/5] Preparando features e labels...")
    X, y, le, features = preparar_xy(df, label_col)

    print("\n[3/5] Treinando e comparando modelos...")
    modelos, resultados, melhor_nome, X_train, X_test, y_train, y_test = \
        treinar_comparar(X, y, le)

    print("\n[4/5] Gerando gráficos de avaliação...")
    gerar_graficos_avaliacao(resultados, melhor_nome, X_test, y_test, le, features)

    print("\n[5/5] Salvando melhor modelo...")
    salvar_modelo(resultados[melhor_nome]["pipeline"], le, features,
                   melhor_nome, resultados)

    # ── Relatório final ─────────────────────────────────────
    melhor = resultados[melhor_nome]
    print("\n" + "="*60)
    print(f"  RELATÓRIO FINAL — {melhor_nome}")
    print("="*60)
    print(f"  Acurácia no teste : {melhor['acc_test']:.1%}")
    print(f"  F1-Score no teste : {melhor['f1_test']:.1%}")
    print(f"  Acurácia CV (5-fold): {melhor['cv_acc']:.1%} ± {melhor['cv_acc_std']:.1%}")
    print("\n  Classification Report (teste):")
    y_pred = melhor["y_pred"]
    print(classification_report(y_test, y_pred,
                                 target_names=le.classes_,
                                 zero_division=0))

    # ── Teste de predição ao vivo ────────────────────────────
    print("="*60)
    print("  TESTE AO VIVO — Predição para cliente fictício")
    print("="*60)
    resultado = prever_tamanho(
        busto_circunf=94.0, largura_ombro=43.0,
        altura=175.0, peso_kg=78.0, genero="M"
    )
    print(f"\n  Input:  busto=94cm | ombro=43cm | altura=175cm | peso=78kg | M")
    print(f"  ➜ Tamanho recomendado : {resultado['tamanho_recomendado']}")
    print(f"  ➜ Confiança           : {resultado['confianca']:.1%}")
    print(f"  ➜ Marca de referência : {resultado['marca_referencia']}")
    print(f"  ➜ Alternativas:")
    for alt in resultado["alternativas"]:
        print(f"      {alt['tamanho']:<5} → {alt['probabilidade']:.1%}")

    print("\n✅ Fase 2 concluída! Modelo pronto para integração na API (Fase 4).")
    print("="*60)


if __name__ == "__main__":
    treinar_modelo()
