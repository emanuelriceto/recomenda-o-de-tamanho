"""
FASE 2C (v2) — Treinamento do modelo de recomendação (XGBoost)
TCC: Sistema de Recomendação de Tamanho para Vestuário Superior (masculino)

MUDANÇAS EM RELAÇÃO À v1
  • Treino/teste e validação cruzada AGRUPADOS POR PESSOA (StratifiedGroupKFold).
    Cada pessoa aparece 3× no dataset (uma por modelagem); na v1 a mesma pessoa
    podia cair no treino e no teste, inflando as métricas.
  • Métrica "acerto adjacente" (±1 tamanho), alinhada ao critério de aceite.
  • Pesos de amostra: balanceamento suave por classe + peso extra para perfis
    extremos (IMC < 16 ou ≥ 40).
  • Relatório por grupo de IMC e teste de sensibilidade à modelagem.
  • Ordem das classes lida do dataset (ordem_Hering): novas numerações
    (ex.: plus size) entram sem alterar código.

Execução:  python model/treinar_modelo.py
"""
import json
import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))
from model.regra_folga import FOLGA  # noqa: E402

MODEL_DIR = RAIZ / "model"
PLOTS_DIR = RAIZ / "docs" / "plots"
DATASET = MODEL_DIR / "dataset_treinamento.csv"

MODELAGEM_MAP = {"oversized": 0, "regular": 1, "slim": 2}
FEATURES_PRINCIPAIS = ["busto_circunf", "largura_ombro", "altura", "peso_kg",
                       "imc", "ratio_busto_ombro", "modelagem_num"]
FEATURES_OPCIONAIS = ["cintura_circunf", "comprimento_braco", "ratio_busto_cintura"]
GRUPOS = ["magreza_extrema", "magreza", "eutrofia", "sobrepeso",
          "obesidade_I_II", "obesidade_extrema"]
GRUPOS_EXTREMOS = {"magreza_extrema", "obesidade_extrema"}
PESO_EXTREMOS = 2.0
MIN_AMOSTRAS_CLASSE = 30
MODELO_PREFERIDO = "XGBoost"   # mantido se estiver a < 0,5 p.p. do melhor F1
SEED = 42


# ── Dados ─────────────────────────────────────────────────────
def carregar() -> pd.DataFrame:
    if not DATASET.exists():
        sys.exit("dataset_treinamento.csv não encontrado. Execute: python model/preparar_dados.py")
    df = pd.read_csv(DATASET).dropna(subset=["tamanho_Hering", "ordem_Hering"])
    for c in FEATURES_OPCIONAIS:
        if c in df.columns:
            df[c] = df[c].fillna(df[c].median())
    cont = df["tamanho_Hering"].value_counts()
    raras = cont[cont < MIN_AMOSTRAS_CLASSE].index.tolist()
    if raras:
        print(f"   ⚠️  Classes com < {MIN_AMOSTRAS_CLASSE} amostras removidas: {raras}")
        df = df[~df["tamanho_Hering"].isin(raras)]
    return df.reset_index(drop=True)


def preparar_xy(df: pd.DataFrame):
    features = FEATURES_PRINCIPAIS + [c for c in FEATURES_OPCIONAIS if c in df.columns]
    ordem = (df.groupby("tamanho_Hering")["ordem_Hering"].min()
               .sort_values().index.tolist())
    le = LabelEncoder()
    le.classes_ = np.array(ordem, dtype=object)
    y = le.transform(df["tamanho_Hering"])
    return df[features].astype(float), y, le, features


def calcular_pesos(y, grupos) -> np.ndarray:
    w = np.sqrt(compute_sample_weight("balanced", y))          # balanceamento suave
    w *= np.where(np.isin(grupos, list(GRUPOS_EXTREMOS)), PESO_EXTREMOS, 1.0)
    return w / w.mean()


# ── Modelos e métricas ────────────────────────────────────────
def criar_modelos() -> dict:
    return {
        "XGBoost": Pipeline([("clf", XGBClassifier(
            n_estimators=300, max_depth=5, learning_rate=0.08,
            subsample=0.9, colsample_bytree=0.9, objective="multi:softprob",
            eval_metric="mlogloss", random_state=SEED, n_jobs=-1, verbosity=0))]),
        "Random Forest": Pipeline([("clf", RandomForestClassifier(
            n_estimators=300, min_samples_leaf=3, random_state=SEED, n_jobs=-1))]),
        "Regressão Logística": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=3000))]),
    }


def metricas(y_true, y_pred) -> dict:
    return {
        "acuracia": float(accuracy_score(y_true, y_pred)),
        "f1_ponderado": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "acerto_adjacente": float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred)) <= 1)),
    }


def ajustar(modelo, X, y, w):
    m = clone(modelo)
    m.fit(X, y, clf__sample_weight=w)
    return m


def validar_cruzado(modelo, X, y, w, pessoas, n_classes) -> dict:
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    res = []
    for tr, va in cv.split(X, y, pessoas):
        if len(np.unique(y[tr])) < n_classes:   # XGBoost exige todas as classes no treino
            continue
        m = ajustar(modelo, X.iloc[tr], y[tr], w[tr])
        res.append(metricas(y[va], m.predict(X.iloc[va])))
    return {k: (float(np.mean([r[k] for r in res])), float(np.std([r[k] for r in res])))
            for k in res[0]}


# ── Relatórios ────────────────────────────────────────────────
def relatorio_grupos(df_te, y_te, y_pred) -> dict:
    out = {}
    print(f"\n   {'Grupo IMC':<18} {'n':>6} {'Exato':>8} {'±1 tam.':>8}")
    for g in GRUPOS:
        mask = (df_te["grupo_imc"] == g).values
        if mask.sum() == 0:
            continue
        m = metricas(y_te[mask], y_pred[mask])
        out[g] = {"n": int(mask.sum()), **m}
        print(f"   {g:<18} {mask.sum():>6} {m['acuracia']:>8.1%} {m['acerto_adjacente']:>8.1%}")
    return out


def sensibilidade_modelagem(df_te, y_pred) -> float:
    """% de pessoas cujo tamanho previsto muda conforme a modelagem."""
    piv = (pd.DataFrame({"id": df_te["id"].values, "mod": df_te["modelagem"].values,
                         "pred": y_pred})
             .pivot_table(index="id", columns="mod", values="pred"))
    return float((piv.nunique(axis=1) > 1).mean())


def gerar_graficos(resultados, melhor, y_te, y_pred, le):
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    nomes = list(resultados)
    exato = [resultados[n]["teste"]["acuracia"] for n in nomes]
    adj = [resultados[n]["teste"]["acerto_adjacente"] for n in nomes]
    x = np.arange(len(nomes))
    fig, ax = plt.subplots(figsize=(8, 4.5))
    b1 = ax.bar(x - 0.18, exato, 0.36, label="Acerto exato", color="#2E75B6")
    b2 = ax.bar(x + 0.18, adj, 0.36, label="Acerto ±1 tamanho", color="#02C39A")
    for b in list(b1) + list(b2):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.005,
                f"{b.get_height():.1%}", ha="center", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(nomes)
    ax.set_ylim(0, 1.1)
    ax.set_title("Modelos — teste agrupado por pessoa")
    ax.legend()
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "04_comparacao_modelos_v2.png", dpi=150)
    plt.close()

    cm = confusion_matrix(y_te, y_pred, labels=range(len(le.classes_)))
    cm_pct = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1) * 100
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm_pct, cmap="Blues", vmin=0, vmax=100)
    ax.set_xticks(range(len(le.classes_)))
    ax.set_xticklabels(le.classes_, rotation=30)
    ax.set_yticks(range(len(le.classes_)))
    ax.set_yticklabels(le.classes_)
    for i in range(cm_pct.shape[0]):
        for j in range(cm_pct.shape[1]):
            ax.text(j, i, f"{cm_pct[i, j]:.0f}", ha="center", va="center",
                    color="white" if cm_pct[i, j] > 50 else "black", fontsize=8)
    ax.set_xlabel("Previsto")
    ax.set_ylabel("Real (regra de folga)")
    ax.set_title(f"Matriz de confusão (%) — {melhor}")
    fig.colorbar(im)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "05_matriz_confusao_v2.png", dpi=150)
    plt.close()


def _json(o):
    if isinstance(o, dict):
        return {k: _json(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_json(v) for v in o]
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    return o


# ── Pipeline principal ────────────────────────────────────────
def treinar_modelo():
    print("=" * 62)
    print("  FASE 2C (v2) — Treinamento · divisão agrupada por pessoa")
    print("=" * 62)

    df = carregar()
    X, y, le, features = preparar_xy(df)
    pessoas = df["id"].values
    grupos = df["grupo_imc"].values
    w = calcular_pesos(y, grupos)
    n_cls = len(le.classes_)
    print(f"\n[1/5] {len(df)} amostras · {df['id'].nunique()} pessoas · classes: {list(le.classes_)}")
    print(f"      Features ({len(features)}): {features}")

    # Holdout de ~20% das PESSOAS
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    tr, te = next(sgkf.split(X, y, pessoas))
    assert not set(pessoas[tr]) & set(pessoas[te]), "vazamento de pessoas entre treino e teste"
    print(f"[2/5] Treino: {len(tr)} amostras | Teste: {len(te)} amostras (pessoas distintas)")

    print("\n[3/5] Comparando modelos (CV 5-fold agrupado + teste)")
    print(f"   {'Modelo':<22} {'F1 CV':>14} {'Exato teste':>12} {'±1 teste':>9}")
    resultados = {}
    for nome, modelo in criar_modelos().items():
        cv = validar_cruzado(modelo, X.iloc[tr], y[tr], w[tr], pessoas[tr], n_cls)
        m = ajustar(modelo, X.iloc[tr], y[tr], w[tr])
        pred = m.predict(X.iloc[te])
        resultados[nome] = {"cv": cv, "teste": metricas(y[te], pred), "pred": pred}
        t = resultados[nome]["teste"]
        print(f"   {nome:<22} {cv['f1_ponderado'][0]:>7.3f} ±{cv['f1_ponderado'][1]:.3f}"
              f" {t['acuracia']:>12.1%} {t['acerto_adjacente']:>9.1%}")

    melhor_f1 = max(r["teste"]["f1_ponderado"] for r in resultados.values())
    melhor = max(resultados, key=lambda n: resultados[n]["teste"]["f1_ponderado"])
    if melhor_f1 - resultados[MODELO_PREFERIDO]["teste"]["f1_ponderado"] < 0.005:
        melhor = MODELO_PREFERIDO
    pred_te = resultados[melhor]["pred"]
    print(f"\n   🏆 Selecionado: {melhor}")

    print("\n[4/5] Análises no conjunto de teste")
    df_te = df.iloc[te]
    por_grupo = relatorio_grupos(df_te, y[te], pred_te)
    sens = sensibilidade_modelagem(df_te, pred_te)
    print(f"\n   Modelagem altera o tamanho previsto de {sens:.1%} das pessoas de teste")
    print("\n" + classification_report(y[te], pred_te, labels=range(n_cls),
                                       target_names=list(le.classes_), zero_division=0))
    gerar_graficos(resultados, melhor, y[te], pred_te, le)

    print("[5/5] Treinando modelo final com todos os dados e salvando...")
    final = ajustar(criar_modelos()[melhor], X, y, w)
    joblib.dump(final, MODEL_DIR / "modelo_recomendacao.joblib")
    joblib.dump(le, MODEL_DIR / "label_encoder.joblib")
    info = {
        "versao": 2,
        "nome_modelo": melhor,
        "marca_referencia": "Hering",
        "escopo": "masculino",
        "features": features,
        "classes": list(le.classes_),
        "modelagem_map": MODELAGEM_MAP,
        "regra_folga": FOLGA,
        "n_amostras": len(df),
        "n_pessoas": int(df["id"].nunique()),
        "fontes": df.drop_duplicates("id")["fonte_dataset"].value_counts().to_dict(),
        "metricas_teste": resultados[melhor]["teste"],
        "metricas_cv": resultados[melhor]["cv"],
        "metricas_por_grupo": por_grupo,
        "sensibilidade_modelagem": sens,
    }
    with open(MODEL_DIR / "modelo_info.json", "w", encoding="utf-8") as f:
        json.dump(_json(info), f, ensure_ascii=False, indent=2)
    print("   ✓ modelo_recomendacao.joblib · label_encoder.joblib · modelo_info.json")

    testes_ao_vivo(final, le, features)
    print("\n✅ Treinamento concluído.")


def testes_ao_vivo(modelo, le, features):
    casos = [
        ("Médio (IMC 25)",        dict(busto_circunf=94,  largura_ombro=43, altura=175, peso_kg=78,  cintura_circunf=84)),
        ("Fronteira M/G",         dict(busto_circunf=97,  largura_ombro=44, altura=176, peso_kg=82,  cintura_circunf=88)),
        ("Magreza extrema (IMC 15)", dict(busto_circunf=78, largura_ombro=39, altura=175, peso_kg=46, cintura_circunf=64)),
        ("Obesidade III (IMC 45)",   dict(busto_circunf=138, largura_ombro=48, altura=175, peso_kg=138, cintura_circunf=145)),
    ]
    print("\n   TESTES AO VIVO (tamanho previsto por modelagem)")
    for nome, m in casos:
        saida = []
        for mod, num in MODELAGEM_MAP.items():
            linha = dict(m)
            linha["imc"] = m["peso_kg"] / (m["altura"] / 100) ** 2
            linha["ratio_busto_ombro"] = m["busto_circunf"] / m["largura_ombro"]
            linha["ratio_busto_cintura"] = m["busto_circunf"] / m["cintura_circunf"]
            linha["comprimento_braco"] = m["altura"] * 0.49
            linha["modelagem_num"] = num
            X = pd.DataFrame([[linha[f] for f in features]], columns=features)
            p = modelo.predict_proba(X)[0]
            saida.append(f"{mod}={le.classes_[p.argmax()]} ({p.max():.0%})")
        print(f"   {nome:<26} " + " · ".join(saida))


if __name__ == "__main__":
    treinar_modelo()
