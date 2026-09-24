"""
Teste de confiabilidade do sistema completo:  foto → YOLOv8 → XGBoost → regra.

Gabarito de cada caso = regra de folga aplicada com a modelagem REAL da foto
(rótulo do Roboflow). Mede-se:
  A) YOLOv8 no conjunto de teste: acurácia, cobertura acima do limiar, confusão
  B) Sistema ponta a ponta: acerto exato e ±1 tamanho, comparado a um
     "YOLO perfeito" (modelagem real) — a diferença é o custo dos erros do YOLO
  C) Resultados por grupo de IMC

Executar na raiz, com venv_yolo ativo:
    python tests/avaliar_confiabilidade.py --pessoas 40
Saídas em tests/resultados/
"""
import argparse
import sys
from pathlib import Path

import pandas as pd
import yaml

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))
from api.config import (DB_PATH, LIMIAR_CONF_YOLO, MARCA_REFERENCIA, MODEL_DIR,  # noqa: E402
                        MODELAGEM_FALLBACK, YOLO_WEIGHTS)
from api.servicos.banco_servico import ServicoBanco  # noqa: E402
from api.servicos.ml_servico import ServicoML  # noqa: E402
from api.servicos.yolo_servico import ServicoYOLO, abrir_imagem  # noqa: E402
from model.regra_folga import escolher_tamanho  # noqa: E402

DATASET_DIR = RAIZ / "data" / "deepfashion"
SAIDA = RAIZ / "tests" / "resultados"


def imagens_de_teste() -> list[dict]:
    with open(DATASET_DIR / "data.yaml", encoding="utf-8") as f:
        nomes = yaml.safe_load(f)["names"]
    itens = []
    for img in sorted((DATASET_DIR / "test" / "images").glob("*")):
        lbl = DATASET_DIR / "test" / "labels" / f"{img.stem}.txt"
        if img.suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp"} or not lbl.exists():
            continue
        linhas = [l.split() for l in lbl.read_text().splitlines() if l.strip()]
        if linhas:
            itens.append({"imagem": img, "real": nomes[int(linhas[0][0])]})
    return itens


def avaliar_yolo(yolo: ServicoYOLO, itens: list[dict]) -> pd.DataFrame:
    linhas = []
    for it in itens:
        det = yolo.detectar(abrir_imagem(it["imagem"].read_bytes()))
        conf = det["confianca"] or 0.0
        usada = det["modelagem"] if det["modelagem"] and conf >= LIMIAR_CONF_YOLO else MODELAGEM_FALLBACK
        linhas.append({"imagem": it["imagem"].name, "real": it["real"],
                       "detectada": det["modelagem"] or "nenhuma", "conf": conf, "usada": usada})
    df = pd.DataFrame(linhas)
    print("\n[A] YOLOv8 — conjunto de teste")
    print(f"    Imagens: {len(df)}")
    print(f"    Acurácia (classe detectada = real): {(df.detectada == df.real).mean():.1%}")
    print(f"    Cobertura (conf ≥ {LIMIAR_CONF_YOLO:.0%}):           {(df.conf >= LIMIAR_CONF_YOLO).mean():.1%}")
    print(f"    Acurácia da modelagem usada (c/ fallback): {(df.usada == df.real).mean():.1%}")
    print("\n    Matriz de confusão (linhas = real):")
    print(pd.crosstab(df.real, df.detectada).to_string())
    return df


def avaliar_ponta_a_ponta(ml, banco, df_yolo, n_pessoas, seed=42) -> pd.DataFrame:
    base = pd.read_csv(MODEL_DIR / "dataset_treinamento.csv").drop_duplicates("id")
    amostra = base.sample(n=min(n_pessoas, len(base)), random_state=seed)
    tabelas = {m: banco.tabela(MARCA_REFERENCIA, m) for m in ("oversized", "regular", "slim")}
    ordem = {c: i for i, c in enumerate(ml.classes)}
    linhas = []
    for _, img in df_yolo.iterrows():
        for _, p in amostra.iterrows():
            medidas = {k: p[k] for k in ("busto_circunf", "largura_ombro", "altura", "peso_kg",
                                         "cintura_circunf", "comprimento_braco")}
            gab = escolher_tamanho(p.busto_circunf, p.cintura_circunf, tabelas[img.real], img.real).tamanho
            sis = ml.prever(medidas, img.usada)["tamanho"]
            ideal = ml.prever(medidas, img.real)["tamanho"]
            linhas.append({"imagem": img.imagem, "pessoa": p.id, "grupo_imc": p.grupo_imc,
                           "real": img.real, "usada": img.usada,
                           "gabarito": gab, "sistema": sis, "yolo_perfeito": ideal})
    df = pd.DataFrame(linhas)
    for col in ("sistema", "yolo_perfeito"):
        df[f"{col}_exato"] = df[col] == df.gabarito
        df[f"{col}_adj"] = [abs(ordem.get(a, 99) - ordem.get(b, -99)) <= 1
                            for a, b in zip(df[col], df.gabarito)]
    print(f"\n[B] Sistema completo — {len(df)} casos ({len(df_yolo)} fotos × {len(amostra)} pessoas)")
    print(f"    {'':<22}{'Exato':>8}{'±1 tam.':>9}")
    print(f"    {'Sistema (YOLO real)':<22}{df.sistema_exato.mean():>8.1%}{df.sistema_adj.mean():>9.1%}")
    print(f"    {'YOLO perfeito':<22}{df.yolo_perfeito_exato.mean():>8.1%}{df.yolo_perfeito_adj.mean():>9.1%}")
    perda = df.yolo_perfeito_exato.mean() - df.sistema_exato.mean()
    print(f"    Perda causada por erros do YOLO: {perda * 100:.1f} p.p.")

    print("\n[C] Por grupo de IMC (sistema completo)")
    g = df.groupby("grupo_imc").agg(casos=("sistema_exato", "size"),
                                    exato=("sistema_exato", "mean"), adj=("sistema_adj", "mean"))
    print(g.to_string(float_format=lambda x: f"{x:.1%}"))
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pessoas", type=int, default=40, help="pessoas do ANSUR por foto")
    args = ap.parse_args()

    yolo = ServicoYOLO(YOLO_WEIGHTS)
    if not yolo.disponivel:
        sys.exit(f"YOLO indisponível: {yolo.erro}")
    ml, banco = ServicoML(MODEL_DIR), ServicoBanco(DB_PATH)
    itens = imagens_de_teste()
    if not itens:
        sys.exit("Nenhuma imagem rotulada em data/deepfashion/test/")

    print("=" * 62 + "\n  TESTE DE CONFIABILIDADE — sistema completo\n" + "=" * 62)
    print(f"  Pesos YOLO: {YOLO_WEIGHTS}\n  Modelo ML: {ml.info.get('nome_modelo')} v{ml.info.get('versao', 1)}")
    df_yolo = avaliar_yolo(yolo, itens)
    df_e2e = avaliar_ponta_a_ponta(ml, banco, df_yolo, args.pessoas)

    SAIDA.mkdir(parents=True, exist_ok=True)
    df_yolo.to_csv(SAIDA / "yolo_teste.csv", index=False)
    df_e2e.to_csv(SAIDA / "ponta_a_ponta.csv", index=False)

    ok = ((df_yolo.usada == df_yolo.real).mean() >= 0.90
          and df_e2e.sistema_exato.mean() >= 0.85 and df_e2e.sistema_adj.mean() >= 0.98)
    print("\n" + "=" * 62)
    print("  Metas internas: YOLO ≥ 90% · exato ≥ 85% · ±1 ≥ 98%")
    print(f"  Resultado: {'✅ ATENDE — pode seguir para perfis extremos' if ok else '⚠️  NÃO ATENDE — revisar antes de avançar'}")
    print(f"  CSVs em {SAIDA.relative_to(RAIZ)}/")
    print("=" * 62)


if __name__ == "__main__":
    main()
