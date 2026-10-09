"""
Converte um export do Roboflow com UMA classe genérica (ex.: 'Camiseta') em um
dataset com as 3 classes do projeto: oversized, regular, slim.

As caixas (bounding boxes) já desenhadas no Roboflow são reaproveitadas; só a
classe muda. A classificação é feita uma vez por FOTO ORIGINAL e vale para todas
as cópias geradas pelo augmentation (mesmo prefixo '<id>_png.rf.<hash>').

Passo 1 — separar uma amostra de cada foto original para classificação visual:
    python yolo/classificar_modelagem.py preparar --fonte data/novas_roboflow --pasta data/classificar

Passo 2 — no Explorador de Arquivos, arraste cada imagem de
    data/classificar/_a_classificar/  para  oversized/, regular/, slim/ ou descartar/

Passo 3 — gerar o dataset com as 3 classes:
    python yolo/classificar_modelagem.py aplicar --fonte data/novas_roboflow --pasta data/classificar --saida data/novas_3classes
"""
import argparse
import re
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

import yaml

CLASSES = ["oversized", "regular", "slim"]   # mesma ordem do data.yaml da v1
DESCARTAR = "descartar"
PENDENTES = "_a_classificar"
SPLITS = ["train", "valid", "test"]
EXT_IMG = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def origem(nome: str) -> str:
    return re.split(r"\.rf\.", Path(nome).stem)[0]


def imagens(fonte: Path):
    for split in SPLITS:
        pasta = fonte / split / "images"
        if pasta.exists():
            for img in sorted(pasta.iterdir()):
                if img.suffix.lower() in EXT_IMG:
                    yield split, img


def preparar(args):
    fonte, pasta = Path(args.fonte), Path(args.pasta)
    if not (fonte / "data.yaml").exists():
        sys.exit(f"❌ {fonte}/data.yaml não encontrado.")
    for sub in CLASSES + [DESCARTAR, PENDENTES]:
        (pasta / sub).mkdir(parents=True, exist_ok=True)
    ja = {p.stem for sub in CLASSES + [DESCARTAR] for p in (pasta / sub).iterdir()}

    # valid/test não têm augmentation: preferir essas cópias (imagem limpa)
    escolhida = {}
    for split, img in imagens(fonte):
        o = origem(img.name)
        if o not in escolhida or (escolhida[o][0] == "train" and split != "train"):
            escolhida[o] = (split, img)
    novas = 0
    for o, (_, img) in sorted(escolhida.items()):
        if o in ja:
            continue
        shutil.copy2(img, pasta / PENDENTES / f"{o}{img.suffix.lower()}")
        novas += 1
    print(f"✅ {len(escolhida)} fotos originais encontradas; {novas} copiadas para {pasta / PENDENTES}")
    print(f"   Arraste cada uma para: {', '.join(CLASSES + [DESCARTAR])}")
    print("   Depois rode o comando 'aplicar'.")


def ler_mapa(pasta: Path) -> dict:
    mapa = {}
    for sub in CLASSES + [DESCARTAR]:
        for p in (pasta / sub).iterdir():
            if p.suffix.lower() in EXT_IMG:
                if p.stem in mapa and mapa[p.stem] != sub:
                    sys.exit(f"❌ {p.stem} está em duas pastas: {mapa[p.stem]} e {sub}")
                mapa[p.stem] = sub
    return mapa


def aplicar(args):
    fonte, pasta, saida = Path(args.fonte), Path(args.pasta), Path(args.saida)
    if saida.exists() and any(saida.iterdir()):
        sys.exit(f"❌ {saida} já existe e não está vazia. Renomeie ou apague antes.")
    mapa = ler_mapa(pasta)
    todas = {origem(img.name) for _, img in imagens(fonte)}
    pendentes = sorted(todas - set(mapa))
    if pendentes and not args.permitir_pendentes:
        sys.exit(f"❌ {len(pendentes)} foto(s) ainda sem classe (ex.: {pendentes[:5]}). "
                 "Classifique todas ou use --permitir-pendentes para ignorá-las.")

    cont = defaultdict(Counter)
    ignoradas = Counter()
    for split, img in imagens(fonte):
        classe = mapa.get(origem(img.name))
        if classe not in CLASSES:
            ignoradas["descartadas" if classe == DESCARTAR else "sem classe"] += 1
            continue
        lbl = fonte / split / "labels" / f"{img.stem}.txt"
        if not lbl.exists():
            ignoradas["sem label"] += 1
            continue
        idx = CLASSES.index(classe)
        linhas = [" ".join([str(idx)] + l.split()[1:])
                  for l in lbl.read_text(encoding="utf-8").splitlines() if l.strip()]
        if not linhas:
            ignoradas["label vazio"] += 1
            continue
        (saida / split / "images").mkdir(parents=True, exist_ok=True)
        (saida / split / "labels").mkdir(parents=True, exist_ok=True)
        shutil.copy2(img, saida / split / "images" / img.name)
        (saida / split / "labels" / f"{img.stem}.txt").write_text("\n".join(linhas) + "\n", encoding="utf-8")
        cont[split][classe] += 1

    (saida / "data.yaml").write_text(yaml.safe_dump({
        "train": "../train/images", "val": "../valid/images", "test": "../test/images",
        "nc": len(CLASSES), "names": CLASSES, "origem": str(fonte)},
        allow_unicode=True, sort_keys=False), encoding="utf-8")

    fotos = Counter(v for v in mapa.values())
    print(f"Fotos originais: " + ", ".join(f"{c}={fotos[c]}" for c in CLASSES + [DESCARTAR]))
    print(f"\n   {'split':<6}" + "".join(f"{c:>11}" for c in CLASSES))
    for s in SPLITS:
        print(f"   {s:<6}" + "".join(f"{cont[s][c]:>11}" for c in CLASSES))
    if ignoradas:
        print(f"\n   Imagens ignoradas: {dict(ignoradas)}")
    print(f"\n✅ Dataset com 3 classes em {saida}/")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p1 = sub.add_parser("preparar")
    p1.add_argument("--fonte", required=True)
    p1.add_argument("--pasta", default="data/classificar")
    p2 = sub.add_parser("aplicar")
    p2.add_argument("--fonte", required=True)
    p2.add_argument("--pasta", default="data/classificar")
    p2.add_argument("--saida", default="data/novas_3classes")
    p2.add_argument("--permitir-pendentes", action="store_true")
    args = ap.parse_args()
    preparar(args) if args.cmd == "preparar" else aplicar(args)


if __name__ == "__main__":
    main()
