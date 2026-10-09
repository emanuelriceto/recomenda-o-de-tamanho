"""
Mescla dois exports YOLOv8 do Roboflow (ex.: dataset v1 + imagens novas) em
data/deepfashion, pronto para yolo/treinar_yolo.py.

- Remapeia as classes PELO NOME (não pelo índice), então a ordem no data.yaml
  de cada export pode ser diferente.
- Prefixa os arquivos (v1_, novo_) para evitar sobrescrever nomes iguais.
- Mantém os splits de cada export (train com train, valid com valid, test com test).
- Alerta sobre: imagem sem label, label sem imagem, classe desconhecida,
  e a mesma foto original aparecendo em splits diferentes (vazamento).

Uso (na raiz do projeto):
    python yolo/mesclar_datasets.py --fontes data/deepfashion_v1 data/novas_roboflow --saida data/deepfashion
"""
import argparse
import re
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

import yaml

CLASSES = ["oversized", "regular", "slim"]      # ordem do projeto (data.yaml da v1)
SPLITS = ["train", "valid", "test"]
EXT_IMG = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def origem(nome: str) -> str:
    """Nome da foto original: o Roboflow gera '<original>_png.rf.<hash>.jpg'."""
    return re.split(r"\.rf\.", Path(nome).stem)[0]


def ler_classes(fonte: Path) -> list[str]:
    arq = fonte / "data.yaml"
    if not arq.exists():
        sys.exit(f"❌ {arq} não encontrado. A pasta precisa ser um export YOLOv8 descompactado.")
    nomes = yaml.safe_load(arq.read_text(encoding="utf-8"))["names"]
    if isinstance(nomes, dict):
        nomes = [nomes[k] for k in sorted(nomes)]
    nomes = [str(n).strip().lower() for n in nomes]
    desconhecidas = [n for n in nomes if n not in CLASSES]
    if desconhecidas:
        sys.exit(f"❌ {fonte.name}: classes fora do projeto {desconhecidas}. "
                 f"Renomeie/remova no Roboflow (Modify Classes) e exporte de novo.")
    return nomes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fontes", nargs="+", required=True, help="pastas de exports YOLOv8 descompactados")
    ap.add_argument("--saida", default="data/deepfashion")
    ap.add_argument("--prefixos", nargs="+", help="prefixo por fonte (padrão: nome da pasta)")
    args = ap.parse_args()

    fontes = [Path(f) for f in args.fontes]
    prefixos = args.prefixos or [f.name for f in fontes]
    if len(prefixos) != len(fontes):
        sys.exit("❌ --prefixos precisa ter um item por fonte.")
    saida = Path(args.saida)
    if saida.exists() and any(saida.iterdir()):
        sys.exit(f"❌ {saida} já existe e não está vazia. Renomeie-a (backup) antes de mesclar.")
    for f in fontes:
        if f.resolve() == saida.resolve():
            sys.exit("❌ A saída não pode ser uma das fontes.")

    contagem = defaultdict(Counter)       # split -> classe -> nº de caixas
    imagens = Counter()                   # split -> nº de imagens
    origens = defaultdict(set)            # foto original -> splits onde aparece
    problemas = []

    for fonte, prefixo in zip(fontes, prefixos):
        nomes = ler_classes(fonte)
        mapa = {i: CLASSES.index(n) for i, n in enumerate(nomes)}
        print(f"\n📦 {fonte}  (classes: {nomes})")
        for split in SPLITS:
            pasta_img, pasta_lbl = fonte / split / "images", fonte / split / "labels"
            if not pasta_img.exists():
                print(f"   {split:<5} — ausente")
                continue
            dest_img, dest_lbl = saida / split / "images", saida / split / "labels"
            dest_img.mkdir(parents=True, exist_ok=True)
            dest_lbl.mkdir(parents=True, exist_ok=True)
            n = 0
            for img in sorted(pasta_img.iterdir()):
                if img.suffix.lower() not in EXT_IMG:
                    continue
                lbl = pasta_lbl / f"{img.stem}.txt"
                if not lbl.exists():
                    problemas.append(f"sem label: {fonte.name}/{split}/{img.name}")
                    continue
                linhas_novas = []
                for linha in lbl.read_text(encoding="utf-8").splitlines():
                    partes = linha.split()
                    if not partes:
                        continue
                    cls = int(partes[0])
                    if cls not in mapa:
                        problemas.append(f"classe {cls} inválida: {fonte.name}/{split}/{lbl.name}")
                        continue
                    novo = mapa[cls]
                    contagem[split][CLASSES[novo]] += 1
                    linhas_novas.append(" ".join([str(novo)] + partes[1:]))
                if not linhas_novas:
                    problemas.append(f"label vazio: {fonte.name}/{split}/{lbl.name}")
                    continue
                nome = f"{prefixo}_{img.stem}"
                shutil.copy2(img, dest_img / f"{nome}{img.suffix.lower()}")
                (dest_lbl / f"{nome}.txt").write_text("\n".join(linhas_novas) + "\n", encoding="utf-8")
                origens[f"{prefixo}:{origem(img.name)}"].add(split)
                imagens[split] += 1
                n += 1
            print(f"   {split:<5} {n} imagens copiadas")

    (saida / "data.yaml").write_text(yaml.safe_dump({
        "train": "../train/images", "val": "../valid/images", "test": "../test/images",
        "nc": len(CLASSES), "names": CLASSES,
        "fontes": [str(f) for f in fontes],
    }, allow_unicode=True, sort_keys=False), encoding="utf-8")

    print("\n📊 Dataset mesclado (imagens · caixas por classe)")
    print(f"   {'split':<6}{'imgs':>6}" + "".join(f"{c:>11}" for c in CLASSES))
    for s in SPLITS:
        print(f"   {s:<6}{imagens[s]:>6}" + "".join(f"{contagem[s][c]:>11}" for c in CLASSES))
    for s in ("valid", "test"):
        faltam = [c for c in CLASSES if contagem[s][c] == 0]
        if faltam:
            print(f"   ⚠️  {s} sem exemplos de {faltam}: métricas dessa classe não serão medidas.")

    vazamento = [o for o, ss in origens.items() if len(ss) > 1]
    if vazamento:
        print(f"\n⚠️  {len(vazamento)} foto(s) original(is) em mais de um split (vazamento). Ex.: {vazamento[:3]}")
    if problemas:
        print(f"\n⚠️  {len(problemas)} problema(s) ignorado(s). Ex.:")
        for p in problemas[:10]:
            print("   -", p)
    print(f"\n✅ Pronto: {saida}/  →  python yolo/treinar_yolo.py")


if __name__ == "__main__":
    main()
