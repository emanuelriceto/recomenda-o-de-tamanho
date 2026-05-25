"""
FASE 3C — Treinamento do YOLOv8 para Detecção de Modelagem de Camisetas
TCC: Sistema de Recomendação de Tamanho para Vestuário Superior

ATUALIZAÇÃO: 5 classes definitivas de modelagem de camiseta manga curta:
  0 → regular
  1 → slim
  2 → oversized
  3 → longline
  4 → henley

Dataset: coletado manualmente nos sites de e-commerce
  (Hering, Renner, Zara, C&A, Reserva)
  ~400-500 imagens de produto (fundo branco, sem corpo)
  Anotado no Roboflow e exportado no formato YOLOv8

Execute APÓS fazer upload e anotação no Roboflow.
"""

import os
import yaml
from pathlib import Path
from ultralytics import YOLO
import torch

BASE_DIR    = Path(__file__).parent.parent
DATASET_DIR = BASE_DIR / "data" / "deepfashion"
YAML_PATH   = DATASET_DIR / "data.yaml"
YAML_FIXED  = DATASET_DIR / "data_fixed.yaml"
RUNS_DIR    = BASE_DIR / "yolo" / "runs"

# ── Classes definitivas do TCC ────────────────────────────────
CLASSES_MODELAGEM = [
    "regular",    # 0 — caimento padrão
    "slim",       # 1 — corte mais justo
    "oversized",  # 2 — propositalmente largo
    "longline",   # 3 — comprimento estendido
    "henley",     # 4 — gola com botões
]


def verificar_dataset():
    """Verifica se o dataset está no lugar certo e bem estruturado."""
    if not DATASET_DIR.exists():
        print(f"\n  ❌ Pasta do dataset não encontrada: {DATASET_DIR}")
        print("     Faça o download do Roboflow primeiro:")
        print("     python -c \"from roboflow import Roboflow; ...\"")
        return False

    for split in ["train", "valid", "test"]:
        img_dir = DATASET_DIR / split / "images"
        lbl_dir = DATASET_DIR / split / "labels"
        if not img_dir.exists() or not lbl_dir.exists():
            print(f"  ❌ Split '{split}' incompleto")
            return False
        n_imgs = len(list(img_dir.glob("*.jpg")))
        print(f"   ✓ {split:<6}: {n_imgs} imagens")

    if not YAML_PATH.exists():
        print(f"  ❌ data.yaml não encontrado em {YAML_PATH}")
        return False

    return True


def corrigir_yaml():
    """Cria data_fixed.yaml com caminhos absolutos e classes corretas."""
    with open(YAML_PATH) as f:
        config = yaml.safe_load(f)

    # Verificar se as classes do dataset batem com as esperadas
    classes_dataset = config.get("names", [])
    print(f"\n   Classes no dataset: {classes_dataset}")
    print(f"   Classes esperadas:  {CLASSES_MODELAGEM}")

    if sorted(classes_dataset) != sorted(CLASSES_MODELAGEM):
        print("\n   ⚠️  As classes do dataset não correspondem às esperadas.")
        print("      Verifique se anotou corretamente no Roboflow.")
        print("      Continuando com as classes do dataset...")

    config["path"]  = str(DATASET_DIR.resolve())
    config["train"] = str((DATASET_DIR / "train" / "images").resolve())
    config["val"]   = str((DATASET_DIR / "valid" / "images").resolve())
    config["test"]  = str((DATASET_DIR / "test"  / "images").resolve())

    with open(YAML_FIXED, "w") as f:
        yaml.dump(config, f, allow_unicode=True, sort_keys=False)

    print(f"\n   ✓ data_fixed.yaml criado")
    return str(YAML_FIXED), config


def treinar():
    print("\n" + "="*60)
    print("  FASE 3C — Treinamento YOLOv8 (Modelagens de Camiseta)")
    print("="*60)

    # Verificar GPU
    print(f"\n  GPU  : {torch.cuda.get_device_name(0)}")
    print(f"  VRAM : {round(torch.cuda.get_device_properties(0).total_memory/1024**3,1)} GB")

    # Verificar dataset
    print("\n[1/4] Verificando dataset...")
    if not verificar_dataset():
        return

    # Corrigir YAML
    print("\n[2/4] Configurando data.yaml...")
    yaml_path, config = corrigir_yaml()
    print(f"      Classes: {config.get('names', [])}")
    print(f"      nc     : {config.get('nc', '?')}")

    # Carregar modelo base
    print("\n[3/4] Carregando YOLOv8s pré-treinado...")
    model = YOLO("yolov8s.pt")
    print("      ✓ yolov8s.pt carregado")

    # Treinar
    print(f"\n[4/4] Iniciando treino...")
    print(f"      Épocas  : 50")
    print(f"      Batch   : 16")
    print(f"      Imagem  : 640×640")
    print(f"      Classes : {CLASSES_MODELAGEM}")
    print(f"      ⏱️  Estimativa: 15–30 min na RTX 4060 Ti\n")
    print("="*60)

    results = model.train(
        data     = yaml_path,
        epochs   = 50,
        imgsz    = 640,
        batch    = 16,
        device   = 0,
        workers  = 4,
        patience = 15,
        name     = "camisetas_modelagem_v1",
        project  = str(RUNS_DIR),
        amp      = True,
        exist_ok = True,
        verbose  = True,
    )

    # Resultados
    modelo_final = RUNS_DIR / "camisetas_modelagem_v1" / "weights" / "best.pt"
    print("\n" + "="*60)
    print("  TREINO CONCLUÍDO!")
    print("="*60)

    if modelo_final.exists():
        print(f"\n  ✅ Modelo salvo em:")
        print(f"     {modelo_final}")
        print(f"\n  📊 Métricas finais:")
        print(f"     mAP50    : {results.results_dict.get('metrics/mAP50(B)', 0):.3f}")
        print(f"     mAP50-95 : {results.results_dict.get('metrics/mAP50-95(B)', 0):.3f}")
        print(f"     Precisão : {results.results_dict.get('metrics/precision(B)', 0):.3f}")
        print(f"     Recall   : {results.results_dict.get('metrics/recall(B)', 0):.3f}")

    print("\n  Próximo: python yolo/avaliar_yolo.py")
    print("="*60)


if __name__ == "__main__":
    treinar()
