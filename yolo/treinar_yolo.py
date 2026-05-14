"""
FASE 3C — Treinamento do YOLOv8 para Detecção de Camisetas
TCC: Sistema de Recomendação de Tamanho para Vestuário Superior

Dataset: Clothing Detection (Roboflow) — 3.094 imagens, 16 classes
GPU: NVIDIA GeForce RTX 4060 Ti (8GB VRAM)
Modelo base: YOLOv8s (small) — melhor equilíbrio velocidade/precisão para 8GB

Classes de interesse para o TCC (vestuário superior masculino):
  - Shirt, Long Shirt, SleevelessShirt, Hoodie, Jacket, Short, Male
"""

import os
import yaml
import shutil
from pathlib import Path
from ultralytics import YOLO
import torch

# ── Verificar GPU ────────────────────────────────────────────
print("=" * 60)
print("  FASE 3C — Treinamento YOLOv8")
print("=" * 60)
print(f"\n  GPU disponível : {torch.cuda.is_available()}")
print(f"  Dispositivo    : {torch.cuda.get_device_name(0)}")
print(f"  VRAM           : {round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 1)} GB")

# ── Caminhos ─────────────────────────────────────────────────
BASE_DIR    = Path(__file__).parent.parent
DATASET_DIR = BASE_DIR / "data" / "deepfashion"
YAML_PATH   = DATASET_DIR / "data.yaml"
YAML_FIXED  = DATASET_DIR / "data_fixed.yaml"
RUNS_DIR    = BASE_DIR / "yolo" / "runs"

# ── Corrigir data.yaml com caminhos absolutos ────────────────
def corrigir_yaml():
    """
    O data.yaml original usa caminhos relativos (../train/images)
    que não funcionam no Windows com YOLOv8. Criamos uma versão
    com caminhos absolutos.
    """
    with open(YAML_PATH, "r") as f:
        config = yaml.safe_load(f)

    config["path"]  = str(DATASET_DIR.resolve())
    config["train"] = str((DATASET_DIR / "train" / "images").resolve())
    config["val"]   = str((DATASET_DIR / "valid" / "images").resolve())
    config["test"]  = str((DATASET_DIR / "test"  / "images").resolve())

    with open(YAML_FIXED, "w") as f:
        yaml.dump(config, f, allow_unicode=True, sort_keys=False)

    print(f"\n  ✓ data_fixed.yaml criado com caminhos absolutos")
    print(f"    train : {config['train']}")
    print(f"    val   : {config['val']}")
    print(f"    nc    : {config['nc']} classes")
    print(f"    names : {config['names']}")

    return str(YAML_FIXED)

# ── Configurações de treino ───────────────────────────────────
# Otimizadas para RTX 4060 Ti (8GB VRAM)
CONFIG_TREINO = {
    # Modelo base — 'yolov8s.pt' é o ponto de partida pré-treinado no COCO
    # s = small: bom equilíbrio entre velocidade e precisão para 8GB VRAM
    "model":     "yolov8s.pt",

    # Épocas — quantas vezes o modelo vê todo o dataset
    # 50 épocas é suficiente para fine-tuning com dataset de 3k imagens
    "epochs":    50,

    # Tamanho da imagem — 640x640 é o padrão do YOLOv8
    "imgsz":     640,

    # Batch size — quantas imagens por vez na GPU
    # 16 é seguro para 8GB VRAM com yolov8s
    "batch":     16,

    # Dispositivo — 0 = primeira GPU (RTX 4060 Ti)
    "device":    0,

    # Workers — threads para carregar dados
    "workers":   4,

    # Paciência para early stopping
    # Para o treino se não melhorar por 15 épocas consecutivas
    "patience":  15,

    # Nome do experimento (pasta onde salva resultados)
    "name":      "clothing_yolov8s_v1",

    # Diretório de saída
    "project":   str(RUNS_DIR),

    # Otimizações de performance
    "cache":     False,   # False = não usa RAM para cache (mais seguro)
    "amp":       True,    # True = mixed precision (mais rápido na RTX)
    "exist_ok":  True,    # Permite reescrever experimento existente
    "verbose":   True,    # Mostrar logs detalhados
}

# ── Executar treino ───────────────────────────────────────────
def treinar():
    print("\n[1/3] Corrigindo data.yaml...")
    yaml_path = corrigir_yaml()

    print("\n[2/3] Carregando modelo base YOLOv8s...")
    model = YOLO(CONFIG_TREINO["model"])
    print("      ✓ Modelo carregado")

    print(f"\n[3/3] Iniciando treino...")
    print(f"      Épocas   : {CONFIG_TREINO['epochs']}")
    print(f"      Batch    : {CONFIG_TREINO['batch']}")
    print(f"      Imagem   : {CONFIG_TREINO['imgsz']}x{CONFIG_TREINO['imgsz']}")
    print(f"      GPU      : RTX 4060 Ti")
    print(f"      Saída    : {RUNS_DIR}/clothing_yolov8s_v1/")
    print("\n  ⏱️  Tempo estimado: 20–40 minutos para 50 épocas")
    print("     (você verá o progresso época por época abaixo)\n")
    print("=" * 60)

    results = model.train(
        data    = yaml_path,
        epochs  = CONFIG_TREINO["epochs"],
        imgsz   = CONFIG_TREINO["imgsz"],
        batch   = CONFIG_TREINO["batch"],
        device  = CONFIG_TREINO["device"],
        workers = CONFIG_TREINO["workers"],
        patience= CONFIG_TREINO["patience"],
        name    = CONFIG_TREINO["name"],
        project = CONFIG_TREINO["project"],
        cache   = CONFIG_TREINO["cache"],
        amp     = CONFIG_TREINO["amp"],
        exist_ok= CONFIG_TREINO["exist_ok"],
        verbose = CONFIG_TREINO["verbose"],
    )

    # ── Resultados finais ─────────────────────────────────────
    print("\n" + "=" * 60)
    print("  TREINO CONCLUÍDO!")
    print("=" * 60)

    modelo_salvo = RUNS_DIR / "clothing_yolov8s_v1" / "weights" / "best.pt"
    if modelo_salvo.exists():
        print(f"\n  ✅ Melhor modelo salvo em:")
        print(f"     {modelo_salvo}")
        print(f"\n  📊 Métricas finais:")
        print(f"     mAP50    : {results.results_dict.get('metrics/mAP50(B)', 'N/A'):.3f}")
        print(f"     mAP50-95 : {results.results_dict.get('metrics/mAP50-95(B)', 'N/A'):.3f}")
        print(f"     Precisão : {results.results_dict.get('metrics/precision(B)', 'N/A'):.3f}")
        print(f"     Recall   : {results.results_dict.get('metrics/recall(B)', 'N/A'):.3f}")
    else:
        print("Modelo não encontrado no caminho esperado.")

    print("\n  Próximo passo: python yolo/avaliar_yolo.py")
    print("=" * 60)

    return results


if __name__ == "__main__":
    treinar()