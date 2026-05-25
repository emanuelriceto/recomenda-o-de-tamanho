"""
FASE 3C — Avaliação do Modelo YOLOv8 Treinado
TCC: Sistema de Recomendação de Tamanho para Vestuário Superior

Avalia o modelo treinado e testa detecção de modelagem em imagens novas.
Execute APÓS treinar_yolo.py.
"""

from pathlib import Path
from ultralytics import YOLO
import torch

BASE_DIR    = Path(__file__).parent.parent
DATASET_DIR = BASE_DIR / "data" / "deepfashion"
YAML_FIXED  = DATASET_DIR / "data_fixed.yaml"
MODELO_PATH = (BASE_DIR / "yolo" / "runs" /
               "camisetas_modelagem_v1" / "weights" / "best.pt")

CLASSES_MODELAGEM = ["regular", "slim", "oversized", "longline", "henley"]


def avaliar():
    print("="*60)
    print("  AVALIAÇÃO DO MODELO YOLOV8 — Modelagens de Camiseta")
    print("="*60)

    if not MODELO_PATH.exists():
        print(f"\n  ❌ Modelo não encontrado em:\n     {MODELO_PATH}")
        print("     Execute primeiro: python yolo/treinar_yolo.py")
        return

    print(f"\n  GPU    : {torch.cuda.get_device_name(0)}")
    print(f"  Modelo : {MODELO_PATH}")
    model = YOLO(str(MODELO_PATH))

    # Avaliar no conjunto de validação
    print("\n[1/2] Avaliando no conjunto de validação...")
    metrics = model.val(
        data    = str(YAML_FIXED),
        imgsz   = 640,
        device  = 0,
        verbose = True,
    )

    print("\n" + "="*60)
    print("  MÉTRICAS FINAIS")
    print("="*60)
    print(f"  mAP50    : {metrics.box.map50:.3f}")
    print(f"  mAP50-95 : {metrics.box.map:.3f}")
    print(f"  Precisão : {metrics.box.mp:.3f}")
    print(f"  Recall   : {metrics.box.mr:.3f}")

    # Testar em imagens do conjunto de teste
    print("\n[2/2] Testando em 5 imagens do conjunto de teste...")
    test_imgs = list((DATASET_DIR / "test" / "images").glob("*.jpg"))[:5]

    if test_imgs:
        results = model.predict(
            source   = [str(img) for img in test_imgs],
            imgsz    = 640,
            device   = 0,
            conf     = 0.25,
            save     = True,
            project  = str(BASE_DIR / "yolo" / "runs"),
            name     = "testes_modelagem",
            exist_ok = True,
        )

        print(f"\n  Detecções:")
        for img, result in zip(test_imgs, results):
            boxes = result.boxes
            if boxes is not None and len(boxes) > 0:
                detectadas = [result.names[int(c)] for c in boxes.cls]
                confianças = [f"{c:.0%}" for c in boxes.conf]
                print(f"   ✓ {img.name[:35]}")
                for det, conf in zip(detectadas, confianças):
                    print(f"      → {det} ({conf})")
            else:
                print(f"   ✗ {img.name[:35]} → nenhuma detecção")

        print(f"\n  Imagens com detecções salvas em:")
        print(f"  yolo/runs/testes_modelagem/")

    print("\n✅ Avaliação concluída!")
    print("="*60)


if __name__ == "__main__":
    avaliar()
