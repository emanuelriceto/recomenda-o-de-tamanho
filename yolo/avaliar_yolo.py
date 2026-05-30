"""
FASE 3C — Avaliação do YOLOv8 Treinado
TCC: Sistema de Recomendação de Tamanho para Vestuário Superior

Classes: oversized (0), regular (1), slim (2)
Execute APÓS treinar_yolo.py com venv_yolo ativo.
"""

from pathlib import Path
from ultralytics import YOLO
import torch

BASE_DIR    = Path(__file__).parent.parent
DATASET_DIR = BASE_DIR / 'data' / 'deepfashion'
YAML_FIXED  = DATASET_DIR / 'data_fixed.yaml'
MODELO_PATH = (BASE_DIR / 'yolo' / 'runs' /
               'camisetas_modelagem_v1' / 'weights' / 'best.pt')

# Ordem exata do data.yaml
CLASSES_YOLO = ['oversized', 'regular', 'slim']


def avaliar():
    print('='*60)
    print('  AVALIAÇÃO YOLOV8 — oversized / regular / slim')
    print('='*60)

    if not MODELO_PATH.exists():
        print(f'\n  ❌ Modelo não encontrado:\n     {MODELO_PATH}')
        print('     Execute primeiro: python yolo/treinar_yolo.py')
        return

    device = 0 if torch.cuda.is_available() else 'cpu'
    print(f'\n  Dispositivo: {"GPU — " + torch.cuda.get_device_name(0) if device == 0 else "CPU"}')
    print(f'  Modelo: {MODELO_PATH}')

    model = YOLO(str(MODELO_PATH))

    # Avaliar no conjunto de validação
    print('\n[1/2] Avaliando no conjunto de validação...')

    if not YAML_FIXED.exists():
        print('  ❌ data_fixed.yaml não encontrado. Execute treinar_yolo.py primeiro.')
        return

    metrics = model.val(
        data    = str(YAML_FIXED),
        imgsz   = 640,
        device  = device,
        verbose = True,
    )

    print('\n' + '='*60)
    print('  MÉTRICAS — Conjunto de Validação')
    print('='*60)
    print(f'  mAP50    : {metrics.box.map50:.3f}')
    print(f'  mAP50-95 : {metrics.box.map:.3f}')
    print(f'  Precisão : {metrics.box.mp:.3f}')
    print(f'  Recall   : {metrics.box.mr:.3f}')
    print(f'\n  Critério (mAP50 ≥ 0.50): '
          f'{"✅ APROVADO" if metrics.box.map50 >= 0.50 else "❌ ABAIXO"}')

    # Testar em imagens do conjunto de teste
    print('\n[2/2] Testando em imagens do conjunto de teste...')
    test_imgs = (list((DATASET_DIR / 'test' / 'images').glob('*.jpg')) +
                 list((DATASET_DIR / 'test' / 'images').glob('*.png')))[:5]

    if test_imgs:
        results = model.predict(
            source   = [str(img) for img in test_imgs],
            imgsz    = 640,
            device   = device,
            conf     = 0.25,
            save     = True,
            project  = str(BASE_DIR / 'yolo' / 'runs'),
            name     = 'testes_modelagem',
            exist_ok = True,
        )

        print(f'\n  Detecções nas imagens de teste:')
        for img, result in zip(test_imgs, results):
            boxes = result.boxes
            if boxes is not None and len(boxes) > 0:
                detectadas = [result.names[int(c)] for c in boxes.cls]
                confianças = [f'{c:.0%}' for c in boxes.conf]
                print(f'   ✓ {img.name[:45]}')
                for det, conf in zip(detectadas, confianças):
                    print(f'      → {det} ({conf})')
            else:
                print(f'   ✗ {img.name[:45]} → nenhuma detecção')

        print(f'\n  Imagens salvas em: yolo/runs/testes_modelagem/')
    else:
        print('  ⚠️  Nenhuma imagem encontrada em test/images/')

    print('\n✅ Avaliação concluída!')
    print('='*60)


if __name__ == '__main__':
    avaliar()
