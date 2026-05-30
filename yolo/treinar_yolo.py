"""
FASE 3C — Treinamento do YOLOv8
TCC: Sistema de Recomendação de Tamanho para Vestuário Superior

Dataset: camisetas-modelagem (Roboflow)
  - 3 classes: oversized (0), regular (1), slim (2)
  - data.yaml: names: ['oversized', 'regular', 'slim']
  - Distribuição: 70% train / 20% valid / 10% test

Execute com o venv_yolo ativo (Python 3.11 + PyTorch CUDA):
  source venv_yolo/Scripts/activate
  python yolo/treinar_yolo.py
"""

import os
import yaml
from pathlib import Path
from ultralytics import YOLO
import torch

BASE_DIR    = Path(__file__).parent.parent
DATASET_DIR = BASE_DIR / 'data' / 'deepfashion'
YAML_PATH   = DATASET_DIR / 'data.yaml'
YAML_FIXED  = DATASET_DIR / 'data_fixed.yaml'
RUNS_DIR    = BASE_DIR / 'yolo' / 'runs'

# Classes exatas do Roboflow — NÃO alterar esta ordem
# Alinhado com data.yaml: names: ['oversized', 'regular', 'slim']
CLASSES_YOLO = ['oversized', 'regular', 'slim']


def verificar_dataset():
    """Verifica se o dataset está na estrutura correta."""
    if not DATASET_DIR.exists():
        print(f'\n  ❌ Pasta do dataset não encontrada: {DATASET_DIR}')
        print('     Coloque o dataset em: data/deepfashion/')
        print('     Estrutura esperada: data/deepfashion/train/images/')
        return False

    for split in ['train', 'valid', 'test']:
        img_dir = DATASET_DIR / split / 'images'
        lbl_dir = DATASET_DIR / split / 'labels'
        if not img_dir.exists():
            print(f'  ❌ {split}/images/ não encontrada')
            return False
        n = len(list(img_dir.glob('*.jpg')) + list(img_dir.glob('*.png')))
        print(f'   ✓ {split:<6}: {n} imagens')

    return True


def corrigir_yaml():
    """
    Cria data_fixed.yaml com caminhos absolutos.
    Verifica que as classes batem com o esperado.
    """
    with open(YAML_PATH) as f:
        config = yaml.safe_load(f)

    classes_dataset = config.get('names', [])
    print(f'\n   Classes no data.yaml: {classes_dataset}')

    if classes_dataset != CLASSES_YOLO:
        print(f'   ⚠️  Esperado: {CLASSES_YOLO}')
        print('      Verifique o data.yaml do Roboflow.')
    else:
        print(f'   ✓ Classes corretas: {CLASSES_YOLO}')

    # Caminhos absolutos (necessário no Windows)
    config['path']  = str(DATASET_DIR.resolve())
    config['train'] = str((DATASET_DIR / 'train' / 'images').resolve())
    config['val']   = str((DATASET_DIR / 'valid' / 'images').resolve())
    config['test']  = str((DATASET_DIR / 'test'  / 'images').resolve())

    with open(YAML_FIXED, 'w') as f:
        yaml.dump(config, f, allow_unicode=True, sort_keys=False)

    print(f'   ✓ data_fixed.yaml criado com caminhos absolutos')
    return str(YAML_FIXED)


def treinar():
    print('\n' + '='*60)
    print('  FASE 3C — Treinamento YOLOv8')
    print('  Classes: oversized | regular | slim')
    print('='*60)

    # Verificar GPU
    if not torch.cuda.is_available():
        print('\n  ⚠️  GPU não detectada — treino será lento na CPU.')
        device = 'cpu'
    else:
        device = 0
        print(f'\n  GPU  : {torch.cuda.get_device_name(0)}')
        print(f'  VRAM : {round(torch.cuda.get_device_properties(0).total_memory/1024**3,1)} GB')

    # Verificar dataset
    print('\n[1/4] Verificando dataset...')
    if not verificar_dataset():
        return

    # Corrigir YAML
    print('\n[2/4] Configurando data.yaml...')
    yaml_path = corrigir_yaml()

    # Carregar modelo base
    print('\n[3/4] Carregando YOLOv8s pré-treinado...')
    model = YOLO('yolov8s.pt')
    print('      ✓ yolov8s.pt carregado (pré-treinado no COCO)')

    # Treinar
    print(f'\n[4/4] Iniciando treino...')
    print(f'      Épocas  : 50')
    print(f'      Batch   : 16')
    print(f'      Imagem  : 640×640')
    print(f'      Classes : {CLASSES_YOLO}')
    print(f'      Split   : 70% train / 20% valid / 10% test')
    print(f'      ⏱️  Estimativa: 15–30 min (RTX 4060 Ti)\n')
    print('='*60)

    results = model.train(
        data     = yaml_path,
        epochs   = 50,
        imgsz    = 640,
        batch    = 16,
        device   = device,
        workers  = 4,
        patience = 15,
        name     = 'camisetas_modelagem_v1',
        project  = str(RUNS_DIR),
        amp      = True,
        exist_ok = True,
        verbose  = True,
    )

    # Resultados
    modelo_final = RUNS_DIR / 'camisetas_modelagem_v1' / 'weights' / 'best.pt'
    print('\n' + '='*60)
    print('  TREINO CONCLUÍDO!')
    print('='*60)

    if modelo_final.exists():
        print(f'\n  ✅ Modelo salvo em:')
        print(f'     {modelo_final}')
        print(f'\n  📊 Métricas finais:')
        print(f'     mAP50    : {results.results_dict.get("metrics/mAP50(B)",    0):.3f}')
        print(f'     mAP50-95 : {results.results_dict.get("metrics/mAP50-95(B)", 0):.3f}')
        print(f'     Precisão : {results.results_dict.get("metrics/precision(B)", 0):.3f}')
        print(f'     Recall   : {results.results_dict.get("metrics/recall(B)",   0):.3f}')

        criterio = results.results_dict.get('metrics/mAP50(B)', 0) >= 0.50
        print(f'\n  Critério de aceite (mAP50 ≥ 0.50): '
              f'{"✅ APROVADO" if criterio else "❌ ABAIXO — coletar mais imagens"}')
    else:
        print('  ⚠️  Modelo não encontrado no caminho esperado.')

    print('\n  Próximo: python yolo/avaliar_yolo.py')
    print('='*60)


if __name__ == '__main__':
    treinar()
