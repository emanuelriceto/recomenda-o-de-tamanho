"""Configurações centrais da API (caminhos, limiares e constantes)."""
import os
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent

DB_PATH = Path(os.getenv("DB_PATH", RAIZ / "database" / "antropometrico.db"))
MODEL_DIR = Path(os.getenv("MODEL_DIR", RAIZ / "model"))
YOLO_WEIGHTS = Path(os.getenv("YOLO_WEIGHTS", RAIZ / "yolo" / "weights" / "best.pt"))

MODELAGENS = ["oversized", "regular", "slim"]          # ordem do data.yaml
MODELAGEM_MAP = {m: i for i, m in enumerate(MODELAGENS)}
MODELAGEM_FALLBACK = "regular"
MARCA_REFERENCIA = "Hering"                            # marca usada no treino do XGBoost

LIMIAR_CONF_YOLO = float(os.getenv("LIMIAR_CONF_YOLO", 0.50))  # abaixo disso usa fallback
CONF_MIN_DETECCAO = 0.25                               # conf mínima para o YOLO devolver caixa
TAMANHO_MAX_FOTO_MB = 10
