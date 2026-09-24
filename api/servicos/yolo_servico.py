"""Carrega o YOLOv8 treinado (best.pt) e detecta a modelagem da camiseta."""
import io
import threading
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError


class ImagemInvalida(ValueError):
    pass


def abrir_imagem(conteudo: bytes) -> Image.Image:
    """Valida e abre a imagem; corrige a rotação EXIF de fotos de celular."""
    try:
        img = Image.open(io.BytesIO(conteudo))
        img.load()
    except (UnidentifiedImageError, OSError) as e:
        raise ImagemInvalida("O arquivo enviado não é uma imagem válida.") from e
    return ImageOps.exif_transpose(img).convert("RGB")


class ServicoYOLO:
    def __init__(self, pesos: Path, conf_min: float = 0.25, imgsz: int = 640):
        self.caminho = Path(pesos)
        self.conf_min = conf_min
        self.imgsz = imgsz
        self.modelo = None
        self.classes: list[str] = []
        self.erro: str | None = None
        self._lock = threading.Lock()   # predict não é thread-safe no mesmo objeto
        if not self.caminho.exists():
            self.erro = f"Pesos não encontrados: {self.caminho}"
            return
        try:
            from ultralytics import YOLO
            self.modelo = YOLO(str(self.caminho))
            self.classes = [self.modelo.names[i] for i in sorted(self.modelo.names)]
            # aquecimento: evita que a 1ª requisição pague a inicialização da GPU
            self.modelo.predict(np.zeros((imgsz, imgsz, 3), dtype=np.uint8),
                                imgsz=imgsz, verbose=False)
        except Exception as e:  # noqa: BLE001 — API sobe mesmo sem YOLO (fallback)
            self.modelo = None
            self.erro = f"{type(e).__name__}: {e}"

    @property
    def disponivel(self) -> bool:
        return self.modelo is not None

    def detectar(self, img: Image.Image) -> dict:
        """Retorna a classe de maior confiança, ou modelagem=None se nada for detectado."""
        with self._lock:
            r = self.modelo.predict(img, conf=self.conf_min, imgsz=self.imgsz, verbose=False)[0]
        if r.boxes is None or len(r.boxes) == 0:
            return {"modelagem": None, "confianca": None}
        i = int(r.boxes.conf.argmax())
        return {"modelagem": r.names[int(r.boxes.cls[i])],
                "confianca": round(float(r.boxes.conf[i]), 4)}
