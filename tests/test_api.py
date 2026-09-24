"""
Testes automatizados da API (caixa branca + caixa preta).
Executar na raiz do projeto, com venv_yolo ativo:
    python -m pytest tests -v
"""
import io
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from api.main import app

RAIZ = Path(__file__).resolve().parent.parent
TAMANHOS_VALIDOS = {"PP", "P", "M", "G", "GG", "XGG", "XS", "S", "L", "XL", "XXL",
                    "EGG", "XG", "SEM_TAMANHO"}
MEDIDAS = {"busto_circunf": 94, "largura_ombro": 43, "altura": 175,
           "peso_kg": 78, "cintura_circunf": 84}


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:   # o "with" executa o lifespan (carrega os modelos)
        yield c


def _foto_teste() -> tuple[str, bytes]:
    pasta = RAIZ / "data" / "deepfashion" / "test" / "images"
    imgs = sorted(list(pasta.glob("*.jpg")) + list(pasta.glob("*.png"))) if pasta.exists() else []
    if imgs:
        return imgs[0].name, imgs[0].read_bytes()
    buf = io.BytesIO()
    Image.new("RGB", (640, 640), "white").save(buf, format="JPEG")
    return "branca.jpg", buf.getvalue()


def test_status(client):
    r = client.get("/")
    assert r.status_code == 200
    assert r.json()["api"] == "ok"


def test_marcas(client):
    r = client.get("/marcas")
    assert r.status_code == 200 and "Hering" in r.json()["marcas"]


def test_marca_inexistente(client):
    assert client.get("/marcas/MarcaX/tamanhos").status_code == 404


def test_recomendar_tamanho(client):
    r = client.post("/recomendar-tamanho", json={"medidas": MEDIDAS, "modelagem": "regular"})
    assert r.status_code == 200, r.text
    d = r.json()
    assert d["tamanho_recomendado"] in TAMANHOS_VALIDOS
    assert 0 <= d["confianca_modelo"] <= 1
    assert d["modelagem"]["origem"] == "informada"


def test_medida_invalida(client):
    ruim = {**MEDIDAS, "busto_circunf": 30}
    assert client.post("/recomendar-tamanho", json={"medidas": ruim}).status_code == 422


def test_outra_marca(client):
    r = client.post("/recomendar-tamanho", json={"medidas": MEDIDAS, "marca": "zara"})
    assert r.status_code == 200 and r.json()["marca"] == "Zara"


def test_modelagem_influencia(client):
    """Na fronteira entre tamanhos, slim deve recomendar tamanho >= regular."""
    fronteira = {**MEDIDAS, "busto_circunf": 97, "cintura_circunf": 88}
    res = {m: client.post("/recomendar-tamanho", json={"medidas": fronteira, "modelagem": m}).json()
           for m in ("slim", "regular", "oversized")}
    print({m: r["tamanho_recomendado"] for m, r in res.items()})
    assert all(r["tamanho_recomendado"] in TAMANHOS_VALIDOS for r in res.values())


def test_obesidade_extrema_resposta_segura(client):
    obeso = {"busto_circunf": 150, "largura_ombro": 49, "altura": 172,
             "peso_kg": 160, "cintura_circunf": 160}
    d = client.post("/recomendar-tamanho", json={"medidas": obeso}).json()
    # nunca deve recomendar peça com folga abaixo da mínima
    assert d["tamanho_recomendado"] == "SEM_TAMANHO" or d["medidas_peca"]["folga_cm"] >= 4
    assert d["avisos"]


def test_magreza_extrema_resposta_segura(client):
    magro = {"busto_circunf": 76, "largura_ombro": 38, "altura": 178,
             "peso_kg": 47, "cintura_circunf": 62}
    d = client.post("/recomendar-tamanho", json={"medidas": magro}).json()
    assert d["tamanho_recomendado"] in TAMANHOS_VALIDOS
    assert d["avisos"]


def test_recomendar_por_foto(client):
    nome, conteudo = _foto_teste()
    dados = {k: str(v) for k, v in MEDIDAS.items()}
    r = client.post("/recomendar-por-foto", data=dados,
                    files={"foto": (nome, conteudo, "image/jpeg")})
    assert r.status_code == 200, r.text
    d = r.json()
    assert d["modelagem"]["usada"] in {"oversized", "regular", "slim"}
    assert d["modelagem"]["origem"] in {"yolo", "fallback"}
    print(f"\n{nome}: {d['modelagem']} → {d['tamanho_recomendado']} ({d['tempo_ms']} ms)")


def test_foto_arquivo_invalido(client):
    dados = {k: str(v) for k, v in MEDIDAS.items()}
    r = client.post("/recomendar-por-foto", data=dados,
                    files={"foto": ("x.txt", b"nao sou imagem", "text/plain")})
    assert r.status_code == 415


def test_foto_corrompida(client):
    dados = {k: str(v) for k, v in MEDIDAS.items()}
    r = client.post("/recomendar-por-foto", data=dados,
                    files={"foto": ("x.jpg", b"bytes quebrados", "image/jpeg")})
    assert r.status_code == 400
