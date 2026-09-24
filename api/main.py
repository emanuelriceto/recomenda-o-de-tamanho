"""
FASE 4 — API REST (FastAPI) integrando YOLOv8 + XGBoost
Executar (na raiz do projeto, com venv_yolo ativo):
    uvicorn api.main:app --reload --port 8000
Documentação interativa: http://localhost:8000/docs
"""
import logging
import time
from contextlib import asynccontextmanager
from typing import Annotated, Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from pydantic import ValidationError

from api.config import (DB_PATH, LIMIAR_CONF_YOLO, MODEL_DIR, MODELAGEM_FALLBACK,
                        MODELAGENS, TAMANHO_MAX_FOTO_MB, YOLO_WEIGHTS, CONF_MIN_DETECCAO)
from api.schemas import MedidasCliente, Modelagem, PedidoRecomendacao, RespostaRecomendacao
from api.servicos.banco_servico import ServicoBanco
from api.servicos.ml_servico import ServicoML
from api.servicos.recomendador import MarcaNaoEncontrada, Recomendador
from api.servicos.yolo_servico import ImagemInvalida, ServicoYOLO, abrir_imagem

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger("api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.banco = ServicoBanco(DB_PATH)
    app.state.ml = ServicoML(MODEL_DIR)
    app.state.yolo = ServicoYOLO(YOLO_WEIGHTS, conf_min=CONF_MIN_DETECCAO)
    app.state.recomendador = Recomendador(app.state.banco, app.state.ml)
    if app.state.yolo.disponivel:
        log.info("YOLOv8 carregado: %s · classes %s", YOLO_WEIGHTS, app.state.yolo.classes)
        if sorted(app.state.yolo.classes) != sorted(MODELAGENS):
            log.warning("Classes do YOLO diferentes das esperadas %s", MODELAGENS)
    else:
        log.warning("YOLOv8 indisponível (%s) — /recomendar-por-foto usará fallback '%s'",
                    app.state.yolo.erro, MODELAGEM_FALLBACK)
    log.info("XGBoost carregado: %s v%s", app.state.ml.info.get("nome_modelo"),
             app.state.ml.info.get("versao", 1))
    yield


app = FastAPI(
    title="API — Recomendação de Tamanho de Camisetas (TCC PUCPR)",
    description="Medidas corporais + foto da camiseta → YOLOv8 (modelagem) → XGBoost (tamanho) "
                "→ verificação por folga na tabela da marca.",
    version="2.0.0",
    lifespan=lifespan,
)


@app.get("/", tags=["status"])
def status():
    s = app.state
    return {
        "api": "ok",
        "versao": app.version,
        "modelo_ml": {"nome": s.ml.info.get("nome_modelo"), "versao": s.ml.info.get("versao", 1),
                      "classes": s.ml.classes},
        "yolo": {"disponivel": s.yolo.disponivel, "pesos": str(s.yolo.caminho),
                 "classes": s.yolo.classes, "erro": s.yolo.erro},
        "marcas": s.banco.listar_marcas(),
    }


@app.get("/marcas", tags=["catálogo"])
def marcas():
    return {"marcas": app.state.banco.listar_marcas()}


@app.get("/marcas/{marca}/tamanhos", tags=["catálogo"])
def tamanhos(marca: str, modelagem: Optional[Modelagem] = None):
    nome = app.state.banco.nome_marca(marca)
    if nome is None:
        raise HTTPException(404, f"Marca '{marca}' não cadastrada.")
    return {"marca": nome, "modelagem": modelagem, "tamanhos": app.state.banco.tabela(nome, modelagem)}


@app.post("/recomendar-tamanho", response_model=RespostaRecomendacao, tags=["recomendação"])
def recomendar_tamanho(pedido: PedidoRecomendacao):
    """Recomendação com a modelagem informada manualmente (sem foto)."""
    inicio = time.perf_counter()
    try:
        r = app.state.recomendador.recomendar(
            pedido.medidas.model_dump(), pedido.marca, pedido.modelagem, origem="informada")
    except MarcaNaoEncontrada as e:
        raise HTTPException(404, str(e))
    r["tempo_ms"] = round((time.perf_counter() - inicio) * 1000, 1)
    return r


@app.post("/recomendar-por-foto", response_model=RespostaRecomendacao, tags=["recomendação"])
def recomendar_por_foto(
    foto: Annotated[UploadFile, File(description="Foto da camiseta (JPG/PNG/WEBP)")],
    busto_circunf: Annotated[float, Form(ge=60, le=180)],
    largura_ombro: Annotated[float, Form(ge=30, le=70)],
    altura: Annotated[float, Form(ge=140, le=220)],
    peso_kg: Annotated[float, Form(ge=30, le=250)],
    cintura_circunf: Annotated[Optional[float], Form(ge=50, le=200)] = None,
    comprimento_braco: Annotated[Optional[float], Form(ge=40, le=110)] = None,
    marca: Annotated[str, Form()] = "Hering",
):
    """Fluxo completo: foto → YOLOv8 detecta a modelagem → XGBoost recomenda o tamanho."""
    inicio = time.perf_counter()

    # 1) validação do arquivo
    if foto.content_type and not foto.content_type.startswith("image/"):
        raise HTTPException(415, f"Tipo de arquivo não suportado: {foto.content_type}")
    conteudo = foto.file.read()
    if len(conteudo) > TAMANHO_MAX_FOTO_MB * 1024 * 1024:
        raise HTTPException(413, f"Imagem maior que {TAMANHO_MAX_FOTO_MB} MB.")
    try:
        img = abrir_imagem(conteudo)
    except ImagemInvalida as e:
        raise HTTPException(400, str(e))

    try:
        medidas = MedidasCliente(
            busto_circunf=busto_circunf, largura_ombro=largura_ombro, altura=altura,
            peso_kg=peso_kg, cintura_circunf=cintura_circunf,
            comprimento_braco=comprimento_braco).model_dump()
    except ValidationError as e:
        raise HTTPException(422, e.errors())

    # 2) YOLOv8 → modelagem (com fallback)
    avisos: list[str] = []
    yolo = app.state.yolo
    detectada, conf_yolo = None, None
    if not yolo.disponivel:
        modelagem, origem = MODELAGEM_FALLBACK, "fallback"
        avisos.append(f"Módulo de visão indisponível; modelagem '{MODELAGEM_FALLBACK}' assumida.")
    else:
        det = yolo.detectar(img)
        detectada, conf_yolo = det["modelagem"], det["confianca"]
        if detectada is None:
            modelagem, origem = MODELAGEM_FALLBACK, "fallback"
            avisos.append(f"Nenhuma camiseta detectada na foto; modelagem '{MODELAGEM_FALLBACK}' assumida.")
        elif conf_yolo < LIMIAR_CONF_YOLO:
            modelagem, origem = MODELAGEM_FALLBACK, "fallback"
            avisos.append(f"Modelagem detectada ({detectada}, {conf_yolo:.0%}) abaixo do limiar de "
                          f"{LIMIAR_CONF_YOLO:.0%}; modelagem '{MODELAGEM_FALLBACK}' assumida.")
        else:
            modelagem, origem = detectada, "yolo"

    # 3) XGBoost + verificação por folga
    try:
        r = app.state.recomendador.recomendar(
            medidas, marca, modelagem, origem=origem, conf_yolo=conf_yolo,
            detectada=detectada, avisos=avisos)
    except MarcaNaoEncontrada as e:
        raise HTTPException(404, str(e))
    r["tempo_ms"] = round((time.perf_counter() - inicio) * 1000, 1)
    log.info("foto=%s | modelagem=%s (%s) | tamanho=%s | %.0f ms",
             foto.filename, modelagem, origem, r["tamanho_recomendado"], r["tempo_ms"])
    return r
