"""Modelos de entrada e saída (Pydantic v2) — documentados automaticamente no /docs."""
from typing import Literal, Optional

from pydantic import BaseModel, Field

Modelagem = Literal["oversized", "regular", "slim"]


class MedidasCliente(BaseModel):
    busto_circunf: float = Field(..., ge=60, le=180, description="Circunferência do tórax (cm)")
    largura_ombro: float = Field(..., ge=30, le=70, description="Largura entre ombros (cm)")
    altura: float = Field(..., ge=140, le=220, description="Altura (cm)")
    peso_kg: float = Field(..., ge=30, le=250, description="Peso (kg)")
    cintura_circunf: Optional[float] = Field(None, ge=50, le=200, description="Circunferência da cintura (cm)")
    comprimento_braco: Optional[float] = Field(None, ge=40, le=110, description="Comprimento do braço (cm)")

    model_config = {"json_schema_extra": {"example": {
        "busto_circunf": 94, "largura_ombro": 43, "altura": 175,
        "peso_kg": 78, "cintura_circunf": 84}}}


class PedidoRecomendacao(BaseModel):
    medidas: MedidasCliente
    modelagem: Modelagem = "regular"
    marca: str = "Hering"


class Alternativa(BaseModel):
    tamanho: str
    probabilidade: float


class InfoModelagem(BaseModel):
    usada: Modelagem
    origem: Literal["yolo", "fallback", "informada"]
    detectada: Optional[str] = None
    confianca_yolo: Optional[float] = None


class MedidasPeca(BaseModel):
    tamanho: str
    largura_busto_cm: float
    circunferencia_peca_cm: float
    largura_ombro_cm: Optional[float] = None
    comprimento_cm: Optional[float] = None
    folga_cm: float


class RespostaRecomendacao(BaseModel):
    tamanho_recomendado: str
    marca: str
    tamanho_referencia_modelo: str
    confianca_modelo: float
    nivel_confianca: Literal["alta", "media", "baixa"]
    ajustado_pela_regra: bool
    modelagem: InfoModelagem
    alternativas: list[Alternativa]
    medidas_peca: Optional[MedidasPeca] = None
    imc: float
    avisos: list[str]
    tempo_ms: float = 0.0
