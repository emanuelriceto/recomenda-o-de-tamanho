"""
Regra de folga (ease) — gabarito de tamanho do projeto.
Usada em dois lugares:
  1. model/preparar_dados.py  -> gera o rótulo (tamanho) para treinar o XGBoost
  2. API (api/servicos/recomendador.py) -> verificação de segurança da resposta

Folga = circunferência da peça − circunferência corporal governante
  • circunferência da peça = largura_busto_min + largura_busto_max
    (a tabela registra a largura da peça estendida, isto é, meia circunferência;
     somar as duas pontas = 2 × largura média)
  • circunferência governante = max(tórax, cintura): em perfis com abdômen
    proeminente (ex.: obesidade) a cintura supera o tórax e a camiseta de corte
    reto precisa vestir as duas regiões.

Os parâmetros (cm) são uma hipótese inicial de projeto e devem ser calibrados
com a validação por voluntários.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

FOLGA = {
    "slim":      {"min": 2.0,  "alvo": 5.0},
    "regular":   {"min": 4.0,  "alvo": 8.0},
    "oversized": {"min": 12.0, "alvo": 18.0},
}
EXCESSO_FOLGA = 12.0          # folga > alvo + 12 cm => peça muito folgada
SEM_TAMANHO = "SEM_TAMANHO"   # nenhum tamanho da marca atende à folga mínima


@dataclass
class Escolha:
    tamanho: str
    folga_cm: float
    status: str  # "ok" | "folgado" | "sem_tamanho"


def _valido(x) -> bool:
    return x is not None and not (isinstance(x, float) and math.isnan(x))


def circunferencia_governante(busto: float, cintura: float | None = None) -> float:
    return float(max(busto, cintura)) if _valido(cintura) else float(busto)


def circunferencia_peca(largura_min: float, largura_max: float) -> float:
    return float(largura_min) + float(largura_max)


def escolher_tamanho(busto, cintura, tamanhos, modelagem) -> Escolha:
    """
    tamanhos: lista de dicts (mesma marca e modelagem) com as chaves
              tamanho_label, tamanho_ordem, largura_busto_min, largura_busto_max.
    Regra: entre os tamanhos com folga >= mínima, escolhe o de folga mais
    próxima do alvo; em empate, o maior (evita peça apertada).
    """
    if modelagem not in FOLGA:
        raise ValueError(f"Modelagem inválida: {modelagem}")
    if not tamanhos:
        raise ValueError("Tabela de tamanhos vazia")
    p = FOLGA[modelagem]
    corpo = circunferencia_governante(busto, cintura)
    ordenados = sorted(tamanhos, key=lambda t: t["tamanho_ordem"])
    candidatos = [
        (t, circunferencia_peca(t["largura_busto_min"], t["largura_busto_max"]) - corpo)
        for t in ordenados
    ]
    validos = [(t, f) for t, f in candidatos if f >= p["min"]]
    if not validos:
        return Escolha(SEM_TAMANHO, round(candidatos[-1][1], 1), "sem_tamanho")
    t, f = min(validos, key=lambda c: (abs(c[1] - p["alvo"]), -c[0]["tamanho_ordem"]))
    status = "folgado" if f > p["alvo"] + EXCESSO_FOLGA else "ok"
    return Escolha(t["tamanho_label"], round(f, 1), status)
