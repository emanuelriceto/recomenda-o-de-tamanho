"""Carrega o XGBoost treinado e produz a predição de tamanho (referência Hering)."""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from api.config import MODELAGEM_MAP

COLUNAS_FAIXA = ["busto_circunf", "largura_ombro", "altura", "peso_kg", "imc"]


class ServicoML:
    def __init__(self, model_dir: Path):
        model_dir = Path(model_dir)
        nomes = ["modelo_recomendacao.joblib", "label_encoder.joblib", "modelo_info.json"]
        faltando = [n for n in nomes if not (model_dir / n).exists()]
        if faltando:
            raise FileNotFoundError(
                f"Arquivos do modelo ausentes em {model_dir}: {faltando}. "
                "Execute: python model/preparar_dados.py && python model/treinar_modelo.py")
        self.pipeline = joblib.load(model_dir / "modelo_recomendacao.joblib")
        self.le = joblib.load(model_dir / "label_encoder.joblib")
        with open(model_dir / "modelo_info.json", encoding="utf-8") as f:
            self.info = json.load(f)
        self.features = self.info["features"]
        self.classes = [str(c) for c in self.le.classes_]
        self.faixas = self._faixas_treino(model_dir / "dataset_treinamento.csv")

    @staticmethod
    def _faixas_treino(csv: Path) -> dict:
        """Percentis 1–99 das medidas de treino: fora disso, a predição é extrapolação."""
        if not csv.exists():
            return {}
        df = pd.read_csv(csv, usecols=lambda c: c in COLUNAS_FAIXA)
        return {c: (float(df[c].quantile(0.01)), float(df[c].quantile(0.99))) for c in df.columns}

    @staticmethod
    def montar_linha(medidas: dict, modelagem: str) -> dict:
        busto = float(medidas["busto_circunf"])
        ombro = float(medidas["largura_ombro"])
        altura = float(medidas["altura"])
        peso = float(medidas["peso_kg"])
        cintura = medidas.get("cintura_circunf") or busto * 0.9      # mesma imputação do treino v1
        braco = medidas.get("comprimento_braco") or altura * 0.49
        return {
            "busto_circunf": busto, "largura_ombro": ombro, "altura": altura,
            "peso_kg": peso, "imc": peso / (altura / 100) ** 2,
            "ratio_busto_ombro": busto / ombro, "modelagem_num": MODELAGEM_MAP[modelagem],
            "cintura_circunf": float(cintura), "comprimento_braco": float(braco),
            "ratio_busto_cintura": busto / float(cintura),
        }

    def prever(self, medidas: dict, modelagem: str, top_k: int = 3) -> dict:
        linha = self.montar_linha(medidas, modelagem)
        X = pd.DataFrame([[linha[f] for f in self.features]], columns=self.features)
        proba = self.pipeline.predict_proba(X)[0]
        ordem = np.argsort(proba)[::-1]
        return {
            "tamanho": self.classes[int(ordem[0])],
            "confianca": float(proba[ordem[0]]),
            "alternativas": [{"tamanho": self.classes[int(i)], "probabilidade": round(float(proba[i]), 3)}
                             for i in ordem[:top_k]],
            "imc": round(linha["imc"], 1),
        }

    def avisos_distribuicao(self, medidas: dict, imc: float) -> tuple[list[str], bool]:
        valores = {**medidas, "imc": imc}
        fora = [c for c, (lo, hi) in self.faixas.items()
                if valores.get(c) is not None and not lo <= float(valores[c]) <= hi]
        avisos = []
        if fora:
            avisos.append("Medidas fora da faixa dos dados de treino (" + ", ".join(fora)
                          + "): recomendação com confiabilidade reduzida.")
        if imc >= 40:
            avisos.append("IMC ≥ 40 (obesidade grau III, OMS): confira as medidas da peça antes da compra.")
        elif imc < 16:
            avisos.append("IMC < 16 (magreza grau III, OMS): confira as medidas da peça antes da compra.")
        return avisos, bool(fora)
