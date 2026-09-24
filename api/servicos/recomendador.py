"""
Orquestra a recomendação:  XGBoost propõe  →  regra de folga verifica.

1. XGBoost prevê o tamanho na marca de referência (Hering) e a confiança.
2. O tamanho é convertido para a marca pedida pela circunferência da peça.
3. Verificação de segurança com a tabela da marca: se o tamanho proposto
   ficar justo (folga < mínima) ou muito folgado, é substituído pelo tamanho
   da regra; se nenhum tamanho servir, a resposta é SEM_TAMANHO.
"""
from api.config import MARCA_REFERENCIA
from model.regra_folga import (EXCESSO_FOLGA, FOLGA, SEM_TAMANHO,
                               circunferencia_governante, circunferencia_peca,
                               escolher_tamanho)


class MarcaNaoEncontrada(LookupError):
    pass


def _circ(linha: dict) -> float:
    return circunferencia_peca(linha["largura_busto_min"], linha["largura_busto_max"])


def _buscar(tabela: list[dict], tamanho: str) -> dict | None:
    return next((t for t in tabela if t["tamanho_label"] == tamanho), None)


class Recomendador:
    def __init__(self, banco, ml):
        self.banco = banco
        self.ml = ml

    @staticmethod
    def _converter(tamanho_ref: str, tabela_ref: list[dict], tabela_marca: list[dict]) -> str:
        """Tamanho equivalente na outra marca = peça de circunferência mais próxima."""
        ref = _buscar(tabela_ref, tamanho_ref)
        if tamanho_ref == SEM_TAMANHO or ref is None:
            return SEM_TAMANHO
        alvo = _circ(ref)
        escolhido = min(tabela_marca, key=lambda t: (abs(_circ(t) - alvo), -t["tamanho_ordem"]))
        return escolhido["tamanho_label"]

    @staticmethod
    def _nivel(conf_ml, conf_yolo, origem, fora_distribuicao, ajustado) -> str:
        if fora_distribuicao or ajustado:
            return "baixa"
        fator = {"yolo": conf_yolo or 0.0, "fallback": 0.6, "informada": 1.0}[origem]
        score = conf_ml * fator
        return "alta" if score >= 0.75 else "media" if score >= 0.50 else "baixa"

    def recomendar(self, medidas: dict, marca: str, modelagem: str, origem: str,
                   conf_yolo: float | None = None, detectada: str | None = None,
                   avisos: list[str] | None = None) -> dict:
        marca_nome = self.banco.nome_marca(marca)
        if marca_nome is None:
            raise MarcaNaoEncontrada(f"Marca '{marca}' não cadastrada.")
        tabela_marca = self.banco.tabela(marca_nome, modelagem)
        if not tabela_marca:
            raise MarcaNaoEncontrada(f"{marca_nome} não possui tabela para a modelagem '{modelagem}'.")
        tabela_ref = (tabela_marca if marca_nome == MARCA_REFERENCIA
                      else self.banco.tabela(MARCA_REFERENCIA, modelagem))
        avisos = list(avisos or [])

        # 1) predição do modelo
        pred = self.ml.prever(medidas, modelagem)
        av, fora = self.ml.avisos_distribuicao(medidas, pred["imc"])
        avisos += av

        # 2) conversão para a marca pedida
        proposto = self._converter(pred["tamanho"], tabela_ref, tabela_marca)

        # 3) verificação de segurança pela regra de folga
        busto, cintura = medidas["busto_circunf"], medidas.get("cintura_circunf")
        corpo = circunferencia_governante(busto, cintura)
        regra = escolher_tamanho(busto, cintura, tabela_marca, modelagem)
        p = FOLGA[modelagem]
        ajustado = False

        if regra.status == "sem_tamanho":
            final = SEM_TAMANHO
            ajustado = proposto != SEM_TAMANHO
            avisos.append(f"Nenhum tamanho {modelagem} da {marca_nome} atende à folga mínima de "
                          f"{p['min']:.0f} cm para estas medidas. Procure numerações estendidas.")
        else:
            linha = _buscar(tabela_marca, proposto)
            folga_prop = _circ(linha) - corpo if linha else None
            if folga_prop is None or folga_prop < p["min"]:
                final, ajustado = regra.tamanho, True
                avisos.append(f"O tamanho sugerido pelo modelo ({proposto}) ficaria justo; "
                              f"ajustado para {final} pela verificação das medidas da peça.")
            elif folga_prop > p["alvo"] + EXCESSO_FOLGA and regra.tamanho != proposto:
                final, ajustado = regra.tamanho, True
                avisos.append(f"O tamanho sugerido pelo modelo ({proposto}) ficaria folgado; "
                              f"ajustado para {final} pela verificação das medidas da peça.")
            else:
                final = proposto

        medidas_peca = None
        linha_final = _buscar(tabela_marca, final)
        if linha_final:
            folga = round(_circ(linha_final) - corpo, 1)
            medidas_peca = {
                "tamanho": final,
                "largura_busto_cm": round(_circ(linha_final) / 2, 1),
                "circunferencia_peca_cm": round(_circ(linha_final), 1),
                "largura_ombro_cm": linha_final.get("largura_ombro"),
                "comprimento_cm": linha_final.get("comprimento_total"),
                "folga_cm": folga,
            }
            if folga > p["alvo"] + EXCESSO_FOLGA:
                avisos.append(f"A peça {final} ficará folgada (folga de {folga:.0f} cm): "
                              f"é o menor tamanho disponível na {marca_nome}.")

        # alternativas convertidas para a marca (sem repetição)
        alts: dict[str, float] = {}
        for a in pred["alternativas"]:
            t = self._converter(a["tamanho"], tabela_ref, tabela_marca)
            alts[t] = max(alts.get(t, 0.0), a["probabilidade"])
        alternativas = [{"tamanho": t, "probabilidade": pr}
                        for t, pr in sorted(alts.items(), key=lambda kv: -kv[1])]

        nivel = self._nivel(pred["confianca"], conf_yolo, origem, fora, ajustado)
        resposta = {
            "tamanho_recomendado": final,
            "marca": marca_nome,
            "tamanho_referencia_modelo": pred["tamanho"],
            "confianca_modelo": round(pred["confianca"], 3),
            "nivel_confianca": nivel,
            "ajustado_pela_regra": ajustado,
            "modelagem": {"usada": modelagem, "origem": origem,
                          "detectada": detectada, "confianca_yolo": conf_yolo},
            "alternativas": alternativas,
            "medidas_peca": medidas_peca,
            "imc": pred["imc"],
            "avisos": avisos,
        }
        try:
            self.banco.registrar({
                "marca": marca_nome, "modelagem": modelagem, "origem": origem,
                "confianca_yolo": conf_yolo, "busto": busto,
                "ombro": medidas["largura_ombro"], "altura": medidas["altura"],
                "peso": medidas["peso_kg"], "cintura": cintura,
                "tamanho_modelo": pred["tamanho"], "tamanho_final": final,
                "confianca": pred["confianca"], "nivel": nivel,
                "ajustado": ajustado, "avisos": avisos})
        except Exception:  # noqa: BLE001 — log nunca derruba a resposta
            pass
        return resposta
