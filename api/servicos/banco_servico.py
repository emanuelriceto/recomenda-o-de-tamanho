"""Acesso ao SQLite: marcas, tabelas de tamanho e log de recomendações."""
import json
import sqlite3
from contextlib import closing
from pathlib import Path


class ServicoBanco:
    def __init__(self, db_path: Path):
        if not Path(db_path).exists():
            raise FileNotFoundError(f"Banco não encontrado: {db_path}. Execute a Fase 1.")
        self.db_path = str(db_path)
        self._criar_log()

    def _conectar(self):
        con = sqlite3.connect(self.db_path)
        con.row_factory = sqlite3.Row
        return con

    def _criar_log(self):
        with closing(self._conectar()) as con:
            con.execute("""
                CREATE TABLE IF NOT EXISTS log_recomendacoes (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    criado_em TEXT DEFAULT CURRENT_TIMESTAMP,
                    marca TEXT, modelagem TEXT, origem_modelagem TEXT, confianca_yolo REAL,
                    busto REAL, ombro REAL, altura REAL, peso REAL, cintura REAL,
                    tamanho_modelo TEXT, tamanho_final TEXT, confianca REAL,
                    nivel TEXT, ajustado INTEGER, avisos TEXT)""")
            con.commit()

    def listar_marcas(self) -> list[str]:
        with closing(self._conectar()) as con:
            return [r["nome"] for r in con.execute("SELECT nome FROM marcas ORDER BY nome")]

    def nome_marca(self, marca: str) -> str | None:
        with closing(self._conectar()) as con:
            r = con.execute("SELECT nome FROM marcas WHERE LOWER(nome) = LOWER(?)",
                            (marca.strip(),)).fetchone()
        return r["nome"] if r else None

    def tabela(self, marca: str, modelagem: str | None = None) -> list[dict]:
        sql = """
            SELECT t.tamanho_label, t.tamanho_ordem, t.modelagem,
                   t.largura_busto_min, t.largura_busto_max, t.largura_ombro,
                   t.comprimento_total, t.busto_corpo_min, t.busto_corpo_max
            FROM tabela_tamanhos t JOIN marcas m ON m.id = t.marca_id
            WHERE LOWER(m.nome) = LOWER(?)"""
        params = [marca]
        if modelagem:
            sql += " AND t.modelagem = ?"
            params.append(modelagem)
        sql += " ORDER BY t.modelagem, t.tamanho_ordem"
        with closing(self._conectar()) as con:
            return [dict(r) for r in con.execute(sql, params)]

    def registrar(self, d: dict):
        with closing(self._conectar()) as con:
            con.execute("""
                INSERT INTO log_recomendacoes (marca, modelagem, origem_modelagem, confianca_yolo,
                    busto, ombro, altura, peso, cintura, tamanho_modelo, tamanho_final,
                    confianca, nivel, ajustado, avisos)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""", (
                d["marca"], d["modelagem"], d["origem"], d.get("confianca_yolo"),
                d["busto"], d["ombro"], d["altura"], d["peso"], d.get("cintura"),
                d["tamanho_modelo"], d["tamanho_final"], d["confianca"], d["nivel"],
                int(d["ajustado"]), json.dumps(d["avisos"], ensure_ascii=False)))
            con.commit()
