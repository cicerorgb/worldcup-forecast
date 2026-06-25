"""Carrega e normaliza os dados dos arquivos CSV."""
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).parent.parent / "data"

def _detect_encoding(path: Path) -> str:
    """Detecta automaticamente o encoding do arquivo (utf-8 ou latin-1)."""
    try:
        with open(path, encoding="utf-8") as f:
            f.read(4096)
        return "utf-8"
    except UnicodeDecodeError:
        return "latin-1"


def load_rankings() -> dict[str, float]:
    """Retorna {pais: pontos_FIFA} para todos os 48 países."""
    path = DATA_DIR / "pais-rank.csv"
    df = pd.read_csv(path, sep=";", encoding=_detect_encoding(path))
    df.columns = ["pais", "rank"]
    # Rank usa vírgula como separador decimal (ex: "1428,38")
    df["rank"] = df["rank"].astype(str).str.replace(",", ".").astype(float)
    return dict(zip(df["pais"].str.strip(), df["rank"]))


def load_matches() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Retorna (played, future):
      - played: partidas com resultado (gols1/gols2 preenchidos)
      - future: partidas sem resultado (3ª rodada)
    """
    path = DATA_DIR / "fase-de-grupos.csv"
    df = pd.read_csv(path, sep=";", encoding=_detect_encoding(path), header=0)
    df.columns = ["data", "grupo", "time1", "gols1", "x", "gols2", "time2"]
    df = df.drop(columns=["x"])

    # Data vem como "11/06" — adiciona o ano 2026
    df["data"] = pd.to_datetime("2026/" + df["data"].str.strip(), format="%Y/%d/%m")

    df["time1"] = df["time1"].str.strip()
    df["time2"] = df["time2"].str.strip()

    played = df[df["gols1"].notna()].copy().reset_index(drop=True)
    played["gols1"] = played["gols1"].astype(int)
    played["gols2"] = played["gols2"].astype(int)
    future = df[df["gols1"].isna()].copy().reset_index(drop=True)
    return played, future
