"""Loads and normalises input data from CSV files."""

from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).parent.parent / "data"


def _detect_encoding(path: Path) -> str:
    """Returns 'utf-8' if readable, falls back to 'latin-1'."""
    try:
        with open(path, encoding="utf-8") as f:
            f.read(4096)
        return "utf-8"
    except UnicodeDecodeError:
        return "latin-1"


def load_rankings() -> dict[str, float]:
    """Returns {team_name: fifa_points} for all 48 qualified nations."""
    path = DATA_DIR / "pais-rank.csv"
    df = pd.read_csv(path, sep=";", encoding=_detect_encoding(path))
    df.columns = ["country", "points"]
    # Source file uses comma as decimal separator (e.g. "1428,38")
    df["points"] = df["points"].astype(str).str.replace(",", ".").astype(float)
    return dict(zip(df["country"].str.strip(), df["points"]))


def load_matches() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns (played, future):
      - played: matches with known results (goals1 / goals2 filled in)
      - future: matches without results (Round 3 predictions)
    """
    path = DATA_DIR / "fase-de-grupos.csv"
    df = pd.read_csv(path, sep=";", encoding=_detect_encoding(path), header=0)
    df.columns = ["date", "group", "team1", "goals1", "_sep", "goals2", "team2"]
    df = df.drop(columns=["_sep"])

    # Source date format is "DD/MM" — prepend tournament year
    df["date"] = pd.to_datetime("2026/" + df["date"].str.strip(), format="%Y/%d/%m")

    df["team1"] = df["team1"].str.strip()
    df["team2"] = df["team2"].str.strip()

    played = df[df["goals1"].notna()].copy().reset_index(drop=True)
    played["goals1"] = played["goals1"].astype(int)
    played["goals2"] = played["goals2"].astype(int)

    future = df[df["goals1"].isna()].copy().reset_index(drop=True)
    return played, future
