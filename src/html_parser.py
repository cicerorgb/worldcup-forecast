"""
Parser for international match history exported from the FIFA website (HTML).

Each HTML file contains all A-international matches for one national team.
The parser:
  1. Auto-detects which team owns the file (most frequent name in the table)
  2. Extracts date, goals scored, goals conceded, and tournament name
  3. Assigns a tournament relevance weight:
       World Cup > Qualifiers > Confederation Cup > Friendly
"""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

import pandas as pd
from bs4 import BeautifulSoup

DATA_DIR = Path(__file__).parent.parent / "data"

# Portuguese month abbreviations as they appear in FIFA's HTML export
_PT_MONTH_ABBR: dict[str, int] = {
    "jan": 1, "fev": 2, "mar": 3, "abr": 4, "mai": 5, "jun": 6,
    "jul": 7, "ago": 8, "set": 9, "out": 10, "nov": 11, "dez": 12,
}

# Maps HTML filename stem → canonical team name used in the CSV schedule
FILE_TO_TEAM: dict[str, str] = {
    "africa-do-sul":        "África do Sul",
    "alemanha":             "Alemanha",
    "arabia-saudita":       "Arábia Saudita",
    "argelia":              "Argélia",
    "argentina":            "Argentina",
    "australia":            "Austrália",
    "austria":              "Áustria",
    "belgica":              "Bélgica",
    "bosnia-e-herzegovina": "Bósnia-Herzegovina",
    "brasil":               "Brasil",
    "cabo-verde":           "Cabo Verde",
    "canada":               "Canadá",
    "colombia":             "Colômbia",
    "congo":                "Congo (RD)",
    "coreia-do-sul":        "Coreia do Sul",
    "costa-do-marfim":      "Costa do Marfim",
    "croacia":              "Croácia",
    "curacao":              "Curaçao",
    "egito":                "Egito",
    "equador":              "Equador",
    "escocia":              "Escócia",
    "espanha":              "Espanha",
    "estados-unidos":       "Estados Unidos",
    "franca":               "França",
    "gana":                 "Gana",
    "haiti":                "Haiti",
    "holanda":              "Holanda",
    "inglaterra":           "Inglaterra",
    "ira":                  "Irã",
    "iraque":               "Iraque",
    "japao":                "Japão",
    "jordania":             "Jordânia",
    "marrocos":             "Marrocos",
    "mexico":               "México",
    "noruega":              "Noruega",
    "nova-zelandia":        "Nova Zelândia",
    "panama":               "Panamá",
    "paraguai":             "Paraguai",
    "portugal":             "Portugal",
    "qatar":                "Qatar",
    "republica-tcheca":     "República Tcheca",
    "senegal":              "Senegal",
    "suecia":               "Suécia",
    "suica":                "Suíça",
    "tunisia":              "Tunísia",
    "turquia":              "Turquia",
    "uruguai":              "Uruguai",
    "uzbequistao":          "Uzbequistão",
}


def _tournament_weight(tournament: str) -> float:
    """Returns a relevance weight for a tournament name (higher = more competitive)."""
    t = tournament.lower()
    if "copa do mundo" in t:
        return 3.0
    if "eliminatórias" in t or "eliminatorias" in t or "qualifying" in t:
        return 2.0
    if "africa cup" in t or " can " in t or "nations cup" in t or "euro" in t or "copa america" in t:
        return 1.5
    if "friendly" in t or "amistoso" in t or "series" in t:
        return 0.5
    return 0.8  # regional tournaments (COSAFA, CHAN, etc.)


def _parse_date(raw: str) -> pd.Timestamp | None:
    parts = raw.strip().split()
    if len(parts) == 3:
        try:
            day, mon_str, year = parts
            month = _PT_MONTH_ABBR.get(mon_str.lower())
            if month:
                return pd.Timestamp(int(year), month, int(day))
        except (ValueError, TypeError):
            pass
    return None


def _parse_file(path: Path, canonical: str) -> list[dict]:
    """Extracts match records from a single HTML file."""
    with open(path, encoding="utf-8", errors="replace") as f:
        soup = BeautifulSoup(f, "html.parser")

    rows = soup.find_all("tr", class_=re.compile(r"row-(even|odd)"))

    raw_rows: list[dict] = []
    all_names: list[str] = []

    for row in rows:
        tds = row.find_all("td", class_="scrollable-column")
        if len(tds) < 5:
            continue
        teams  = tds[1].find_all("span", class_="dsk-description-info")
        scores = tds[1].find_all("span", class_="scoring-value")
        if len(teams) < 2 or len(scores) < 2:
            continue
        t1 = teams[0].get_text(strip=True)
        t2 = teams[1].get_text(strip=True)
        all_names += [t1, t2]
        raw_rows.append({
            "date_raw":   tds[0].get_text(strip=True),
            "t1": t1, "t2": t2,
            "s1": int(scores[0].get_text(strip=True)),
            "s2": int(scores[1].get_text(strip=True)),
            "tournament": tds[2].get_text(strip=True),
        })

    if not all_names:
        return []

    # The team that appears most often is the owner of the file
    self_name = Counter(all_names).most_common(1)[0][0]

    records: list[dict] = []
    for r in raw_rows:
        dt = _parse_date(r["date_raw"])
        if dt is None:
            continue
        if r["t1"] == self_name:
            goals_for, goals_against = r["s1"], r["s2"]
        elif r["t2"] == self_name:
            goals_for, goals_against = r["s2"], r["s1"]
        else:
            continue
        records.append({
            "date":               dt,
            "team":               canonical,
            "goals_for":          goals_for,
            "goals_against":      goals_against,
            "tournament":         r["tournament"],
            "tournament_weight":  _tournament_weight(r["tournament"]),
        })
    return records


def load_historical(year_from: int = 2020) -> pd.DataFrame:
    """
    Reads all HTML files in data/ and returns a DataFrame with the full
    match history of all 48 nations since `year_from`.

    Columns: date, team, goals_for, goals_against, tournament, tournament_weight
    """
    all_records: list[dict] = []
    missing: list[str] = []

    for path in sorted(DATA_DIR.glob("*.html")):
        canonical = FILE_TO_TEAM.get(path.stem)
        if canonical is None:
            missing.append(path.name)
            continue
        all_records.extend(_parse_file(path, canonical))

    if missing:
        print(f"  [WARNING] HTML files without a team mapping were skipped: {missing}")

    if not all_records:
        return pd.DataFrame()

    df = pd.DataFrame(all_records)
    df = df[df["date"].dt.year >= year_from].copy()
    return df.sort_values("date").reset_index(drop=True)
