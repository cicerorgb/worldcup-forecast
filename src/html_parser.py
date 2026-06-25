"""
Parser do histórico de partidas internacionais exportadas do site FIFA (HTML).

Para cada seleção, o arquivo HTML contém todas as suas partidas A internacionais.
O parser:
  1. Detecta automaticamente qual time é o "dono" do arquivo (nome mais frequente)
  2. Extrai data, gols marcados, gols sofridos e torneio
  3. Atribui peso ao torneio (Copa do Mundo > Eliminatórias > Conf. > Amistoso)
"""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

import pandas as pd
from bs4 import BeautifulSoup

DATA_DIR = Path(__file__).parent.parent / "data"

_MESES: dict[str, int] = {
    "jan": 1, "fev": 2, "mar": 3, "abr": 4, "mai": 5, "jun": 6,
    "jul": 7, "ago": 8, "set": 9, "out": 10, "nov": 11, "dez": 12,
}

# Mapeamento: nome do arquivo (sem extensão) → nome canônico usado no CSV
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


def _tournament_weight(torneio: str) -> float:
    """Relevância do torneio para estimativa de força da equipe."""
    t = torneio.lower()
    if "copa do mundo" in t:
        return 3.0
    if "eliminatórias" in t or "eliminatorias" in t or "qualifying" in t:
        return 2.0
    if "africa cup" in t or " can " in t or "nations cup" in t or "euro" in t or "copa america" in t:
        return 1.5
    if "friendly" in t or "amistoso" in t or "series" in t:
        return 0.5
    return 0.8   # torneios regionais (COSAFA, CHAN, etc.)


def _parse_date(raw: str) -> pd.Timestamp | None:
    parts = raw.strip().split()
    if len(parts) == 3:
        try:
            day, mon_str, year = parts
            month = _MESES.get(mon_str.lower())
            if month:
                return pd.Timestamp(int(year), month, int(day))
        except (ValueError, TypeError):
            pass
    return None


def _parse_file(path: Path, canonical: str) -> list[dict]:
    """Extrai registros de partidas de um único arquivo HTML."""
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
            "date_raw": tds[0].get_text(strip=True),
            "t1": t1, "t2": t2,
            "s1": int(scores[0].get_text(strip=True)),
            "s2": int(scores[1].get_text(strip=True)),
            "torneio": tds[2].get_text(strip=True),
        })

    if not all_names:
        return []

    # O time com mais aparições é o dono do arquivo
    self_name = Counter(all_names).most_common(1)[0][0]

    records: list[dict] = []
    for r in raw_rows:
        dt = _parse_date(r["date_raw"])
        if dt is None:
            continue
        if r["t1"] == self_name:
            gf, ga = r["s1"], r["s2"]
        elif r["t2"] == self_name:
            gf, ga = r["s2"], r["s1"]
        else:
            continue
        records.append({
            "data":           dt,
            "team":           canonical,
            "gols_for":       gf,
            "gols_against":   ga,
            "torneio":        r["torneio"],
            "torneio_weight": _tournament_weight(r["torneio"]),
        })
    return records


def load_historical(year_from: int = 2020) -> pd.DataFrame:
    """
    Lê todos os arquivos HTML em data/ e retorna DataFrame com o histórico
    completo das 48 seleções desde `year_from`.

    Colunas: data, team, gols_for, gols_against, torneio, torneio_weight
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
        print(f"  [AVISO] HTMLs sem mapeamento ignorados: {missing}")

    if not all_records:
        return pd.DataFrame()

    df = pd.DataFrame(all_records)
    df = df[df["data"].dt.year >= year_from].copy()
    return df.sort_values("data").reset_index(drop=True)
