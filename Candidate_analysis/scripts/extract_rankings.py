"""
Build the anonymised candidate dataset from the official SSM national rankings (2020-2024).

The original ranking files contain candidates' names and dates of birth and are NOT
distributed. They were downloaded from the Universitaly portal (MUR) when each ranking was
published and are no longer publicly available there. To rebuild the dataset, place them in
Candidate_analysis/original_rankings/ with the following names and run, from Candidate_analysis/:

    python scripts/extract_rankings.py

    Graduatoria SSM2020 - 23756.pdf     (PDF)
    Graduatoria SSM2021 - 19449.xlsx    (spreadsheet as published)
    Graduatoria SSM2022 - 15869.pdf     (PDF)
    Graduatoria SSM2023 - 14036.xlsx    (spreadsheet as published)
    Graduatoria SSM2024 - 14121.pdf     (PDF)

Output: data/candidate_scores_2020_2024.csv with one row per ranked candidate and ONLY the
following columns (no names, no dates of birth, no assignment information):
    year, rank_order, total_score, test_score, titles_score
rank_order is the row order of the published ranking (1 = first).

Checks performed: total_score == test_score + titles_score for every candidate; for the
rankings that report positions (2022, 2024) the script lists missing positions.
Requirements: pdfplumber, pandas, openpyxl.
"""
import collections
import os
import re

import pandas as pd
import pdfplumber

SRC = "original_rankings"
OUT = os.path.join("data", "candidate_scores_2020_2024.csv")
NUM = r"-?\d+(?:[.,]\d+)?"


def fl(s):
    return float(s.replace(",", "."))


def column_values(page, x0, x1):
    """Numbers in a vertical column of the page, one per text line (robust to name overflow)."""
    rows = collections.defaultdict(list)
    for c in page.chars:
        if x0 <= c["x0"] < x1 and re.fullmatch(r"[\d,\-]", c["text"]):
            rows[round(c["top"])].append(c)
    out = []
    for top in sorted(rows):
        s = "".join(ch["text"] for ch in sorted(rows[top], key=lambda c: c["x0"]))
        if re.fullmatch(r"-?\d+(,\d+)?", s):
            out.append((top, fl(s)))
    return out


def ssm2020(path):
    """2020 PDF: pages 1-422 list name, total score and test score; pages 423-844 list the
    title points in the same order; the last page is a legend."""
    tot, test, tit = [], [], []
    with pdfplumber.open(path) as pdf:
        n_data = sum(1 for p in pdf.pages if re.search(r"\(\d{2}/\d{2}/\d{4}\)", p.extract_text() or ""))
        for i, p in enumerate(pdf.pages):
            if i < n_data:
                a, b = column_values(p, 380, 432), column_values(p, 475, 530)
                assert len(a) == len(b) and all(abs(x[0] - y[0]) <= 2 for x, y in zip(a, b)), f"page {i + 1}"
                tot += [x[1] for x in a]
                test += [y[1] for y in b]
            elif i < 2 * n_data:
                tit += [x[1] for x in column_values(p, 100, 140)]
    assert len(tot) == len(tit), (len(tot), len(tit))
    return pd.DataFrame({"total_score": tot, "test_score": test, "titles_score": tit})


def ssm_positions(path):
    """2022 and 2024 PDFs: one line per candidate: position, name (date of birth), total, test, titles."""
    rx = re.compile(rf"^(\d+)\s(.*?)\s({NUM})\s({NUM})\s({NUM})(\s.*)?$")
    rows = []
    with pdfplumber.open(path) as pdf:
        for p in pdf.pages:
            for line in (p.extract_text() or "").split("\n"):
                m = rx.match(line.strip())
                if m and re.search(r"\d{2}/\d{2}/\d{4}", m[2]):
                    rows.append((int(m[1]), fl(m[3]), fl(m[4]), fl(m[5])))
    d = pd.DataFrame(rows, columns=["position", "total_score", "test_score", "titles_score"])
    missing = sorted(set(range(1, d.position.max() + 1)) - set(d.position))
    return d, missing


def from_xlsx(path, total, test, titles):
    d = pd.read_excel(path)
    d = d[d["Posizione"].notna()]          # the last row of the published file holds column means
    return pd.DataFrame({"total_score": d[total].astype(float), "test_score": d[test].astype(float),
                         "titles_score": pd.to_numeric(d[titles]).astype(float)})


parts = []
d = ssm2020(os.path.join(SRC, "Graduatoria SSM2020 - 23756.pdf")); d["year"] = 2020; parts.append(d)
d = from_xlsx(os.path.join(SRC, "Graduatoria SSM2021 - 19449.xlsx"), "Punteggio", "Punteggio senza CV", "CV")
d["year"] = 2021; parts.append(d)
for year, fn in [(2022, "Graduatoria SSM2022 - 15869.pdf"), (2024, "Graduatoria SSM2024 - 14121.pdf")]:
    d, missing = ssm_positions(os.path.join(SRC, fn))
    ties = [m for m in missing if m - 1 in set(d.position[d.position.duplicated()])]
    gaps = [m for m in missing if m not in ties]
    print(f"{year}: {len(d)} candidates; positions absent from the file (not due to ties): "
          f"{len(gaps)}" + (f" ({gaps[0]}-{gaps[-1]})" if gaps else ""))
    d["year"] = year; parts.append(d.drop(columns="position"))
d = from_xlsx(os.path.join(SRC, "Graduatoria SSM2023 - 14036.xlsx"), "Punteggio Totale", "Punteggio Prova",
              "Punteggio Titoli")
d["year"] = 2023; parts.append(d)

out = pd.concat(parts, ignore_index=True)
out["rank_order"] = out.groupby("year").cumcount() + 1
out = out.sort_values(["year", "rank_order"])[["year", "rank_order", "total_score", "test_score", "titles_score"]]
bad = ((out.total_score - out.test_score - out.titles_score).abs() > 1e-6).sum()
print("rows per year:", out.groupby("year").size().to_dict(), "| total != test + titles:", bad)
os.makedirs("data", exist_ok=True)
out.to_csv(OUT, index=False)
print("written", OUT)
