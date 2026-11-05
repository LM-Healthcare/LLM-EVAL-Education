"""
Select the items used in the contamination analyses (run from the repository root):
    python Contamination_analysis/select_items.py

Eligible items: text-only questions (Image = No) whose options do not already contain a
"none/all of the other answers" option, whose stem is not negatively worded (TRANNE, NON,
eccetto, falsa, errata), and that are not among the seven questions with defective
administered options (see the main README, "Dataset version").
From the eligible items, 30 per examination year are drawn at random (seed 2026).
For each item the script also fixes, at random:
  - nota_distractor: which distractor (B-E) is replaced by the NOTA option in the
    "NOTA as distractor" control condition;
  - ts_mask: which distractor (B-E) is masked in the option-completion (TS-Guessing) probe.
Output: Contamination_analysis/items.csv
"""
import random
import re

import pandas as pd

SEED, PER_YEAR = 2026, 30
DEFECTIVE = {(2020, 112), (2022, 9), (2022, 41), (2022, 77), (2022, 100), (2022, 119), (2023, 134)}
NOTA_LIKE = re.compile(r"nessuna delle|tutte le (altre )?risposte|tutti i precedenti|le precedenti", re.I)
NEG_STEM = re.compile(r"\bTRANNE\b|\bNON\b|\beccetto\b|\bfalsa\b|\berrata\b")

rng = random.Random(SEED)
rows = []
for y in range(2020, 2025):
    d = pd.read_excel(f"Dataset/{y}_SSM_[checked].xlsx")
    elig = []
    for i, r in d.iterrows():
        q = i + 1
        opts = [str(r[f"Answer {c}"]) for c in "ABCDE"]
        if str(r["Image"]).strip().lower() == "yes" or (y, q) in DEFECTIVE:
            continue
        if any(NOTA_LIKE.search(o) for o in opts) or NEG_STEM.search(str(r["Question"])):
            continue
        elig.append((q, r))
    for q, r in sorted(rng.sample(elig, PER_YEAR), key=lambda t: t[0]):
        rows.append({"year": y, "question_idx": q, "question_code": r["Question Code"],
                     "question_type": r["Question type"], "question": r["Question"],
                     "correct": r["Answer A"], "d1": r["Answer B"], "d2": r["Answer C"],
                     "d3": r["Answer D"], "d4": r["Answer E"],
                     "nota_distractor": rng.randint(1, 4), "ts_mask": rng.randint(1, 4)})
    print(y, "eligible", len(elig))
out = pd.DataFrame(rows)
out.to_csv("Contamination_analysis/items.csv", index=False)
print("written Contamination_analysis/items.csv:", len(out), "items;",
      out.question_type.value_counts().to_dict())
