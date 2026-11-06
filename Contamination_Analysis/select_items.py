"""
Select the items used in the contamination analyses (run from the repository root):
    python Contamination_Analysis/select_items.py

Eligible items: text-only questions (Image = No) whose options do not already contain a
"none/all of the other answers" option, whose stem is not negatively worded (the question asks
for the exception or for the false statement: TRANNE, ECCETTO, FALSA, ERRATA, SBAGLIATA, NON in
capitals), and that are not among the seven questions with defective administered options
(see the main README, "Dataset version").

30 items per examination year are drawn at random (seed 2026). Items of the first draw whose stem
is negatively worded are replaced by items of the same year drawn at random from the remaining
eligible items (seed "2026-replacement"). For each item the script also fixes, at random:
  - nota_distractor: which distractor (B-E) is replaced by the NOTA option in the
    "NOTA as distractor" control condition;
  - ts_mask: which distractor (B-E) is masked in the option-completion (TS-Guessing) probe.
Output: Contamination_Analysis/items.csv (sorted by year and question).
"""
import random
import re

import pandas as pd

SEED, PER_YEAR = 2026, 30
DEFECTIVE = {(2020, 112), (2022, 9), (2022, 41), (2022, 77), (2022, 100), (2022, 119), (2023, 134)}
NOTA_LIKE = re.compile(r"nessuna delle|tutte le (altre )?risposte|tutti i precedenti|le precedenti", re.I)
FIRST_DRAW_NEG = re.compile(r"\bTRANNE\b|\bNON\b|\beccetto\b|\bfalsa\b|\berrata\b")       # first draw
NEG_STEM = re.compile(r"\btranne\b|\beccetto\b|\bfalsa\b|\berrata\b|\bsbagliata\b", re.I)
NEG_CAPS = re.compile(r"\bNON\b")


def negative(stem):
    return bool(NEG_STEM.search(stem) or NEG_CAPS.search(stem))


def row(y, q, r, nota, mask):
    return {"year": y, "question_idx": q, "question_code": r["Question Code"],
            "question_type": r["Question type"], "question": r["Question"],
            "correct": r["Answer A"], "d1": r["Answer B"], "d2": r["Answer C"],
            "d3": r["Answer D"], "d4": r["Answer E"], "nota_distractor": nota, "ts_mask": mask}


rng, rng_rep = random.Random(SEED), random.Random(f"{SEED}-replacement")
rows = []
for y in range(2020, 2025):
    d = pd.read_excel(f"Dataset/{y}_SSM_[checked].xlsx")
    base = []
    for i, r in d.iterrows():
        q = i + 1
        opts = [str(r[f"Answer {c}"]) for c in "ABCDE"]
        if str(r["Image"]).strip().lower() == "yes" or (y, q) in DEFECTIVE or any(NOTA_LIKE.search(o) for o in opts):
            continue
        base.append((q, r))
    first = [(q, r) for q, r in base if not FIRST_DRAW_NEG.search(str(r["Question"]))]
    drawn = []
    for q, r in sorted(rng.sample(first, PER_YEAR), key=lambda t: t[0]):
        drawn.append(row(y, q, r, rng.randint(1, 4), rng.randint(1, 4)))
    keep = [x for x in drawn if not negative(str(x["question"]))]
    taken = {x["question_idx"] for x in drawn}
    pool = [(q, r) for q, r in base if q not in taken and not negative(str(r["Question"]))]
    for q, r in sorted(rng_rep.sample(pool, PER_YEAR - len(keep)), key=lambda t: t[0]):
        keep.append(row(y, q, r, rng_rep.randint(1, 4), rng_rep.randint(1, 4)))
    n_replaced = PER_YEAR - sum(not negative(str(x["question"])) for x in drawn)
    rows += sorted(keep, key=lambda x: x["question_idx"])
    print(y, "replaced", n_replaced)
out = pd.DataFrame(rows)
out.to_csv("Contamination_Analysis/items.csv", index=False)
print("written Contamination_Analysis/items.csv:", len(out), "items;", out.question_type.value_counts().to_dict())
