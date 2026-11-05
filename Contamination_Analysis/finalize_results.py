"""
Align the result files with items.csv (run from the repository root, after all runs are complete):
    python Contamination_analysis/finalize_results.py

For every file in results/ (perturbation_*.jsonl, ts_guessing_*.jsonl, tester_raw_*.jsonl) it keeps
only the responses to the items listed in items.csv, removes duplicates (keeping the last response),
and sorts the lines (perturbation: run, year, question, condition; TS-Guessing: year, question).
The original files are copied first to results/_before_finalize/. It also reports, for each model,
whether every item, condition and run is complete.
"""
import glob
import json
import os
import shutil

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
COND = {"original": 0, "nota_correct": 1, "nota_distractor": 2}

items = pd.read_csv(os.path.join(HERE, "items.csv"))
keys = set(zip(items.year, items.question_idx))
stems = {(int(r.year), str(r.question).strip()) for r in items.itertuples()}
backup = os.path.join(RES, "_before_finalize")
os.makedirs(backup, exist_ok=True)


def read(path):
    with open(path, encoding="utf-8") as fh:
        return [json.loads(l) for l in fh if l.strip()]


def write(path, recs):
    with open(path, "w", encoding="utf-8") as fh:
        for r in recs:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")


for path in sorted(glob.glob(os.path.join(RES, "*.jsonl"))):
    name = os.path.basename(path)
    if name.startswith("openrouter_calls"):
        continue
    shutil.copy2(path, os.path.join(backup, name))
    recs = read(path)
    if name.startswith("perturbation_"):
        uniq = {(r["condition"], r["run"], r["year"], r["question_idx"]): r for r in recs
                if (r["year"], r["question_idx"]) in keys}
        out = sorted(uniq.values(), key=lambda r: (r["run"], r["year"], r["question_idx"], COND[r["condition"]]))
        runs = sorted({r["run"] for r in out})
        conds = sorted({r["condition"] for r in out}, key=COND.get)
        expected = len(keys) * len(runs) * len(conds)
        status = "complete" if len(out) == expected else f"INCOMPLETE ({len(out)} of {expected})"
        print(f"{name:<52} {len(recs):>5} -> {len(out):>5} lines; runs {runs}; {conds}; {status}")
    elif name.startswith("ts_guessing_"):
        uniq = {(r["year"], r["question_idx"]): r for r in recs if (r["year"], r["question_idx"]) in keys}
        out = sorted(uniq.values(), key=lambda r: (r["year"], r["question_idx"]))
        status = "complete" if len(out) == len(keys) else f"INCOMPLETE ({len(out)} of {len(keys)})"
        print(f"{name:<52} {len(recs):>5} -> {len(out):>5} lines; {status}")
    else:                                    # tester_raw_*: matched on year and question text
        out = [r for r in recs if (int(r.get("year", 0)), str(r.get("question", "")).strip()) in stems]
        print(f"{name:<52} {len(recs):>5} -> {len(out):>5} lines")
    write(path, out)
print(f"originals copied to {backup}")
