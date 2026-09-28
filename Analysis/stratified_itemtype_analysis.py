"""
Exploratory stratified analysis by item type (knowledge-based vs case-based)
and image dependence, computed from the public repository files.

Run from the root of LLM-EVAL-Education:
    python stratified_itemtype_analysis.py

Inputs : Dataset/{year}_SSM_[checked].xlsx  (Question type, Image)
         Results/*/Consistency/consistency_*.xlsx  (1 = correct, letter = wrong, ND = unparsed)
Output : stratified_results.csv

Unit of analysis = item (question). For each model, item accuracy = share of the
50 runs answered correctly. Differences are reported with 95% bootstrap CIs
resampling items (2,000 resamples), which respects the fact that the 50 runs
are not independent observations.
KB vs CB is computed on text-only items (image items are 32/35 case-based,
so including them would confound the comparison).
NOTE: ND cells are counted as incorrect here; align with the final manual-resolution audit.
"""
import glob
import re
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
rng = np.random.default_rng(0)
EXCLUDE = {"Llama32_1B_Instruct", "Llama32_3B_Instruct"}  # in repo, not in the paper


def boot_diff(a, b, n=2000):
    d = [rng.choice(a, len(a)).mean() - rng.choice(b, len(b)).mean() for _ in range(n)]
    return np.percentile(d, [2.5, 97.5])


meta = pd.concat(
    [pd.read_excel(f"Dataset/{y}_SSM_[checked].xlsx").assign(year=y, q=list(range(1, 141)))
     for y in range(2020, 2025)],
    ignore_index=True,
)
meta["Image"] = meta["Image"].replace({"Si": "Yes"})  # 2020 file still uses "Si"

rows = []
for f in sorted(glob.glob("Results/*/Consistency/*.xlsx")):
    name = re.sub(r"consistency_|\.xlsx", "", f.split("/")[-1])
    if name in EXCLUDE:
        continue
    c = pd.read_excel(f)
    yrs = c.RUN_EXAM.str[:4].astype(int)
    parts = []
    for y in range(2020, 2025):
        g = c[yrs == y][[f"Domanda_{i}" for i in range(1, 141)]].astype(str)
        parts.append(pd.DataFrame({"year": y, "q": list(range(1, 141)), "acc": (g == "1").mean().values}))
    a = pd.concat(parts, ignore_index=True).merge(meta[["year", "q", "Question type", "Image"]], on=["year", "q"])

    txt = a[a.Image == "No"]
    kb = txt[txt["Question type"] == "knowledge-based"].acc.values
    cb = txt[txt["Question type"] == "case-based"].acc.values
    im = a[a.Image == "Yes"].acc.values
    ni = a[a.Image == "No"].acc.values
    lo1, hi1 = boot_diff(kb, cb)
    lo2, hi2 = boot_diff(ni, im)
    rows.append({
        "model": name,
        "acc_KB_textonly": 100 * kb.mean(), "acc_CB_textonly": 100 * cb.mean(),
        "diff_KB_minus_CB": 100 * (kb.mean() - cb.mean()), "ci_lo": 100 * lo1, "ci_hi": 100 * hi1,
        "acc_text_items": 100 * ni.mean(), "acc_image_items": 100 * im.mean(),
        "diff_text_minus_image": 100 * (ni.mean() - im.mean()), "ci_lo_img": 100 * lo2, "ci_hi_img": 100 * hi2,
        "n_KB": len(kb), "n_CB": len(cb), "n_img": len(im),
    })

out = pd.DataFrame(rows).round(1)
out.to_csv("stratified_results.csv", index=False)
print(out.to_string())
