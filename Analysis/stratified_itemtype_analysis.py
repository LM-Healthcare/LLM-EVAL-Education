"""
Exploratory item-type analysis (ERQ): knowledge-based vs case-based items and
image-dependent vs text-only items.

Run from the root of the LLM-EVAL-Education repository:
    python Analysis/stratified_itemtype_analysis.py

Inputs : Dataset/{year}_SSM_[checked].xlsx          (Question type, Image)
         Results/*/Consistency/consistency_*.xlsx   (1 = correct, letter = wrong, ND = unresolved)
Output : Analysis/itemtype_results_by_model.csv
         Analysis/itemtype_results_by_group.csv

Method
- Unit of analysis = question. For each model, item accuracy = share of the
  50 repetitions answered correctly.
- Contrast 1: knowledge-based vs case-based, restricted to text-only items
  (image-dependent items are almost all case-based and would confound it).
- Contrast 2: text-only vs image-dependent items (all item types).
- Differences in mean item accuracy with 95% percentile bootstrap CIs,
  resampling questions within each category (B = 4,000; seed = 2026).
- Group rows use item accuracy averaged across the models of the group.
- Cells coded "ND" are counted as incorrect.
"""
import glob
import os
import re
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
B = 4000
SEED = 2026
OUT = "Analysis"

GROUP = {"Closed Models": "Proprietary", "Open Models": "Open-source", "Quantized models": "Quantized"}
LABEL = {
    "claude": "Claude 4.5 Sonnet", "deepseek": "DeepSeek V3.2", "gpt-52-2025-12-11": "GPT 5.2",
    "grok": "Grok 4.1 Fast Reasoning", "mistral": "Mistral Large 3", "qwen": "Qwen 3 Max",
    "Qwen3_1_7B": "Qwen3-1.7B", "Ministral_3B_Instruct": "Ministral-3B", "Qwen3_4B_Instruct": "Qwen3-4B-Instruct",
    "MedGemma_4B_IT": "MedGemma 1.5-4B", "Qwen3_8B": "Qwen3-8B", "Meditron3_8B_FP16": "Meditron3-8B",
    "MedGemma_4B_Q8_0": "MedGemma 1.5-4B-Q8_0", "MedGemma_4B_Q6_K": "MedGemma 1.5-4B-Q6_K",
    "MedGemma_4B_Q4_K_M": "MedGemma 1.5-4B-Q4_K_M", "Meditron3_8B_Q8_0": "Meditron3-8B-Q8_0",
    "Meditron3_8B_Q6_K": "Meditron3-8B-Q6_K", "Meditron3_8B_Q4_K_M": "Meditron3-8B-Q4_K_M",
}
ORDER = list(LABEL)

meta = pd.concat(
    [pd.read_excel(f"Dataset/{y}_SSM_[checked].xlsx").assign(year=y) for y in range(2020, 2025)],
    ignore_index=True,
)
meta["Image"] = meta["Image"].replace({"Si": "Yes"})
img = (meta["Image"] == "Yes").values
kb = ((meta["Question type"] == "knowledge-based").values) & ~img
cb = ((meta["Question type"] == "case-based").values) & ~img
txt = ~img

acc, grp = {}, {}
for f in glob.glob("Results/*/Consistency/*.xlsx"):
    key = re.sub(r"consistency_|\.xlsx", "", os.path.basename(f))
    if key not in LABEL:
        continue
    c = pd.read_excel(f)
    yrs = c.RUN_EXAM.str[:4].astype(int)
    cols = [f"Domanda_{i}" for i in range(1, 141)]
    acc[key] = np.concatenate([(c.loc[yrs == y, cols].astype(str) == "1").mean().values for y in range(2020, 2025)])
    grp[key] = GROUP[f.split(os.sep)[1] if os.sep in f else f.split("/")[1]]


def contrast(x, a, b, rng):
    d = x[a].mean() - x[b].mean()
    xa, xb = x[a], x[b]
    boot = [rng.choice(xa, len(xa)).mean() - rng.choice(xb, len(xb)).mean() for _ in range(B)]
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return 100 * xa.mean(), 100 * xb.mean(), 100 * d, 100 * lo, 100 * hi


def row(name, group, x, rng):
    k, c_, d1, l1, h1 = contrast(x, kb, cb, rng)
    t, i, d2, l2, h2 = contrast(x, txt, img, rng)
    return {"Model": name, "Group": group,
            "KB acc (%)": k, "CB acc (%)": c_, "KB-CB (pp)": d1, "KB-CB CI low": l1, "KB-CB CI high": h1,
            "Text-only acc (%)": t, "Image acc (%)": i, "Text-Image (pp)": d2, "Text-Image CI low": l2, "Text-Image CI high": h2}


rng = np.random.default_rng(SEED)
by_model = pd.DataFrame([row(LABEL[k], grp[k], acc[k], rng) for k in ORDER]).round(1)

rows = []
for g in ["Proprietary", "Open-source", "Quantized"]:
    ks = [k for k in ORDER if grp[k] == g]
    rows.append(row(f"{g} (mean of {len(ks)} models)", g, np.mean([acc[k] for k in ks], axis=0), rng))
rows.append(row("All models (mean of 18)", "All", np.mean([acc[k] for k in ORDER], axis=0), rng))
by_group = pd.DataFrame(rows).round(1)

os.makedirs(OUT, exist_ok=True)
by_model.to_csv(f"{OUT}/itemtype_results_by_model.csv", index=False)
by_group.to_csv(f"{OUT}/itemtype_results_by_group.csv", index=False)
print(f"Items: knowledge-based text-only = {kb.sum()}, case-based text-only = {cb.sum()}, "
      f"text-only = {txt.sum()}, image-dependent = {img.sum()}")
pd.set_option("display.width", 250)
print(by_model.to_string(index=False))
print(by_group.to_string(index=False))
