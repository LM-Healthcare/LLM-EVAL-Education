"""
Cross-check between the final dataset (Results/*/Consistency/consistency_*.xlsx) and the
raw model outputs (Results/*/Raw_Responses/raw_responses_*.jsonl).

Run from the repository root, after `git lfs pull`:
    python Analysis/check_final_dataset_vs_raw.py

For every model, run and question it compares:
  - consistency cell: "1" = correct, a letter = incorrect
  - raw file: extracted_letter == correct_answer  (last entry kept for resumed runs)

Outputs in Analysis/audit_output/: final_dataset_vs_raw.csv (every cell where the final dataset
and the raw file disagree) and final_dataset_vs_raw_summary.csv (counts per model).
See the main README, section "Answer extraction audit". Requirements: pandas, openpyxl.
"""
import glob
import json
import os
from collections import Counter

import pandas as pd

PAIRS = {
    "claude": "claude", "deepseek": "deepseek", "gpt-52-2025-12-11": "gpt", "grok": "grok",
    "mistral": "mistral", "qwen": "qwen",
    "MedGemma_4B_IT": "MedGemma_4B_IT", "Meditron3_8B_FP16": "Meditron3_8B_FP16",
    "Ministral_3B_Instruct": "Ministral_3B_Instruct", "Qwen3_1_7B": "Qwen3_1_7B",
    "Qwen3_4B_Instruct": "Qwen3_4B_Instruct", "Qwen3_8B": "Qwen3_8B",
    "MedGemma_4B_Q4_K_M": "MedGemma_4B_Q4_K_M", "MedGemma_4B_Q6_K": "MedGemma_4B_Q6_K",
    "MedGemma_4B_Q8_0": "MedGemma_4B_Q8_0", "Meditron3_8B_Q4_K_M": "Meditron3_8B_Q4_K_M",
    "Meditron3_8B_Q6_K": "Meditron3_8B_Q6_K", "Meditron3_8B_Q8_0": "Meditron3_8B_Q8_0",
}
rows, summary = [], []
for cfile in sorted(glob.glob("Results/*/Consistency/consistency_*.xlsx")):
    key = os.path.basename(cfile)[len("consistency_"):-5]
    folder = os.path.dirname(os.path.dirname(cfile))
    rfile = os.path.join(folder, "Raw_Responses", f"raw_responses_{PAIRS[key]}.jsonl")
    if not os.path.exists(rfile) or os.path.getsize(rfile) < 1000:
        print(f"[skip] {key}: raw file missing or Git LFS pointer ({rfile})")
        continue
    raw = {}
    with open(rfile, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                e = json.loads(line)
                raw[(int(e["year"]), int(e["run"]), int(e["question_idx"]))] = e
    qidx = sorted(q for _, _, q in raw)
    offset = 1 if qidx[0] == 0 else 0          # raw question_idx 0-based -> Domanda_ 1-based
    cons = pd.read_excel(cfile)
    cnt = Counter()
    for _, r in cons.iterrows():
        y, run = int(r["RUN_EXAM"][:4]), int(str(r["RUN_EXAM"]).split("_")[-1])
        for q in range(1, 141):
            cell = r.get(f"Domanda_{q}")
            cell = "" if pd.isna(cell) else str(cell).strip()
            c_ok = cell == "1"
            e = raw.get((y, run, q - offset))
            if e is None:
                cnt["missing_in_raw"] += 1
                rows.append([key, y, run, q, cell, "", "", "missing in raw"])
                continue
            r_ok = e.get("extracted_letter") == e.get("correct_answer")
            cnt["cells"] += 1
            cnt["cons_correct"] += c_ok
            cnt["raw_correct"] += r_ok
            if cell == "":
                cnt["empty_in_consistency"] += 1
                rows.append([key, y, run, q, cell, e.get("extracted_letter"), e.get("correct_answer"), "empty in consistency"])
            elif c_ok != r_ok:
                cnt["correctness_differs"] += 1
                rows.append([key, y, run, q, cell, e.get("extracted_letter"), e.get("correct_answer"),
                             "consistency correct, raw incorrect" if c_ok else "consistency incorrect, raw correct"])
    summary.append({"model": key, **cnt})
    print(f"{key:<24} cells {cnt['cells']:>6}  correct: consistency {cnt['cons_correct']:>6}  raw {cnt['raw_correct']:>6}  "
          f"empty {cnt['empty_in_consistency']:>3}  differ {cnt['correctness_differs']:>3}  missing_in_raw {cnt['missing_in_raw']:>3}")

pd.DataFrame(rows, columns=["model", "year", "run", "question", "consistency_cell", "raw_extracted_letter",
                            "raw_correct_answer", "issue"]).to_csv("Analysis/audit_output/final_dataset_vs_raw.csv", index=False)
pd.DataFrame(summary).to_csv("Analysis/audit_output/final_dataset_vs_raw_summary.csv", index=False)
print("written Analysis/audit_output/final_dataset_vs_raw.csv and final_dataset_vs_raw_summary.csv")
