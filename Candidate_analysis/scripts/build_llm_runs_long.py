"""
Build data/llm_runs_long.csv (one row per model x year x run) from the final dataset,
i.e. the Consistency files in Results/. Run from Candidate_analysis/:

    python scripts/build_llm_runs_long.py

accuracy = correct / 140; score = correct - 0.25 * (140 - correct), because every question
received an answer (forced response); an empty cell, if any, is counted as incorrect.
"""
import glob
import os

import pandas as pd

KEY = {"claude": ("CLA", "closed"), "deepseek": ("DPSK", "closed"), "gpt-52-2025-12-11": ("GPT", "closed"),
       "grok": ("GROK", "closed"), "mistral": ("MSTRL", "closed"), "qwen": ("QWEN", "closed"),
       "MedGemma_4B_IT": ("MG4B", "open"), "Meditron3_8B_FP16": ("MT8B", "open"),
       "Ministral_3B_Instruct": ("mSTRL3B", "open"), "Qwen3_1_7B": ("Q1_7B", "open"),
       "Qwen3_4B_Instruct": ("Q4B", "open"), "Qwen3_8B": ("Q8B", "open"),
       "MedGemma_4B_Q4_K_M": ("MG4Bq4", "quantized"), "MedGemma_4B_Q6_K": ("MG4Bq6", "quantized"),
       "MedGemma_4B_Q8_0": ("MG4Bq8", "quantized"), "Meditron3_8B_Q4_K_M": ("MT8Bq4", "quantized"),
       "Meditron3_8B_Q6_K": ("MT8Bq6", "quantized"), "Meditron3_8B_Q8_0": ("MT8Bq8", "quantized")}
rows = []
for f in glob.glob(os.path.join("..", "Results", "*", "Consistency", "consistency_*.xlsx")):
    model, cat = KEY[os.path.basename(f)[len("consistency_"):-5]]
    d = pd.read_excel(f)
    q = [f"Domanda_{i}" for i in range(1, 141)]
    correct = (d[q].astype(str) == "1").sum(axis=1)
    for run_exam, c in zip(d["RUN_EXAM"], correct):
        rows.append({"RUN_EXAM": run_exam, "year": int(run_exam[:4]), "run": int(run_exam.split("_")[-1]),
                     "model": model, "category": cat, "correct": int(c), "accuracy": c / 140,
                     "score": c - 0.25 * (140 - c)})
out = pd.DataFrame(rows)
out = out.sort_values(["year", "run", "model"]).reset_index(drop=True)
os.makedirs("data", exist_ok=True)
out.to_csv(os.path.join("data", "llm_runs_long.csv"), index=False)
print("written data/llm_runs_long.csv:", len(out), "rows")
