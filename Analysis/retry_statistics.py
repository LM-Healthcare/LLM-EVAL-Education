"""
Retry and extended-budget responses of the locally executed models (Reviewer 3, Comment 5).

The scripts of the open-source and quantized models (Official_testing_code/Open, /Quantized) contain
two fallbacks that the API scripts do not have:
  - extended budget: when a Qwen3 or Meditron3 model opened a reasoning block (<think>), the same
    prompt was generated again with a budget of 2,048 tokens. With the /no_think instruction, Qwen3
    models open an empty block (<think></think>) before the answer, so this fallback was triggered
    for their responses, but it only matters when the block contains reasoning; responses with a
    non-empty reasoning block are counted here;
  - retry: when no answer letter could be extracted, the model was asked once more with a short
    prompt requesting only the letter; the raw output then contains "[RETRY]".
For every model this script counts both kinds of responses in Results/*/Raw_Responses and reports
the accuracy of each model (i) as in the paper and (ii) with every retry response scored as
incorrect, i.e. as if the retry had not been available (the proprietary models had no retry).
Correctness is taken from the final dataset (Results/*/Consistency).

Run from the repository root after `git lfs pull`:
    python Analysis/retry_statistics.py
Output: Analysis/statistics_output/retry_statistics.csv
"""
import json
import os
import re

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "Analysis", "statistics_output")
MODELS = {  # code: (raw responses, consistency file)
    "CLA": ("Closed Models", "claude", "claude"), "DPSK": ("Closed Models", "deepseek", "deepseek"),
    "GPT": ("Closed Models", "gpt", "gpt-52-2025-12-11"), "GROK": ("Closed Models", "grok", "grok"),
    "MSTRL": ("Closed Models", "mistral", "mistral"), "QWEN": ("Closed Models", "qwen", "qwen"),
    "MG4B": ("Open Models", "MedGemma_4B_IT", "MedGemma_4B_IT"), "MT8B": ("Open Models", "Meditron3_8B_FP16", "Meditron3_8B_FP16"),
    "mSTRL3B": ("Open Models", "Ministral_3B_Instruct", "Ministral_3B_Instruct"), "Q1_7B": ("Open Models", "Qwen3_1_7B", "Qwen3_1_7B"),
    "Q4B": ("Open Models", "Qwen3_4B_Instruct", "Qwen3_4B_Instruct"), "Q8B": ("Open Models", "Qwen3_8B", "Qwen3_8B"),
    "MG4Bq4": ("Quantized models", "MedGemma_4B_Q4_K_M", "MedGemma_4B_Q4_K_M"), "MG4Bq6": ("Quantized models", "MedGemma_4B_Q6_K", "MedGemma_4B_Q6_K"),
    "MG4Bq8": ("Quantized models", "MedGemma_4B_Q8_0", "MedGemma_4B_Q8_0"), "MT8Bq4": ("Quantized models", "Meditron3_8B_Q4_K_M", "Meditron3_8B_Q4_K_M"),
    "MT8Bq6": ("Quantized models", "Meditron3_8B_Q6_K", "Meditron3_8B_Q6_K"), "MT8Bq8": ("Quantized models", "Meditron3_8B_Q8_0", "Meditron3_8B_Q8_0"),
}


def main():
    rows = []
    for code, (folder, raw, cons) in MODELS.items():
        flags = {}
        with open(os.path.join(ROOT, "Results", folder, "Raw_Responses", f"raw_responses_{raw}.jsonl"), encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    r = json.loads(line)
                    text = str(r.get("raw_response") or "")
                    reasoning = False
                    if folder != "Closed Models" and text.lstrip().startswith("<think>"):
                        m = re.match(r"\s*<think>(.*?)(</think>|$)", text, re.S)
                        reasoning = bool(m and m.group(1).strip())
                    flags[(r["year"], r["run"], r["question_idx"])] = ("[RETRY]" in text, reasoning)
        c = pd.read_excel(os.path.join(ROOT, "Results", folder, "Consistency", f"consistency_{cons}.xlsx"))
        c = c[["RUN_EXAM"] + [k for k in c.columns if str(k).startswith("Domanda_")]]   # drop summary columns
        long = c.melt(id_vars="RUN_EXAM", var_name="q", value_name="correct")
        long["year"] = long.RUN_EXAM.str[:4].astype(int)
        long["run"] = long.RUN_EXAM.str.split("_").str[-1].astype(int)
        long["question_idx"] = long.q.str.replace("Domanda_", "").astype(int)
        long["correct"] = long.correct.astype(str).eq("1")
        f = [flags.get(k, (False, False)) for k in zip(long.year, long.run, long.question_idx)]
        long["retry"] = [a for a, _ in f]
        long["extended"] = [b for _, b in f]
        n = len(long)
        rows.append({"model": code, "responses": n,
                     "reasoning_n": int(long.extended.sum()), "reasoning_pct": round(100 * long.extended.mean(), 2),
                     "reasoning_correct_pct": round(100 * long.loc[long.extended, "correct"].mean(), 1) if long.extended.any() else None,
                     "retry_n": int(long.retry.sum()), "retry_pct": round(100 * long.retry.mean(), 2),
                     "retry_correct_pct": round(100 * long.loc[long.retry, "correct"].mean(), 1) if long.retry.any() else None,
                     "accuracy": round(100 * long.correct.mean(), 2),
                     "accuracy_retry_as_incorrect": round(100 * (long.correct & ~long.retry).mean(), 2)})
    d = pd.DataFrame(rows)
    d["accuracy_change"] = (d.accuracy_retry_as_incorrect - d.accuracy).round(2)
    d.to_csv(os.path.join(OUT, "retry_statistics.csv"), index=False)
    print(d.to_string(index=False))


if __name__ == "__main__":
    main()
