"""
Sensitivity analysis of the forced-response design (Reviewer 3, Comment 3).

Candidates may leave a question unanswered (0 points) instead of answering (+1 correct, -0.25
incorrect); the models answered every question. This script recomputes each model's mean SSM
score and its percentile rank among the candidates under two abstention conditions:

1. Consistency-based abstention (main sensitivity analysis). A model abstains on a question when
   its most frequent answer (the option text, independent of the shuffled letter, as extracted
   automatically in Results/*/Raw_Responses) was given in fewer than THRESHOLD of the 50 runs, i.e. when its answer is unstable; on the other questions
   it keeps the answers it gave. Thresholds 0.4, 0.5 (main) and 0.6 are reported.
2. Item-level optimal abstention (upper bound). A model abstains on every question whose
   accuracy over the 50 runs is below 20%, the accuracy below which answering has a negative
   expected value (0.2 x 1 - 0.8 x 0.25 = 0). No strategy that abstains on fixed questions can
   gain more; the gain is therefore the maximum cost of the forced-response design.

Scores are computed from the accuracy of each question over the 50 runs in the final dataset
(Analysis/statistics_output/llm_item_accuracy.csv); expected item score = 1.25 x accuracy - 0.25,
or 0 when the model abstains. Percentile rank = percentage of candidates with a test score lower
than or equal to the model's mean score; 95% CI by percentile bootstrap over the 140 questions
of each year (4,000 resamples, seed 2026), as in Candidate_analysis/scripts/candidate_analysis.R.

Run from the repository root (requires the Git LFS raw responses in Results/*/Raw_Responses):
    python Analysis/abstention_sensitivity.py
Outputs (Analysis/statistics_output/):
    abstention_sensitivity.csv          one row per model x year x condition
    abstention_sensitivity_summary.csv  one row per model x condition (mean over years)
"""
import collections
import json
import os

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "Analysis", "statistics_output")
B, SEED = 4000, 2026
THRESHOLDS = [0.4, 0.5, 0.6]
MAIN = 0.5

RAW = {
    "CLA": "Closed Models/Raw_Responses/raw_responses_claude.jsonl",
    "DPSK": "Closed Models/Raw_Responses/raw_responses_deepseek.jsonl",
    "GPT": "Closed Models/Raw_Responses/raw_responses_gpt.jsonl",
    "GROK": "Closed Models/Raw_Responses/raw_responses_grok.jsonl",
    "MSTRL": "Closed Models/Raw_Responses/raw_responses_mistral.jsonl",
    "QWEN": "Closed Models/Raw_Responses/raw_responses_qwen.jsonl",
    "MG4B": "Open Models/Raw_Responses/raw_responses_MedGemma_4B_IT.jsonl",
    "MT8B": "Open Models/Raw_Responses/raw_responses_Meditron3_8B_FP16.jsonl",
    "mSTRL3B": "Open Models/Raw_Responses/raw_responses_Ministral_3B_Instruct.jsonl",
    "Q1_7B": "Open Models/Raw_Responses/raw_responses_Qwen3_1_7B.jsonl",
    "Q4B": "Open Models/Raw_Responses/raw_responses_Qwen3_4B_Instruct.jsonl",
    "Q8B": "Open Models/Raw_Responses/raw_responses_Qwen3_8B.jsonl",
    "MG4Bq4": "Quantized models/Raw_Responses/raw_responses_MedGemma_4B_Q4_K_M.jsonl",
    "MG4Bq6": "Quantized models/Raw_Responses/raw_responses_MedGemma_4B_Q6_K.jsonl",
    "MG4Bq8": "Quantized models/Raw_Responses/raw_responses_MedGemma_4B_Q8_0.jsonl",
    "MT8Bq4": "Quantized models/Raw_Responses/raw_responses_Meditron3_8B_Q4_K_M.jsonl",
    "MT8Bq6": "Quantized models/Raw_Responses/raw_responses_Meditron3_8B_Q6_K.jsonl",
    "MT8Bq8": "Quantized models/Raw_Responses/raw_responses_Meditron3_8B_Q8_0.jsonl",
}
GROUP = {m: ("Closed" if i < 6 else "Open" if i < 12 else "Quantized") for i, m in enumerate(RAW)}


def modal_share(path):
    """Share of runs giving the most frequent answer, for every (year, question)."""
    answers = {}
    with open(os.path.join(ROOT, "Results", path), encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            r = json.loads(line)
            letter = (r.get("extracted_letter") or "").strip()
            text = r.get("options", {}).get(letter, "ND") if letter else "ND"
            answers[(r["year"], r["question_idx"], r["run"])] = text   # last record wins (duplicates)
    by_item = collections.defaultdict(list)
    for (y, q, _), t in answers.items():
        by_item[(y, q)].append(t)
    return {k: collections.Counter(v).most_common(1)[0][1] / len(v) for k, v in by_item.items()}


def main():
    items = pd.read_csv(os.path.join(OUT, "llm_item_accuracy.csv"))
    cand = pd.read_csv(os.path.join(ROOT, "Candidate_analysis", "data", "candidate_scores_2020_2024.csv"))
    rng = np.random.default_rng(SEED)
    years = sorted(items.year.unique())
    idx = {y: rng.integers(0, 140, size=(B, 140)) for y in years}            # same resamples for all models
    cands = {y: np.sort(cand.loc[cand.year == y, "test_score"].to_numpy()) for y in years}
    prank = lambda s, y: 100 * np.searchsorted(cands[y], s, side="right") / len(cands[y])

    rows = []
    for m, path in RAW.items():
        share = modal_share(path)
        for y in years:
            it = items[(items.model == m) & (items.year == y)].sort_values("question")
            acc = it.item_accuracy.to_numpy()
            mod = np.array([share[(y, q)] for q in it.question])
            sc = 1.25 * acc - 0.25
            conds = {"forced": np.zeros(140, bool),
                     **{f"abstain_modal<{t:.0%}": mod < t for t in THRESHOLDS},
                     "abstain_optimal(acc<20%)": acc < 0.2}
            for c, ab in conds.items():
                s = np.where(ab, 0.0, sc)
                boot = s[idx[y]].sum(1)
                pb = prank(boot, y)
                rows.append({"model": m, "group": GROUP[m], "year": y, "condition": c,
                             "items_abstained": int(ab.sum()), "accuracy_abstained_items": round(acc[ab].mean() * 100, 1) if ab.any() else None,
                             "mean_score": round(s.sum(), 2), "percentile": round(prank(s.sum(), y), 1),
                             "pct_ci_low": round(np.percentile(pb, 2.5), 1), "pct_ci_high": round(np.percentile(pb, 97.5), 1)})
    d = pd.DataFrame(rows)
    f = d[d.condition == "forced"].set_index(["model", "year"])
    d["score_change"] = (d.mean_score - f.mean_score.reindex(d.set_index(["model", "year"]).index).to_numpy()).round(2)
    d["percentile_change"] = (d.percentile - f.percentile.reindex(d.set_index(["model", "year"]).index).to_numpy()).round(1)
    d.to_csv(os.path.join(OUT, "abstention_sensitivity.csv"), index=False)
    s = d.groupby(["model", "group", "condition"], sort=False).agg(
        items_abstained_per_year=("items_abstained", "mean"), score_change_mean=("score_change", "mean"),
        score_change_min=("score_change", "min"), score_change_max=("score_change", "max"),
        percentile_change_min=("percentile_change", "min"), percentile_change_max=("percentile_change", "max")).round(2).reset_index()
    s.to_csv(os.path.join(OUT, "abstention_sensitivity_summary.csv"), index=False)
    print(s[s.condition.isin(["abstain_modal<50%", "abstain_optimal(acc<20%)"])].to_string(index=False))


if __name__ == "__main__":
    main()
