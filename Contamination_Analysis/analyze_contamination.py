"""
Analysis of the contamination experiments (Reviewer 3, Comment 1). Run from the repository root:
    python Contamination_Analysis/analyze_contamination.py

Inputs: Contamination_Analysis/items.csv, Contamination_Analysis/results/*.jsonl and, for the
comparison with the main study, the Consistency files in Results/.
Outputs (Contamination_Analysis/analysis/):
  perturbation_summary.csv   per model: accuracy on the 150 items in the main study (February 2026)
                             and in the three conditions; differences with 95% CIs (item bootstrap)
  perturbation_by_type.csv   the same accuracies by question type
  ts_guessing_summary.csv    per model: exact-match and near-exact (ROUGE-L >= 0.8) reproduction of
                             the masked option, overall and for options not derivable from the visible
                             ones; accuracy on items reproduced vs not reproduced
  sampled_decoding.csv       MedGemma 1.5-4B, greedy (main study) vs sampled decoding (if available)
Bootstrap: items resampled with replacement (B = 4000, seed 2026); the responses of all runs to an
item are kept together, so that the intervals reflect the sampling of questions.
"""
import glob
import json
import os
import re
import unicodedata

import warnings

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RES, OUT = os.path.join(HERE, "results"), os.path.join(HERE, "analysis")
B, SEED = 4000, 2026
DERIVABLE = 0.5            # ROUGE-L between masked option and the most similar visible option

CONSISTENCY = {            # result file name -> (Consistency file of the main study, label)
    "claude": ("Closed Models/Consistency/consistency_claude.xlsx", "Claude Sonnet 4.5"),
    "gpt": ("Closed Models/Consistency/consistency_gpt-52-2025-12-11.xlsx", "GPT 5.2"),
    "grok": ("Closed Models/Consistency/consistency_grok.xlsx", "Grok 4.1 Fast Reasoning"),
    "deepseek_v32_openrouter": ("Closed Models/Consistency/consistency_deepseek.xlsx", "DeepSeek V3.2"),
    "mistral_openrouter": ("Closed Models/Consistency/consistency_mistral.xlsx", "Mistral Large 3"),
    "qwen3_8b": ("Open Models/Consistency/consistency_Qwen3_8B.xlsx", "Qwen3-8B"),
    "meditron3": ("Open Models/Consistency/consistency_Meditron3_8B_FP16.xlsx", "Meditron3-8B"),
    "medgemma": ("Open Models/Consistency/consistency_MedGemma_4B_IT.xlsx", "MedGemma 1.5-4B"),
    "ministral": ("Open Models/Consistency/consistency_Ministral_3B_Instruct.xlsx", "Ministral-3B"),
    "qwen3_4b": ("Open Models/Consistency/consistency_Qwen3_4B_Instruct.xlsx", "Qwen3-4B"),
    "qwen3_1_7b": ("Open Models/Consistency/consistency_Qwen3_1_7B.xlsx", "Qwen3-1.7B"),
}
ORDER = list(CONSISTENCY)


def norm(s):
    s = unicodedata.normalize("NFKD", str(s))
    s = "".join(c for c in s if not unicodedata.combining(c)).lower()
    s = re.sub(r"^\s*[a-e]\s*[:.)]\s+", "", s)            # leading "A: " if the model added one
    return re.sub(r"[^a-z0-9%]+", " ", s).strip()


def rouge_l(a, b):
    a, b = a.split(), b.split()
    if not a or not b:
        return 0.0
    dp = [0] * (len(b) + 1)
    for x in a:
        prev = 0
        for j, y in enumerate(b):
            cur = dp[j + 1]
            dp[j + 1] = prev + 1 if x == y else max(dp[j + 1], dp[j])
            prev = cur
    lcs = dp[-1]
    return 0.0 if lcs == 0 else 2 * (lcs / len(a)) * (lcs / len(b)) / (lcs / len(a) + lcs / len(b))


def read_jsonl(pattern):
    return pd.DataFrame([json.loads(l) for f in sorted(glob.glob(os.path.join(RES, pattern)))
                         for l in open(f, encoding="utf-8") if l.strip()])


def feb_accuracy(model, keys):
    path = os.path.join(ROOT, "Results", CONSISTENCY[model][0])
    c = pd.read_excel(path).copy()
    c["year"] = c.iloc[:, 0].astype(str).str[:4].astype(int)
    return {(y, q): (c.loc[c.year == y, f"Domanda_{q}"].astype(str) == "1").mean() for y, q in keys}


def boot(X, rng):
    """X: items x columns of item-level accuracy -> bootstrap means (B x columns)."""
    idx = rng.integers(0, len(X), (B, len(X)))
    return X[idx].mean(axis=1)


def ci(a):
    return f"[{np.percentile(a, 2.5) * 100:.1f}, {np.percentile(a, 97.5) * 100:.1f}]"


def main():
    warnings.filterwarnings("ignore")
    os.makedirs(OUT, exist_ok=True)
    items = pd.read_csv(os.path.join(HERE, "items.csv"))
    keys = list(zip(items.year, items.question_idx))
    rng = np.random.default_rng(SEED)

    # ---------------- perturbation ----------------
    p = read_jsonl("perturbation_*.jsonl")
    p = p[[k in set(keys) for k in zip(p.year, p.question_idx)]]
    p["decoding"] = p.get("decoding", "default")
    p["decoding"] = p["decoding"].fillna("default")
    main_p = p[p.decoding == "default"]
    rows, by_type = [], []
    for model in [m for m in ORDER if m in set(main_p.model)]:
        g = main_p[main_p.model == model]
        acc = g.groupby(["year", "question_idx", "condition"]).is_correct.mean().unstack("condition")
        acc = acc.reindex(keys)
        feb = feb_accuracy(model, keys)
        acc["feb"] = [feb[k] for k in keys]
        cols = ["feb", "original", "nota_correct", "nota_distractor"]
        X = acc[cols].values
        bs, est = boot(X, rng), X.mean(axis=0)
        rows.append({
            "model": CONSISTENCY[model][1], "runs": int(g.run.nunique()), "items": len(keys),
            "acc_main_study": round(est[0] * 100, 1), "acc_original": round(est[1] * 100, 1),
            "acc_nota_correct": round(est[2] * 100, 1), "acc_nota_distractor": round(est[3] * 100, 1),
            "diff_original_vs_main_study": round((est[1] - est[0]) * 100, 1),
            "diff_original_vs_main_study_ci": ci(bs[:, 1] - bs[:, 0]),
            "drop_nota_correct": round((est[1] - est[2]) * 100, 1),
            "drop_nota_correct_ci": ci(bs[:, 1] - bs[:, 2]),
            "relative_drop_pct": round((est[1] - est[2]) / est[1] * 100, 1),
            "drop_nota_distractor": round((est[1] - est[3]) * 100, 1),
            "drop_nota_distractor_ci": ci(bs[:, 1] - bs[:, 3]),
            "nota_chosen_when_distractor_pct": round(g[g.condition == "nota_distractor"].chose_nota.mean() * 100, 1),
        })
        t = g.groupby(["question_type", "condition"]).is_correct.mean().mul(100).round(1).unstack("condition")
        for qt, r in t.iterrows():
            by_type.append({"model": CONSISTENCY[model][1], "question_type": qt, **r.to_dict()})
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "perturbation_summary.csv"), index=False)
    pd.DataFrame(by_type).to_csv(os.path.join(OUT, "perturbation_by_type.csv"), index=False)

    # ---------------- sampled decoding (MedGemma) ----------------
    s = p[(p.decoding == "sampled") & (p.condition == "original")]
    if len(s):
        rows_s = []
        for model, g in s.groupby("model"):
            acc = g.groupby(["year", "question_idx"]).is_correct.mean().reindex(keys)
            feb = feb_accuracy(model, keys)
            X = np.column_stack([[feb[k] for k in keys], acc.values])
            bs, est = boot(X, rng), X.mean(axis=0)
            # consistency: share of items answered identically (same option text) in every run
            g = g.assign(chosen=[r["options"].get(r["answer"]) if isinstance(r["options"], dict) else None
                                 for _, r in g.iterrows()])
            same = g.groupby(["year", "question_idx"]).chosen.nunique().eq(1).mean()
            rows_s.append({"model": CONSISTENCY[model][1], "runs": int(g.run.nunique()),
                           "acc_greedy_main_study": round(est[0] * 100, 1), "acc_sampled": round(est[1] * 100, 1),
                           "diff": round((est[1] - est[0]) * 100, 1), "diff_ci": ci(bs[:, 1] - bs[:, 0]),
                           "items_same_answer_all_runs_pct": round(same * 100, 1)})
        pd.DataFrame(rows_s).to_csv(os.path.join(OUT, "sampled_decoding.csv"), index=False)

    # ---------------- TS-Guessing ----------------
    t = read_jsonl("ts_guessing_*.jsonl")
    if len(t):
        it = items.set_index(["year", "question_idx"])
        deriv = {}
        for (y, q), r in it.iterrows():
            opts = [r.correct, r.d1, r.d2, r.d3, r.d4]
            m = int(r.ts_mask)
            deriv[(y, q)] = max(rouge_l(norm(opts[m]), norm(o)) for i, o in enumerate(opts) if i != m) >= DERIVABLE
        t["em"] = [norm(a) == norm(b) for a, b in zip(t.target, t.response)]
        t["rouge"] = [rouge_l(norm(b), norm(a)) for a, b in zip(t.target, t.response)]
        t["derivable"] = [deriv[(y, q)] for y, q in zip(t.year, t.question_idx)]
        t["reproduced"] = t.em | (t.rouge >= 0.8)
        rows_t = []
        for model in [m for m in ORDER if m in set(t.model)]:
            g = t[t.model == model]
            nd = g[~g.derivable]
            a = main_p[(main_p.model == model)].groupby(["year", "question_idx", "condition"]).is_correct.mean().unstack()
            feb = feb_accuracy(model, keys)
            rep = set(zip(g[g.reproduced].year, g[g.reproduced].question_idx))
            in_rep = [k in rep for k in keys]
            febv = np.array([feb[k] for k in keys])
            notac = a.reindex(keys)["nota_correct"].values if "nota_correct" in a else np.full(len(keys), np.nan)
            rows_t.append({
                "model": CONSISTENCY[model][1], "items": len(g),
                "exact_match_pct": round(g.em.mean() * 100, 1),
                "near_exact_pct": round(g.reproduced.mean() * 100, 1),
                "nonderivable_items": len(nd),
                "exact_match_nonderivable_pct": round(nd.em.mean() * 100, 1),
                "near_exact_nonderivable_pct": round(nd.reproduced.mean() * 100, 1),
                "items_reproduced": int(sum(in_rep)),
                "acc_main_study_reproduced": round(febv[in_rep].mean() * 100, 1) if any(in_rep) else None,
                "acc_main_study_not_reproduced": round(febv[[not x for x in in_rep]].mean() * 100, 1),
                "acc_nota_correct_reproduced": round(np.nanmean(notac[in_rep]) * 100, 1) if any(in_rep) else None,
                "acc_nota_correct_not_reproduced": round(np.nanmean(notac[[not x for x in in_rep]]) * 100, 1),
            })
        pd.DataFrame(rows_t).to_csv(os.path.join(OUT, "ts_guessing_summary.csv"), index=False)
        print(f"derivable masked options: {sum(deriv.values())} of {len(deriv)}")

    for f in sorted(glob.glob(os.path.join(OUT, "*.csv"))):
        print(f"\n== {os.path.basename(f)}")
        print(pd.read_csv(f).to_string(index=False))


if __name__ == "__main__":
    main()
