"""
Image-dependent questions administered without their images (Reviewer 3, Comment 4).

1. Accuracy on the 665 text-only questions compared with accuracy on all 700 questions, for every
   model (mean item accuracy over the 50 runs).
2. Comparison with the human candidates without credit for image-dependent questions. The
   candidates' test scores include the image-dependent questions and are not available item by
   item, so these questions cannot be removed symmetrically. They are therefore scored as
   unanswered (0 points) for the models, the condition least favourable to the models: the
   model's mean score is recomputed and located in the candidate distribution of the same year
   (percentile rank = percentage of candidates with a test score lower than or equal to the
   model's mean score; 95% CI by percentile bootstrap over the 140 questions of that year, 4,000
   resamples, seed 2026, as in Candidate_analysis/scripts/candidate_analysis.R).

Run from the repository root:
    python Analysis/image_items_sensitivity.py
Outputs (Analysis/statistics_output/):
    image_items_accuracy.csv      one row per model: accuracy on all and on text-only questions
    image_items_sensitivity.csv   one row per model x year: mean score and percentile rank with
                                  image-dependent questions scored as unanswered
"""
import glob
import os

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "Analysis", "statistics_output")
B, SEED = 4000, 2026
ORDER = ["CLA", "DPSK", "GPT", "GROK", "MSTRL", "QWEN", "MG4B", "MT8B", "mSTRL3B", "Q1_7B", "Q4B", "Q8B",
         "MG4Bq4", "MG4Bq6", "MG4Bq8", "MT8Bq4", "MT8Bq6", "MT8Bq8"]
GROUP = {m: ("Closed" if i < 6 else "Open" if i < 12 else "Quantized") for i, m in enumerate(ORDER)}


def main():
    img = []
    for f in sorted(glob.glob(os.path.join(ROOT, "Dataset", "*_SSM_*.xlsx"))):
        d = pd.read_excel(f)
        year = int(os.path.basename(f)[:4])
        img += [(year, q + 1, str(v).strip().lower() == "yes") for q, v in enumerate(d["Image"])]
    img = pd.DataFrame(img, columns=["year", "question", "image"])
    items = pd.read_csv(os.path.join(OUT, "llm_item_accuracy.csv")).merge(img, on=["year", "question"])
    assert len(items) == 18 * 700 and img.image.sum() == 35

    # 1. accuracy on all vs text-only questions
    acc = items.groupby("model").apply(lambda g: pd.Series({
        "accuracy_all_700": g.item_accuracy.mean() * 100,
        "accuracy_text_only_665": g.loc[~g.image, "item_accuracy"].mean() * 100,
        "accuracy_image_35": g.loc[g.image, "item_accuracy"].mean() * 100}), include_groups=False).reindex(ORDER)
    acc["difference_text_only_minus_all"] = acc.accuracy_text_only_665 - acc.accuracy_all_700
    acc.insert(0, "group", [GROUP[m] for m in acc.index])
    for col in ["rank_all", "rank_text_only"]:
        src = "accuracy_all_700" if col == "rank_all" else "accuracy_text_only_665"
        acc[col] = acc.groupby("group")[src].rank(ascending=False).astype(int)
    acc.round(3).reset_index().to_csv(os.path.join(OUT, "image_items_accuracy.csv"), index=False)

    # 2. human comparison with image-dependent questions scored as unanswered
    cand = pd.read_csv(os.path.join(ROOT, "Candidate_analysis", "data", "candidate_scores_2020_2024.csv"))
    rng = np.random.default_rng(SEED)
    rows = []
    for y in sorted(items.year.unique()):
        cs = np.sort(cand.loc[cand.year == y, "test_score"].to_numpy())
        prank = lambda s: 100 * np.searchsorted(cs, s, side="right") / len(cs)
        idx = rng.integers(0, 140, size=(B, 140))
        for m in ORDER:
            it = items[(items.model == m) & (items.year == y)].sort_values("question")
            sc = 1.25 * it.item_accuracy.to_numpy() - 0.25
            sc0 = np.where(it.image.to_numpy(), 0.0, sc)
            boot = prank(sc0[idx].sum(1))
            rows.append({"model": m, "group": GROUP[m], "year": y, "image_items": int(it.image.sum()),
                         "mean_score": round(sc.sum(), 2), "percentile": round(prank(sc.sum()), 1),
                         "mean_score_images_unanswered": round(sc0.sum(), 2),
                         "percentile_images_unanswered": round(prank(sc0.sum()), 1),
                         "pct_ci_low": round(np.percentile(boot, 2.5), 1), "pct_ci_high": round(np.percentile(boot, 97.5), 1)})
    r = pd.DataFrame(rows)
    r["score_change"] = (r.mean_score_images_unanswered - r.mean_score).round(2)
    r["percentile_change"] = (r.percentile_images_unanswered - r.percentile).round(1)
    r.to_csv(os.path.join(OUT, "image_items_sensitivity.csv"), index=False)
    print(acc.round(1).to_string())
    print(r.groupby("model", sort=False).agg(score_change_min=("score_change", "min"), score_change_max=("score_change", "max"),
                                             pct_min=("percentile_images_unanswered", "min"),
                                             pct_change_min=("percentile_change", "min")).to_string())


if __name__ == "__main__":
    main()
