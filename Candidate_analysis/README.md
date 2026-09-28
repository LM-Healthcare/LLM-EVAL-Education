# Candidate_analysis — human reference data (SSM 2020–2024)

This folder contains the anonymised human-candidate scores used as reference in the paper, the code
that builds them from the official national rankings, and the analysis that compares them with the
models (Tables 2, 3 and 5, Figures 1–4, Supplementary Tables S8–S9).

## Source of the candidate data

Candidate scores come from the official national rankings (*graduatorie nazionali*) of the SSM
examination, published by the Italian Ministry of University and Research (MUR) on the Universitaly
portal. The authors downloaded each ranking when it was published (2020–2024). The rankings are
no longer publicly accessible on the portal, so no working public link can be provided. The original
files contain candidates' names and dates of birth and are therefore **not** distributed; this folder
contains only the anonymised scores derived from them.

| Year | Participants (reported) ² | Candidates in the ranking | Candidates with scores in the archived file |
|---|---|---|---|
| 2020 | 23,756 | 23,600 | 23,600 |
| 2021 | 19,449 | 19,442 | 19,442 |
| 2022 | 15,873 | 15,869 | 15,769 ¹ |
| 2023 | 14,043 | 14,036 | 14,036 |
| 2024 | 14,125 | 14,121 | 14,121 |

² Number of candidates who sat the examination as reported by MUR (2021, 2022, 2024: mur.gov.it press releases) or by specialised
sources (2020: specializzazionemedicina.com; 2023: blog.edises.it).

¹ The archived 2022 ranking lacks positions 401–500 (100 candidates, all with total scores between
117.0 and 118.5, i.e. well above the third quartile). Including them at any possible value changes the
2022 median from 80.5 to 80.75 and the third quartile from 93.25 to 93.5, leaves the first quartile
unchanged, changes model percentile ranks by at most 0.6 points and changes one significance level in
Table 5 (Meditron3-8B, 2022: p < 0.001 → p < 0.01).

## Score used

The rankings report, for each candidate, the **test score** (*Punteggio Prova*; in 2021 *Punteggio senza
CV*) and the points for academic titles/curriculum. Only the test score is used, because it is obtained
with the same scoring rule applied to the models (+1 correct, 0 unanswered, −0.25 incorrect).
Candidates may leave questions unanswered, whereas the models answered every question
(forced response).

## Contents

```
Candidate_analysis/
├── data/
│   ├── candidate_scores_2020_2024.csv   year, rank_order, total_score, test_score, titles_score
│   └── llm_runs_long.csv                one row per model × year × run (built from Results/*/Consistency)
├── scripts/
│   ├── build_llm_runs_long.py           builds data/llm_runs_long.csv from the Consistency files
│   ├── extract_rankings.py              builds data/candidate_scores_2020_2024.csv from the original
│   │                                    rankings (requires the original files, not distributed)
│   └── candidate_analysis.R             descriptives, percentiles, Mann-Whitney tests, tables, figures
└── outputs/
    ├── candidate_descriptives.csv/.xlsx          Supplementary Table S8
    ├── model_percentiles.csv                     Supplementary Table S9
    ├── mann_whitney_LLM_vs_candidates.csv/.xlsx  Table 5
    ├── table_descriptive.csv                     Tables 2–3
    └── Figure1–Figure4 (.png)
```

## Methods

- **Extraction.** The 2021 and 2023 rankings were published as spreadsheets and are read directly.
  The 2020, 2022 and 2024 rankings were published as PDF files and are extracted with `pdfplumber`.
  For every candidate the script checks that total score = test score + title points (no exceptions
  in any year). Two independent extractions of the PDF rankings, with PyMuPDF and with pdfplumber, produced
  identical scores for all candidates.
- **Descriptive statistics.** Quantiles use linear interpolation (R default, type 7).
- **Percentile rank of a model.** Percentage of candidates whose test score is lower than or equal to
  the model's median score over its 50 runs in the same year.
- **Mann-Whitney U tests (Table 5).** For each year and model, the 50 run scores of the model are
  compared with the test scores of all candidates of that year (two-sided, normal approximation with
  continuity correction, `wilcox.test(exact = FALSE)`); p-values are Holm-adjusted across the 18 models
  within each year.

## How to reproduce

From this folder:

```
python scripts/build_llm_runs_long.py       # rebuilds data/llm_runs_long.csv from ../Results
Rscript scripts/candidate_analysis.R        # uses data/ only; writes outputs/
python scripts/extract_rankings.py          # only with the original rankings in original_rankings/
```

R packages: ggplot2, dplyr, tidyr, ggtext, patchwork, scales, openxlsx. Python: pandas, openpyxl (and pdfplumber for the extraction).

The CSV files in `data/` are comma-separated with a dot as decimal separator. Opening and re-saving them with a spreadsheet program set to a different locale can change the separators and corrupt long decimal numbers; if needed, regenerate them with the scripts above.
