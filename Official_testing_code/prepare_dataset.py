"""
Prepare the question files in the format expected by the evaluation scripts.

The evaluation scripts read, from the folder given in `dataset_path` (default "Dataset",
relative to the working directory), one file per year named
    {year}_answers_converted_[checked].xlsx
with the Italian column names `Domanda` and `Risposta A` ... `Risposta E` (Risposta A = correct
answer). The files distributed in Dataset/ use English column names and the name
{year}_SSM_[checked].xlsx. This script converts them without modifying Dataset/.

Two sources are available:

  --source dataset  (default) the files in Dataset/ (checked transcription of the questions)
  --source raw      the text actually administered at run time, rebuilt from one of the
                    archived raw-output files (requires `git lfs pull`). The order of the four
                    distractors (Risposta B-E) cannot be recovered from the raw file and is
                    irrelevant because the options are shuffled at every run.
                    See the main README, section "Dataset version".

Usage, from the repository root:
    python Official_testing_code/prepare_dataset.py                       # -> run/Dataset/
    python Official_testing_code/prepare_dataset.py --source raw \
        --raw-file "Results/Closed Models/Raw_Responses/raw_responses_deepseek.jsonl"
    cd run && python ../Official_testing_code/Closed/test_deepseek_medical_exam_v2.py

Requirements: pandas, openpyxl.
"""
import argparse
import json
import os

import pandas as pd

YEARS = [2020, 2021, 2022, 2023, 2024]
LETTERS = "ABCDE"


def from_dataset(src_dir):
    out = {}
    for y in YEARS:
        d = pd.read_excel(os.path.join(src_dir, f"{y}_SSM_[checked].xlsx"))
        ren = {"Question": "Domanda", **{f"Answer {c}": f"Risposta {c}" for c in LETTERS}}
        out[y] = d.rename(columns=ren)
    return out


def from_raw(raw_file, src_dir):
    first = {}
    with open(raw_file, encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            r = json.loads(line)
            first.setdefault((r["year"], r["question_idx"]), r)
    out = {}
    for y in YEARS:
        meta = pd.read_excel(os.path.join(src_dir, f"{y}_SSM_[checked].xlsx"))
        rows = []
        for i in range(len(meta)):
            r = first[(y, i + 1)]
            opts = r["options"]
            correct = opts[r["correct_answer"]]
            others = [opts[k] for k in sorted(opts) if k != r["correct_answer"]]
            row = {"Question Code": meta.loc[i, "Question Code"], "Domanda": r["question"],
                   "Risposta A": correct}
            row.update({f"Risposta {c}": v for c, v in zip("BCDE", others)})
            for col in ["Category", "Question type", "Image", "Image Category"]:
                row[col] = meta.loc[i, col]
            rows.append(row)
        out[y] = pd.DataFrame(rows)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", choices=["dataset", "raw"], default="dataset")
    ap.add_argument("--raw-file", help="raw_responses_*.jsonl file (with --source raw)")
    ap.add_argument("--dataset-dir", default="Dataset")
    ap.add_argument("--out", default=os.path.join("run", "Dataset"))
    a = ap.parse_args()
    if a.source == "raw" and not a.raw_file:
        ap.error("--source raw requires --raw-file")
    if a.source == "raw" and os.path.getsize(a.raw_file) < 1000:
        ap.error(f"{a.raw_file} looks like a Git LFS pointer: run `git lfs pull` first")
    data = from_dataset(a.dataset_dir) if a.source == "dataset" else from_raw(a.raw_file, a.dataset_dir)
    os.makedirs(a.out, exist_ok=True)
    for y, d in data.items():
        f = os.path.join(a.out, f"{y}_answers_converted_[checked].xlsx")
        d.to_excel(f, index=False)
        print(f"written {f} ({len(d)} questions)")


if __name__ == "__main__":
    main()
