"""
Compare the question files in Dataset/ with the text actually administered to the models,
as recorded in the archived raw outputs (Results/*/Raw_Responses/*.jsonl).

Run from the repository root, after `git lfs pull`:
    python Analysis/compare_dataset_administered.py

The administered text is identical for all 18 models (checked by the script), except for
2023 Q134, whose answer options were missing ("nan") in the runs of nine models.
Output: Analysis/dataset_vs_administered_text.csv, one row per field (question stem, correct
option or distractor) whose text differs, with the type of difference:
  typographic       differences in ligatures (fi/fl), accents, apostrophes, spacing, case
                    or punctuation only
  footer residue    a page-footer fragment of the source PDF ("del <n>") appended to an option
  wording           minor wording/spelling differences (similarity >= 0.80)
  substantive       defective options in the administered version (seven questions; see `note`)
Requirements: pandas.
"""
import difflib
import glob
import json
import os
import re
import unicodedata

import pandas as pd

LIG = {"ﬁ": "fi", "ﬂ": "fl", "ﬃ": "ffi", "ﬀ": "ff", "ﬄ": "ffl", "’": "'", "‘": "'", "“": '"', "”": '"',
       "–": "-", "—": "-", "…": "...", " ": " "}
NOTES = {
    (2020, 112): "image-dependent item; options garbled in the administered version (text split across options)",
    (2022, 9): "one distractor replaced by explanatory text from the source",
    (2022, 41): "one distractor replaced by explanatory text from the source",
    (2022, 77): "one distractor replaced by explanatory text from the source",
    (2022, 100): "correct option replaced by explanatory text from the source",
    (2022, 119): "explanatory text appended to one distractor",
    (2023, 134): "options missing ('nan') for Meditron3-8B (all variants), Qwen3-1.7B, Qwen3-4B, Qwen3-8B and "
                 "Ministral-3B; option texts present, unlettered, at the end of the stem for all models",
    (2023, 25): "correct option: synonym ('precoce' administered, 'prematura' in Dataset/)",
    (2023, 41): "distractor shortened in the administered version (text in parentheses missing)",
}
SUBSTANTIVE = {(2020, 112), (2022, 9), (2022, 41), (2022, 77), (2022, 100), (2022, 119), (2023, 134)}


def soft(s):
    s = str(s)
    for a, b in LIG.items():
        s = s.replace(a, b)
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c)).lower()
    return re.sub(r"[^a-z0-9%<>=]+", "", s)


def norm(s):
    return re.sub(r"\s+", " ", str(s)).strip()


def unfoot(s):
    return re.sub(r"\s+del\s+\d+\s*$", "", norm(s))


def classify(adm, ds):
    if soft(adm) == soft(ds):
        return "typographic"
    if soft(unfoot(adm)) == soft(ds):
        return "footer residue"
    return "wording" if difflib.SequenceMatcher(None, soft(adm), soft(ds)).ratio() >= 0.80 else "substantive"


# administered text, first occurrence per question, for every model
files = sorted(glob.glob("Results/*/Raw_Responses/raw_responses_*.jsonl"))
if not files or os.path.getsize(files[0]) < 1000:
    raise SystemExit("raw files missing or Git LFS pointers: run `git lfs pull` first")
adm = {}
for f in files:
    m = os.path.basename(f)[len("raw_responses_"):-len(".jsonl")]
    d = {}
    with open(f, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                r = json.loads(line)
                d.setdefault((r["year"], r["question_idx"]), r)
    adm[m] = d
ref = adm["deepseek"]
for m, d in adm.items():
    diff = [k for k in d if (d[k]["question"], sorted(map(str, d[k]["options"].values())))
            != (ref[k]["question"], sorted(map(str, ref[k]["options"].values())))]
    print(f"{m:<24} questions administered differently from deepseek: {diff}")

rows = []
for y in range(2020, 2025):
    ds = pd.read_excel(f"Dataset/{y}_SSM_[checked].xlsx")
    for i, R in ds.iterrows():
        k = (y, i + 1)
        r = ref[k]
        opts = r["options"]
        pairs = [("question", r["question"], R["Question"]), ("correct option", opts[r["correct_answer"]], R["Answer A"])]
        a_rest = [opts[c] for c in sorted(opts) if c != r["correct_answer"]]
        d_rest = [R[f"Answer {c}"] for c in "BCDE"]
        # match distractors: exact, then without footer, then most similar
        for key in (soft, lambda s: soft(unfoot(s))):
            for dv in list(d_rest):
                hit = [av for av in a_rest if key(av) == soft(dv)]
                if hit:
                    if norm(hit[0]) != norm(dv):
                        pairs.append(("distractor", hit[0], dv))
                    a_rest.remove(hit[0])
                    d_rest.remove(dv)
        for dv in d_rest:
            av = max(a_rest, key=lambda s: difflib.SequenceMatcher(None, soft(s), soft(dv)).ratio())
            a_rest.remove(av)
            pairs.append(("distractor", av, dv))
        for field, a, d in pairs:
            if norm(a) != norm(d):
                rows.append({"year": y, "question": i + 1, "field": field, "type": classify(a, d),
                             "note": NOTES.get(k, ""), "text_administered": norm(a), "text_dataset": norm(d)})
        if k in NOTES and not any(x["year"] == y and x["question"] == i + 1 for x in rows):
            rows.append({"year": y, "question": i + 1, "field": "", "type": "substantive", "note": NOTES[k],
                         "text_administered": "", "text_dataset": ""})
out = pd.DataFrame(rows)
for y, q in (set(zip(out.year, out.question))):
    sel = (out.year == y) & (out.question == q)
    if (y, q) in SUBSTANTIVE:
        out.loc[sel, "type"] = "substantive"
    elif (out.loc[sel, "type"] == "substantive").any():
        out.loc[sel & (out.type == "substantive"), "type"] = "wording"
out.to_csv("Analysis/dataset_vs_administered_text.csv", index=False)
print(out.groupby("type").agg(fields=("type", "size"), questions=("question", lambda s: len(set(zip(out.loc[s.index, "year"], s))))))
print("questions with substantive differences:", sorted(set(zip(out[out.type == "substantive"].year, out[out.type == "substantive"].question))))
