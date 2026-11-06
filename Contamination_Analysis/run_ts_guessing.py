"""
Option-completion probe for benchmark contamination (TS-Guessing, Deng et al., 2024), added in
the second revision. For each of the 150 items in items.csv, one incorrect option is replaced by
[MASK] and the model is asked to reproduce its original text. Incorrect options are arbitrary
pieces of text that cannot be inferred from medical knowledge, so reproducing them exactly or
almost exactly indicates prior exposure to the item.

Usage, from the repository root (API models only; one model at a time; resumable):
    python Contamination_Analysis/run_ts_guessing.py --model gpt

The API clients and endpoints are those of the scripts in Official_testing_code/Closed/.
Output: Contamination_Analysis/results/ts_guessing_<model>.jsonl
"""
import argparse
import json
import os
import random
import sys
import time

import pandas as pd
import requests

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_perturbation import MODELS, ROOT, load_tester  # noqa: E402

LETTERS = "ABCDE"
MAX_TOKENS = 2000          # large budget: GPT 5.2 and Grok 4.1 spend tokens on internal reasoning
TEMPERATURE = 0.1          # as in the main study

PROMPT = """The following multiple-choice question comes from the Italian national medical specialty \
examination (SSM, {year}). One of the incorrect answer options has been replaced by [MASK].

Fill in the [MASK] in option {letter} with the exact original text of that option, as it appears \
in the examination. Reply ONLY with the text of option {letter}, in Italian, without the letter and \
without any explanation. Do not repeat any of the other options.

QUESTION:
{question}

OPTIONS:
{options}"""


def chat(tester, name, prompt):
    if name == "claude":
        r = tester.client.messages.create(model=tester.model_name, max_tokens=MAX_TOKENS,
                                          temperature=TEMPERATURE, messages=[{"role": "user", "content": prompt}])
        return "".join(b.text for b in r.content if getattr(b, "type", "") == "text").strip()
    if name == "gpt":
        r = tester.client.chat.completions.create(model=tester.model_name, temperature=TEMPERATURE,
                                                  max_completion_tokens=MAX_TOKENS,
                                                  messages=[{"role": "user", "content": prompt}])
        return (r.choices[0].message.content or "").strip()
    data = {"model": tester.model_name, "messages": [{"role": "user", "content": prompt}],
            "temperature": TEMPERATURE, "max_tokens": MAX_TOKENS}
    for attempt in range(6):
        resp = requests.post(tester.api_url, headers=tester.headers, json=data, timeout=300)
        if resp.status_code == 200:
            return (resp.json()["choices"][0]["message"].get("content") or "").strip()
        time.sleep(2 * 2 ** attempt)
    raise RuntimeError(f"API error {resp.status_code}: {resp.text[:200]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=[m for m, v in MODELS.items() if v[2] == "api"])
    ap.add_argument("--items", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "items.csv"))
    ap.add_argument("--model-id", default=None, help="query this model identifier instead of the script's one")
    ap.add_argument("--outdir", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "results"))
    a = ap.parse_args()
    items = pd.read_csv(a.items)
    os.makedirs(a.outdir, exist_ok=True)
    out_file = os.path.join(a.outdir, f"ts_guessing_{a.model}.jsonl")
    done = set()
    if os.path.exists(out_file):
        with open(out_file, encoding="utf-8") as fh:
            done = {(r["year"], r["question_idx"]) for r in map(json.loads, filter(str.strip, fh))}
    os.chdir(a.outdir)
    tester = load_tester(a.model)
    if a.model_id:
        print(f"{a.model}: querying {a.model_id} instead of {tester.model_name}")
        tester.model_name = a.model_id
    fails = 0
    with open(out_file, "a", encoding="utf-8") as out:
        for _, it in items.iterrows():
            y, q = int(it["year"]), int(it["question_idx"])
            if (y, q) in done:
                continue
            opts = [str(it["correct"])] + [str(it[f"d{k}"]) for k in range(1, 5)]
            masked_idx = int(it["ts_mask"])                     # 1-4: one of the distractors
            order = list(range(5))
            random.Random(f"ts|{y}|{q}").shuffle(order)          # same option order for every model
            shown = [("[MASK]" if i == masked_idx else opts[i]) for i in order]
            letter = LETTERS[order.index(masked_idx)]
            prompt = PROMPT.format(year=y, letter=letter, question=it["question"],
                                   options="\n".join(f"{L}: {t}" for L, t in zip(LETTERS, shown)))
            try:
                text = chat(tester, a.model, prompt)
                fails = 0
            except Exception as e:                                # not recorded: retried on relaunch
                fails += 1
                print(f"  call failed ({y} Q{q}): {e}")
                if fails >= 5:
                    sys.exit("Stopped after 5 consecutive failed calls: check the error above, "
                             "then relaunch the same command to resume.")
                continue
            out.write(json.dumps({"model": a.model, "model_id": tester.model_name, "year": y, "question_idx": q, "masked_letter": letter,
                                  "target": opts[masked_idx], "response": text,
                                  "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")}, ensure_ascii=False) + "\n")
            out.flush()
    print(f"finished: {out_file}")


if __name__ == "__main__":
    main()
