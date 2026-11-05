"""
Answer-perturbation analysis ("None of the other answers"), added in the second revision.

Design (adapted from Bedi et al.): the 150 items in items.csv are administered in three
conditions, with the same prompt, generation parameters and answer extraction as in the main
study (the tester classes in Official_testing_code/ are imported and their process_question()
method is called unchanged):

  original         the five original options (control: re-administration today)
  nota_correct     the correct option is replaced by "Nessuna delle altre risposte è corretta",
                   which therefore becomes the correct answer
  nota_distractor  one distractor is replaced by the same NOTA option; the correct option is
                   unchanged (control for the presence of a NOTA option)

Options are shuffled at every administration (seeded, reproducible). A model that answers
from the content of the question should keep its accuracy when the correct option is replaced
by NOTA; a model that relies on recognising a memorised answer among the options should not.

Usage, from the repository root (one model at a time; the run can be interrupted and resumed):
    python Contamination_analysis/run_perturbation.py --model gpt --runs 5
    python Contamination_analysis/run_perturbation.py --model medgemma --runs 3
    python Contamination_analysis/run_perturbation.py --model medgemma --runs 5 \
        --conditions original --decoding sampled          # Reviewer 3, Comment 5

Models: claude, deepseek, gpt, grok, mistral, qwen (API key from the environment variable
shown in MODELS, or asked at start-up); qwen3_8b, medgemma, meditron3, ministral, qwen3_4b,
qwen3_1_7b (local, Hugging Face Transformers, GPU).
--decoding sampled (local models only) replaces greedy decoding in the primary generation with
sampling at temperature 0.1, top-p 0.95, as for the other open-weight models; everything else is
unchanged. It is meant for MedGemma, the only model run with greedy decoding.
Output: Contamination_analysis/results/perturbation_<model>[_sampled].jsonl (one line per response).
"""
import argparse
import getpass
import importlib.util
import json
import os
import random
import sys
import time

import pandas as pd

NOTA = "Nessuna delle altre risposte è corretta"
CONDITIONS = ["original", "nota_correct", "nota_distractor"]
LETTERS = "ABCDE"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CODE = os.path.join(ROOT, "Official_testing_code")

# name: (script, class, kind, environment variable for the API key)
MODELS = {
    "claude": ("Closed/test_claude_medical_exam_v2.py", "ClaudeMedicalExamTester", "api", "ANTHROPIC_API_KEY"),
    "deepseek": ("Closed/test_deepseek_medical_exam_v2.py", "DeepseekMedicalExamTester", "api", "DEEPSEEK_API_KEY"),
    "gpt": ("Closed/test_gpt_medical_exam_v2.py", "GPTMedicalExamTester", "api", "OPENAI_API_KEY"),
    "grok": ("Closed/test_grok_medical_exam_v2.py", "GrokMedicalExamTester", "api", "XAI_API_KEY"),
    "mistral": ("Closed/test_mistral_medical_exam_v2.py", "MistralMedicalExamTester", "api", "MISTRAL_API_KEY"),
    "qwen": ("Closed/test_qwen_medical_exam_v2.py", "QwenMedicalExamTester", "api", "DASHSCOPE_API_KEY"),
    "qwen3_8b": ("Open/test_qwen3_8b_medical_exam.py", "Qwen3_8BMedicalExamTester", "local", None),
    "qwen3_4b": ("Open/test_qwen3_4b_medical_exam.py", "Qwen3_4BInstructMedicalExamTester", "local", None),
    "qwen3_1_7b": ("Open/test_qwen3_1_7b_medical_exam.py", "Qwen3_1_7BMedicalExamTester", "local", None),
    "ministral": ("Open/test_ministral_3b_instruct_medical_exam.py", "Ministral3BInstructMedicalExamTester", "local", None),
    "meditron3": ("Open/test_meditron3_fp16_medical_exam.py", "Meditron3FP16MedicalExamTester", "local", None),
    "medgemma": ("Open/test_medgemma_4b_it_medical_exam.py", "MedGemma4BItMedicalExamTester", "local", None),
}


def load_tester(name):
    script, cls, kind, env = MODELS[name]
    path = os.path.join(CODE, script)
    sys.path.insert(0, os.path.dirname(path))
    spec = importlib.util.spec_from_file_location(f"tester_{name}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    Tester = getattr(mod, cls)
    if kind == "api":
        key = os.environ.get(env) or getpass.getpass(f"API key for {name} ({env}): ")
        return Tester(key)
    if name == "medgemma":
        return Tester(device="auto")
    import torch
    return Tester(device="cuda" if torch.cuda.is_available() else "cpu")


def build_options(item, condition, rng):
    """Return (shuffled options, correct letter, letter of the NOTA option or None)."""
    correct = str(item["correct"])
    distractors = [str(item[f"d{k}"]) for k in range(1, 5)]
    if condition == "nota_correct":
        correct = NOTA
    elif condition == "nota_distractor":
        distractors[int(item["nota_distractor"]) - 1] = NOTA
    opts = [(correct, True)] + [(d, False) for d in distractors]
    rng.shuffle(opts)
    texts = [t for t, _ in opts]
    corr = LETTERS[[c for _, c in opts].index(True)]
    nota = LETTERS[texts.index(NOTA)] if NOTA in texts else None
    return texts, corr, nota


def last_line(path, size_before):
    if not os.path.exists(path) or os.path.getsize(path) <= size_before:
        return None
    with open(path, "rb") as fh:
        fh.seek(size_before)
        lines = [l for l in fh.read().decode("utf-8", "replace").splitlines() if l.strip()]
    try:
        return json.loads(lines[-1]) if lines else None
    except json.JSONDecodeError:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=sorted(MODELS))
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--items", default=os.path.join(ROOT, "Contamination_analysis", "items.csv"))
    ap.add_argument("--conditions", nargs="+", default=CONDITIONS, choices=CONDITIONS)
    ap.add_argument("--decoding", default="default", choices=["default", "sampled"])
    ap.add_argument("--outdir", default=os.path.join(ROOT, "Contamination_analysis", "results"))
    a = ap.parse_args()

    items = pd.read_csv(a.items)
    os.makedirs(a.outdir, exist_ok=True)
    if a.decoding == "sampled" and MODELS[a.model][2] != "local":
        ap.error("--decoding sampled is available only for local models")
    suffix = "_sampled" if a.decoding == "sampled" else ""
    out_file = os.path.join(a.outdir, f"perturbation_{a.model}{suffix}.jsonl")
    done = set()
    if os.path.exists(out_file):
        with open(out_file, encoding="utf-8") as fh:
            for l in fh:
                if l.strip():
                    r = json.loads(l)
                    done.add((r["condition"], r["run"], r["year"], r["question_idx"]))
    total = a.runs * len(items) * len(a.conditions)
    print(f"{a.model}: {len(done)} of {total} responses already done")

    os.chdir(a.outdir)                       # log, checkpoint and raw files of the tester go here
    tester = load_tester(a.model)
    raw_file = os.path.abspath(f"tester_raw_{a.model}{suffix}.jsonl")
    if hasattr(tester, "raw_responses_handle"):          # open-weight testers keep a file handle
        tester.raw_responses_handle.close()
        tester.raw_responses_handle = open(raw_file, "a", encoding="utf-8")
    tester.raw_responses_file = raw_file
    if a.decoding == "sampled":
        generate = tester.model.generate

        def sampled_generate(*args, **kw):
            if not kw.get("do_sample", False):           # primary greedy generation only
                kw.update(do_sample=True, temperature=0.1, top_p=0.95)
            return generate(*args, **kw)
        tester.model.generate = sampled_generate

    n = len(done)
    with open(out_file, "a", encoding="utf-8") as out:
        for run in range(1, a.runs + 1):
            for _, item in items.iterrows():
                y, q = int(item["year"]), int(item["question_idx"])
                for cond in a.conditions:
                    if (cond, run, y, q) in done:
                        continue
                    rng = random.Random(f"{a.model}|{cond}|{run}|{y}|{q}")
                    opts, corr, nota = build_options(item, cond, rng)
                    size = os.path.getsize(raw_file) if os.path.exists(raw_file) else 0
                    answer, elapsed = tester.process_question(
                        question=str(item["question"]), options=opts, correct_answer=corr,
                        year=y, run=run, question_idx=q)
                    raw = last_line(raw_file, size)
                    rec = {"model": a.model, "decoding": a.decoding, "condition": cond, "run": run, "year": y, "question_idx": q,
                           "question_type": item["question_type"],
                           "options": dict(zip(LETTERS, opts)), "correct_letter": corr, "nota_letter": nota,
                           "answer": answer, "is_correct": answer == corr,
                           "chose_nota": nota is not None and answer == nota,
                           "raw_response": raw.get("raw_response") if raw else None,
                           "seconds": round(elapsed, 3), "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")}
                    out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    out.flush()
                    n += 1
                    if n % 50 == 0:
                        print(f"  {n}/{total} done")
    print(f"finished: {out_file}")


if __name__ == "__main__":
    main()
