"""
Audit of answer extraction and manual resolution (Reviewer 1, Comment 4).

Run from the root of the LLM-EVAL-Education repository, after `git lfs pull`:
    python Analysis/audit_extraction.py

Only the Python standard library is needed (no pandas, no GPU, no API calls).

For every model it reads Results/*/Raw_Responses/raw_responses_*.jsonl and, for each
response (year, run, question):
  1. runtime_letter  = letter produced by the ORIGINAL extraction function of that model's
                       evaluation script (re-executed exactly as in Official_testing_code/)
  2. stored_letter   = letter currently stored in the file (after manual resolution)
  3. strict_letter   = letter produced by a revised, conservative extraction procedure
                       (see strict_extract below); "ND" when no unambiguous letter is found

It also recomputes the per-run accuracy implied by the runtime letters and compares it with
the accuracy printed in the execution logs at run time. If they match, the raw_response text
is unchanged since the evaluation.

Outputs (folder Analysis/audit_output/):
  summary_by_model.csv        counts per model
  log_check_by_model.csv      runtime accuracy recomputed vs accuracy in the logs
  audit_items.csv             every response where the three letters are not all identical,
                              or where the strict procedure returns ND (first 300 and last 1500
                              characters of the response)
  gpt_check_items.csv         GPT 5.2 responses to the 7 questions that returned an empty response
                              in the execution log (to document how they were resolved)
"""
import ast
import csv
import glob
import json
import logging
import os
import re
import sys
import unicodedata
from collections import Counter, defaultdict

csv.field_size_limit(10**9)
OUT = os.path.join("Analysis", "audit_output")
os.makedirs(OUT, exist_ok=True)
logging.basicConfig(level=logging.ERROR)
LOGGER = logging.getLogger("audit")

# ---------------------------------------------------------------------------------------
# model key -> (raw responses file, evaluation script, execution log or None)
# ---------------------------------------------------------------------------------------
C, O, Q = "Results/Closed Models", "Results/Open Models", "Results/Quantized models"
S = "Official_testing_code"
MODELS = {
    "Claude 4.5 Sonnet": (f"{C}/Raw_Responses/raw_responses_claude.jsonl", f"{S}/Closed/test_claude_medical_exam_v2.py", f"{C}/Log/claude_medical_exam_test_v2.log"),
    "DeepSeek V3.2": (f"{C}/Raw_Responses/raw_responses_deepseek.jsonl", f"{S}/Closed/test_deepseek_medical_exam_v2.py", f"{C}/Log/deepseek_medical_exam_test_v2.log"),
    "GPT 5.2": (f"{C}/Raw_Responses/raw_responses_gpt.jsonl", f"{S}/Closed/test_gpt_medical_exam_v2.py", f"{C}/Log/gpt_medical_exam_test_v2.log"),
    "Grok 4.1": (f"{C}/Raw_Responses/raw_responses_grok.jsonl", f"{S}/Closed/test_grok_medical_exam_v2.py", f"{C}/Log/grok_medical_exam_test_v2.log"),
    "Mistral Large 3": (f"{C}/Raw_Responses/raw_responses_mistral.jsonl", f"{S}/Closed/test_mistral_medical_exam_v2.py", f"{C}/Log/mistral_medical_exam_test_v2.log"),
    "Qwen 3 Max": (f"{C}/Raw_Responses/raw_responses_qwen.jsonl", f"{S}/Closed/test_qwen_medical_exam_v2.py", f"{C}/Log/qwen_medical_exam_test_v2.log"),
    "Qwen3-1.7B": (f"{O}/Raw_Responses/raw_responses_Qwen3_1_7B.jsonl", f"{S}/Open/test_qwen3_1_7b_medical_exam.py", f"{O}/Log/qwen3_1_7b_medical_exam_test.log"),
    "Ministral-3B": (f"{O}/Raw_Responses/raw_responses_Ministral_3B_Instruct.jsonl", f"{S}/Open/test_ministral_3b_instruct_medical_exam.py", f"{O}/Log/ministral_3b_instruct_medical_exam_test.log"),
    "Qwen3-4B-Instruct": (f"{O}/Raw_Responses/raw_responses_Qwen3_4B_Instruct.jsonl", f"{S}/Open/test_qwen3_4b_medical_exam.py", f"{O}/Log/qwen3_4b_instruct_medical_exam_test.log"),
    "MedGemma 1.5-4B": (f"{O}/Raw_Responses/raw_responses_MedGemma_4B_IT.jsonl", f"{S}/Open/test_medgemma_4b_it_medical_exam.py", f"{O}/Log/medgemma_4b_it_medical_exam_test.log"),
    "Qwen3-8B": (f"{O}/Raw_Responses/raw_responses_Qwen3_8B.jsonl", f"{S}/Open/test_qwen3_8b_medical_exam.py", f"{O}/Log/qwen3_8b_medical_exam_test.log"),
    "Meditron3-8B": (f"{O}/Raw_Responses/raw_responses_Meditron3_8B_FP16.jsonl", f"{S}/Open/test_meditron3_fp16_medical_exam.py", f"{O}/Log/meditron3_fp16_medical_exam_test.log"),
    "MedGemma 1.5-4B-Q8_0": (f"{Q}/Raw_Responses/raw_responses_MedGemma_4B_Q8_0.jsonl", f"{S}/Quantized/test_medgemma_4b_Q8_0_medical_exam.py", f"{Q}/Log/medgemma_4b_Q8_0_medical_exam_test.log"),
    "MedGemma 1.5-4B-Q6_K": (f"{Q}/Raw_Responses/raw_responses_MedGemma_4B_Q6_K.jsonl", f"{S}/Quantized/test_medgemma_4b_Q6_K_medical_exam.py", f"{Q}/Log/medgemma_4b_Q6_K_medical_exam_test.log"),
    "MedGemma 1.5-4B-Q4_K_M": (f"{Q}/Raw_Responses/raw_responses_MedGemma_4B_Q4_K_M.jsonl", f"{S}/Quantized/test_medgemma_4b_Q4_K_M_medical_exam.py", f"{Q}/Log/medgemma_4b_Q4_K_M_medical_exam_test.log"),
    "Meditron3-8B-Q8_0": (f"{Q}/Raw_Responses/raw_responses_Meditron3_8B_Q8_0.jsonl", f"{S}/Quantized/test_meditron3_gguf_medical_exam.py", None),
    "Meditron3-8B-Q6_K": (f"{Q}/Raw_Responses/raw_responses_Meditron3_8B_Q6_K.jsonl", f"{S}/Quantized/test_meditron3_gguf_medical_exam.py", None),
    "Meditron3-8B-Q4_K_M": (f"{Q}/Raw_Responses/raw_responses_Meditron3_8B_Q4_K_M.jsonl", f"{S}/Quantized/test_meditron3_gguf_medical_exam.py", None),
}
LETTERS = "ABCDE"


# ---------------------------------------------------------------------------------------
# 1. Re-create the ORIGINAL extraction function of each script (no model loading)
# ---------------------------------------------------------------------------------------
def load_original_extractor(script_path):
    """Extract `_extract_letter` and every helper method it calls from the script's class,
    and build a light-weight object exposing them. Heavy imports (torch, APIs) are skipped."""
    src = open(script_path, encoding="utf-8").read()
    tree = ast.parse(src)
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and any(
        isinstance(f, ast.FunctionDef) and f.name == "_extract_letter" for f in n.body))
    methods = {f.name: f for f in cls.body if isinstance(f, ast.FunctionDef)}
    needed, todo = set(), ["_extract_letter"]
    while todo:
        name = todo.pop()
        if name in needed or name not in methods:
            continue
        needed.add(name)
        for node in ast.walk(methods[name]):
            if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id == "self":
                todo.append(node.attr)
    # module-level constants / helper functions used by the methods (e.g. regex lists)
    module_defs = [n for n in tree.body if isinstance(n, (ast.Assign, ast.FunctionDef))
                   and not (isinstance(n, ast.Assign) and any(isinstance(t, ast.Attribute) for t in n.targets))]
    new_cls = ast.ClassDef(name="Extractor", bases=[], keywords=[], decorator_list=[],
                           body=[methods[n] for n in sorted(needed)])
    mod = ast.Module(body=[*module_defs, new_cls], type_ignores=[])
    ast.fix_missing_locations(mod)
    import typing, difflib, string
    ns = {"re": re, "logging": logging, "logger": LOGGER, "unicodedata": unicodedata,
          "difflib": difflib, "string": string, "os": os, "json": json, "Counter": Counter,
          **{k: getattr(typing, k) for k in ("List", "Dict", "Optional", "Tuple", "Any")}}
    safe_defs = []
    for node in mod.body:  # drop module-level statements that fail without heavy deps
        try:
            exec(compile(ast.Module(body=[node], type_ignores=[]), script_path, "exec"), ns)
            safe_defs.append(node)
        except Exception:
            pass
    obj = ns["Extractor"]()
    obj.logger = LOGGER
    return obj


def call_original(ex, text, options_list):
    try:
        return ex._extract_letter(text, options_list)
    except TypeError:
        return ex._extract_letter(text)


# ---------------------------------------------------------------------------------------
# 2. Revised, conservative extraction (applied identically to ALL responses)
# ---------------------------------------------------------------------------------------
THINK_PATTERNS = [
    r"<think>.*?</think>",
    r"<unused94>.*?<unused95>",
    r"<unused94>thought.*?</thought>",
]


def strip_reasoning(text):
    t = text or ""
    for p in THINK_PATTERNS:
        t = re.sub(p, " ", t, flags=re.S | re.I)
    t = re.sub(r"<unused\d+>", " ", t)
    return t


def norm(s):
    s = unicodedata.normalize("NFKC", s or "").lower()
    s = re.sub(r"[\*`_\"“”'’]", "", s)
    return re.sub(r"\s+", " ", s).strip()


L = r"([A-E])"  # letters must be UPPERCASE
STRICT_PATTERNS = [
    # 1. response starts with the letter followed by punctuation or nothing:
    #    "B", "B:", "B)", "(B)", "B.", "**B**"  (not "A causa di…", "E' …")
    rf"^\s*[\*\(\[]*\s*{L}\s*(?:[\*\)\]:.\-]|$)",
    # 2. explicit answer statements (English / Italian)
    rf"(?i:final answer|correct answer|answer|risposta corretta|risposta|opzione|option)"
    rf"\s*(?i:is|è|:|=)?\s*[\*\(\[]*\s*{L}(?![A-Za-z])",
    # 3. bold or bracketed standalone letter anywhere: **B**, (B), [B]
    rf"(?:\*\*|\(|\[)\s*{L}\s*(?:\*\*|\)|\])",
]


def strict_extract(text, options):
    """Return (letter or 'ND', rule). Conservative: never guesses from lower-case letters or
    arbitrary words. Explicit answer statements take precedence and the LAST one is used
    (reasoning traces often mention several letters before the final answer)."""
    t = strip_reasoning(text).strip()
    if not t:
        return "ND", "empty"
    # 1. explicit answer statements -> last occurrence
    explicit = re.findall(STRICT_PATTERNS[1], t)
    if explicit:
        return explicit[-1], "explicit_answer_last"
    # 2. response starts with the letter
    m = re.search(STRICT_PATTERNS[0], t)
    if m:
        return m.group(1), "starts_with_letter"
    # 3. bold / bracketed letters: accept only if a single distinct letter appears
    br = set(re.findall(STRICT_PATTERNS[2], t))
    if len(br) == 1:
        return br.pop(), "bracketed_letter"
    if len(br) > 1:
        return "ND", "conflict_bracketed:" + "".join(sorted(br))
    # 4. option-text match: exactly one option text contained in the response
    if options:
        nt = norm(t)
        hits = [k for k, v in options.items() if v and len(norm(v)) >= 3 and norm(v) in nt]
        if len(hits) == 1:
            return hits[0], "option_text"
        if len(hits) > 1:
            return "ND", "conflict_option_text"
    # 5. a single standalone uppercase letter in a short response
    if len(t) <= 40:
        solo = set(re.findall(r"(?<![A-Za-z])([A-E])(?![A-Za-z])", t))
        if len(solo) == 1:
            return solo.pop(), "short_single_letter"
    return "ND", "no_match"


# ---------------------------------------------------------------------------------------
# 3. Runtime accuracy printed in the logs
# ---------------------------------------------------------------------------------------
def log_accuracy(path):
    if not path or not os.path.exists(path):
        return {}
    cur, res = None, {}
    for line in open(path, encoding="utf-8", errors="ignore"):
        m = re.search(r"(?:Anno|Year) (\d{4}) - Run (\d+)/50", line)
        if m:
            cur = int(m[1]); continue
        m = re.search(r"Run (\d+) (?:completata|completed) - Accuracy: ([\d.]+)%", line)
        if m and cur:
            res[(cur, int(m[1]))] = float(m[2])
    return res


# ---------------------------------------------------------------------------------------
# 4. Main loop
# ---------------------------------------------------------------------------------------
summary, logcheck = [], []
audit_f = open(f"{OUT}/audit_items.csv", "w", newline="", encoding="utf-8")
aw = csv.writer(audit_f)
aw.writerow(["model", "year", "run", "question_idx", "timestamp", "correct_answer", "runtime_letter", "stored_letter",
             "strict_letter", "strict_rule", "response_length", "response_head_300", "response_tail_1500"])
gpt_f = open(f"{OUT}/gpt_check_items.csv", "w", newline="", encoding="utf-8")
gw = csv.writer(gpt_f)
gw.writerow(["year", "run", "question_idx", "timestamp", "correct_answer", "stored_letter", "raw_response"])
# questions for which the GPT log records an empty response in (almost) every run: (year, question number 1-based)
GPT_EMPTY = {(2020, 8), (2020, 116), (2021, 139), (2022, 100), (2023, 89), (2023, 116), (2023, 120)}


for model, (raw_path, script, log_path) in MODELS.items():
    if not os.path.exists(raw_path) or os.path.getsize(raw_path) < 1000:
        print(f"[skip] {model}: {raw_path} missing or still a Git LFS pointer (run `git lfs pull`)")
        continue
    print(f"[run ] {model}")
    ex = load_original_extractor(script)
    entries = {}
    n_lines = 0
    with open(raw_path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            n_lines += 1
            e = json.loads(line)
            entries[(int(e["year"]), int(e["run"]), int(e["question_idx"]))] = e  # keep last (resumed runs)

    cnt = Counter()
    per_run_rt = defaultdict(lambda: [0, 0])
    for (y, r, q), e in sorted(entries.items()):
        raw = e.get("raw_response") or ""
        opts = e.get("options") or {}
        olist = [opts.get(k, "") for k in LETTERS]
        corr = e.get("correct_answer")
        stored = e.get("extracted_letter")
        try:
            rt = call_original(ex, raw, olist)
        except Exception:
            rt = "ERR"
        st, rule = strict_extract(raw, opts)

        cnt["n"] += 1
        cnt["empty_response"] += (strip_reasoning(raw).strip() == "")
        cnt["runtime_ND"] += (rt == "ND")
        cnt["runtime_correct"] += (rt == corr)
        cnt["stored_correct"] += (stored == corr)
        cnt["strict_correct"] += (st == corr)
        cnt["strict_ND"] += (st == "ND")
        cnt["stored_ne_runtime"] += (stored != rt)
        cnt["stored_changed_wrong_to_correct"] += (stored != rt and stored == corr and rt != corr)
        cnt["stored_changed_correct_to_wrong"] += (stored != rt and rt == corr and stored != corr)
        cnt["stored_changed_wrong_to_wrong"] += (stored != rt and rt != corr and stored != corr)
        cnt["strict_ne_stored"] += (st != "ND" and st != stored)
        cnt["strict_ne_runtime"] += (st != "ND" and st != rt)
        per_run_rt[(y, r)][0] += (rt == corr)
        per_run_rt[(y, r)][1] += 1

        if not (rt == stored == st):
            aw.writerow([model, y, r, q, e.get("timestamp", ""), corr, rt, stored, st, rule, len(raw), raw[:300], raw[-1500:]])
        if model == "GPT 5.2" and ((y, q) in GPT_EMPTY or (y, q + 1) in GPT_EMPTY):
            gw.writerow([y, r, q, e.get("timestamp", ""), corr, stored, raw[:1000]])

    logacc = log_accuracy(log_path)
    diffs = [abs(100 * c / n - logacc[k]) for k, (c, n) in per_run_rt.items() if k in logacc and n]
    logcheck.append({
        "model": model, "runs_compared": len(diffs),
        "runs_matching_log_(<0.01pp)": sum(d < 0.01 for d in diffs),
        "max_abs_diff_pp": round(max(diffs), 3) if diffs else "",
    })
    row = {"model": model, "lines_in_file": n_lines, "unique_responses": cnt["n"]}
    for k in ["empty_response", "runtime_ND", "stored_ne_runtime", "stored_changed_wrong_to_correct",
              "stored_changed_correct_to_wrong", "stored_changed_wrong_to_wrong", "strict_ND",
              "strict_ne_stored", "strict_ne_runtime"]:
        row[k] = cnt[k]
    for k in ["runtime", "stored", "strict"]:
        row[f"{k}_accuracy_%"] = round(100 * cnt[f"{k}_correct"] / cnt["n"], 2) if cnt["n"] else ""
    summary.append(row)

audit_f.close()
gpt_f.close()
for name, rows in [("summary_by_model.csv", summary), ("log_check_by_model.csv", logcheck)]:
    if rows:
        with open(f"{OUT}/{name}", "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader(); w.writerows(rows)
print(f"\nDone. Send the folder '{OUT}/' back.")
for r in summary:
    print(f"{r['model']:<24} runtime {r['runtime_accuracy_%']:>6}  stored {r['stored_accuracy_%']:>6}  "
          f"strict {r['strict_accuracy_%']:>6}  manual changes {r['stored_ne_runtime']:>5}  strict ND {r['strict_ND']:>5}")
