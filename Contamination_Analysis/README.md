# Contamination_Analysis — benchmark contamination probes (second revision)

Two analyses added in response to Reviewer 3, Comment 1, on a common set of 150 items.

## Items

`items.csv` (built by `select_items.py`, seed 2026): 30 items per examination year, drawn at random from
the text-only questions whose options do not already contain a "none/all of the other answers" option,
whose stem is not negatively worded (the question asks for the exception or the false statement), and
that are not among the seven questions with defective administered options (main README, *Dataset
version*). The item text is that of `Dataset/`.

`finalize_results.py` aligns the result files with `items.csv` (keeps only the listed items, removes
duplicates, sorts the lines) and reports whether every model is complete.

## 1. Answer perturbation (`run_perturbation.py`)

Each item is administered in three conditions, with the prompt, generation parameters and answer
extraction of the main study (the `process_question()` method of the scripts in
`Official_testing_code/` is called unchanged):

| Condition | Options | Correct answer |
|---|---|---|
| `original` | the five original options | original correct option |
| `nota_correct` | correct option replaced by *"Nessuna delle altre risposte è corretta"* | the NOTA option |
| `nota_distractor` | one distractor replaced by the same NOTA option | original correct option |

The NOTA wording is the one used in the SSM itself. Options are shuffled at every administration
(seeded). The `original` condition re-administers the unperturbed items at the same time as the
perturbed ones, so that the comparison is not affected by changes of the API models since February 2026.

```
python Contamination_Analysis/run_perturbation.py --model gpt --runs 5
```

## 2. Option completion, TS-Guessing (`run_ts_guessing.py`, API models only)

One incorrect option of each item is masked and the model is asked to reproduce its exact text
(Deng et al., 2024). Incorrect options cannot be inferred from medical knowledge, so exact or
near-exact reproduction indicates prior exposure to the item.

```
python Contamination_Analysis/run_ts_guessing.py --model gpt
```

## Models no longer available through their original API (`run_openrouter.py`)

Between the main evaluation (February 2026) and the revision (November 2026) three API models
became unavailable through the endpoints used in the study: the DeepSeek alias `deepseek-chat` was
retired on 24 July 2026, the alias `qwen3-max` is no longer served, and Mistral Large 3 is no longer
included in the authors' Mistral subscription tier. `run_openrouter.py` queries the same models through
OpenRouter, reusing the original tester classes unchanged except for endpoint, model identifier and key:

| Model | OpenRouter id | Provider | Notes |
|---|---|---|---|
| DeepSeek V3.2 | `deepseek/deepseek-v3.2` | OpenRouter routing (or `--provider`) | reasoning disabled, as `deepseek-chat` |
| Mistral Large 3 | `mistralai/mistral-large-2512` | Mistral (pinned) | |
| Qwen3-Max | `qwen/qwen3-max` | Alibaba (pinned) | |

The model and provider that served each call are recorded in `results/openrouter_calls.jsonl`.

```
python Contamination_Analysis/run_openrouter.py --check
python Contamination_Analysis/run_openrouter.py --model mistral --task perturbation --runs 5
python Contamination_Analysis/run_openrouter.py --model mistral --task ts_guessing
```

The Mistral and Qwen testers pause 3–4 s and 1–1.5 s between calls (as in the main study), so 2,250
responses take about 2.5 h and 1 h.

## 3. MedGemma under sampled decoding (Reviewer 3, Comment 5)

`run_perturbation.py --model medgemma --conditions original --decoding sampled --runs 5` re-runs
MedGemma 1.5-4B on the original items with sampling at temperature 0.1 and top-p 0.95 (as for the
other open-weight models) instead of greedy decoding. Output: `results/perturbation_medgemma_sampled.jsonl`.

## Analysis (`analyze_contamination.py`)

```
python Contamination_Analysis/analyze_contamination.py
```

Writes to `analysis/`: `perturbation_summary.csv` (accuracy on the 150 items in the main study and in
the three conditions, differences with 95% CIs from an item bootstrap), `perturbation_by_type.csv`,
`ts_guessing_summary.csv` (exact and near-exact reproduction of the masked option, overall and for options
not derivable from the visible ones; accuracy on reproduced vs other items) and, when available,
`sampled_decoding.csv` (MedGemma, greedy vs sampled decoding).

`check_qwen_snapshots.py` documents that Qwen3-Max (`qwen3-max` and its dated snapshots) was no longer
served by Alibaba Model Studio at the time of the revision; Qwen3-Max is therefore not included.

## Running

- From the repository root, one model at a time. Both scripts can be interrupted (Ctrl+C) and
  relaunched with the same command: they resume where they stopped.
- API keys are read from `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `DEEPSEEK_API_KEY`, `XAI_API_KEY`,
  `MISTRAL_API_KEY`, `DASHSCOPE_API_KEY`, or asked at start-up.
- Local models (`qwen3_8b`, `medgemma`, `meditron3`, `ministral`, `qwen3_4b`, `qwen3_1_7b`) need
  the same environment as the main study (GPU, transformers, access to the gated MedGemma weights).
- Outputs: `results/perturbation_<model>.jsonl`, `results/ts_guessing_<model>.jsonl`, plus the
  tester's own raw file and log in `results/` (the files of the API models are archived in
  `results/Closed/`; the analysis scripts search both folders).

| Model | Responses (perturbation) | Indicative time |
|---|---|---|
| each API model, 5 runs | 2,250 | 30–60 min |
| `ts_guessing`, each API model | 150 | 5–15 min |
| `qwen3_8b`, 5 runs | 2,250 | ~1 h |
| `meditron3`, 5 runs | 2,250 | ~30 min |
| `medgemma`, 3 runs | 1,350 | ~6 h |
| `medgemma --conditions original --decoding sampled`, 5 runs (Comment 5) | 750 | ~3 h |
