# LLM-EVAL-Education

Evaluation of Large Language Models on the Italian Medical Specialization Entrance Exam (SSM – *Esame per l'Accesso alle Scuole di Specializzazione Medica*).

This repository contains the code, dataset, and complete evaluation outputs used in our study, which benchmarks 18 LLMs (6 proprietary API-based models, 6 open-weight models, and 6 quantized variants of medically fine-tuned models) on real multiple-choice questions from the Italian national medical residency entrance exams (2020–2024).

## Repository Structure

```
.
|── Analysis
|   
|── Candidate_analysis/          # Anonymised human-candidate scores and human-referenced analysis
|   ├── data/
│   ├── output/
│   ├── script/
│   └── README.md
|
├── Dataset/                      # Exam questions (2020-2024)
│   ├── 2020_SSM_[checked].xlsx
│   ├── 2021_SSM_[checked].xlsx
│   ├── 2022_SSM_[checked].xlsx
│   ├── 2023_SSM_[checked].xlsx
│   └── 2024_SSM_[checked].xlsx
│
├── Official_testing_code/        # Evaluation scripts for all models (see its own README)
│   ├── Closed/                   # Proprietary / API-based models
│   ├── Open/                     # Open-weight models (Hugging Face Transformers)
│   └── Quantized/                # GGUF quantized models (llama-cpp-python)
│
├── Results/                      # Full evaluation outputs
│   ├── Closed Models/
│   ├── Open Models/
│   └── Quantized models/
│       ├── Consistency/          # Per-question answer tracking (.xlsx)
│       ├── Log/                  # Execution logs (.log)
│       ├── Metrics/              # Per-run accuracy & scores (.xlsx)
│       ├── Raw_Responses/        # Raw model outputs (.jsonl, Git LFS)
│       └── Results/              # Aggregated results (.json)
│
└── README.md
```

## Dataset

The `Dataset/` folder contains five Excel files, one per exam year (2020–2024), each with 140 questions (700 in total). Questions are transcribed verbatim in Italian from the official Ministry of University and Research (MUR) documents, with five answer options (A–E). **Option A is always the correct answer** in the stored dataset; the answer order is randomized at runtime by the testing scripts (independently for every question and every repetition) to eliminate positional bias.

Each row in the Excel files has the following columns:

| Column | Description |
|---|---|
| `Question Code` | Unique question identifier |
| `Question` | Question text (Italian) |
| `Answer A` | Correct answer |
| `Answer B` – `Answer E` | Distractor options |
| `Category` | Disciplinary area(s) of the question (multiple areas separated by `;`) |
| `Question type` | `knowledge-based` or `case-based` |
| `Image` | `Yes` if the original question included a clinical image or figure, `No` otherwise |
| `Image Category` | Type of visual content for image-dependent questions (e.g., ECG tracing, radiological image); empty otherwise |

**Image-dependent questions** were administered to all models in text-only form: images were not provided, and models were not informed that the original question contained visual content.

## Results

The `Results/` folder contains the complete evaluation outputs for all 18 models, organized into three subcategories mirroring the code structure: `Closed Models/`, `Open Models/`, and `Quantized models/`. Each subcategory contains five subfolders:

| Subfolder | Format | Description |
|---|---|---|
| `Consistency/` | `.xlsx` | Per-question answer tracking across all 250 runs (5 years × 50 repetitions). Each row is a run (`{year}_run_{n}`); each column is a question (`Domanda_1` … `Domanda_140`). Values are `1` (correct), the original wrong letter, or `ND` (automated extraction failure). |
| `Log/` | `.log` | Full execution logs with timestamps, per-question details, and error traces (see *Notes on execution logs* below). |
| `Metrics/` | `.xlsx` | Per-run summary metrics including accuracy (%), SSM score (with −0.25 penalty for wrong answers), total time, and average response time per question. |
| `Raw_Responses/` | `.jsonl` | One JSON object per question per run, containing the raw model output, the question and options as presented, extracted answer, timing, and correctness. Stored with Git LFS. |
| `Results/` | `.json` | Aggregated results per model with overall statistics across all years and runs, including the completion timestamp. |

### File Naming Convention

Files follow the pattern `{type}_{model_name}.{ext}`, e.g.:
- `consistency_claude.xlsx`, `metrics_grok.xlsx` (closed models)
- `results_Qwen3_8B.json`, `raw_responses_Meditron3_8B_FP16.jsonl` (open models)
- `metrics_MedGemma_4B_Q4_K_M.xlsx`, `results_Meditron3_8B_Q6_K.json` (quantized models)

## Models Evaluated

Evaluation dates are taken from the execution logs (first run start → last run completion) and, where no log is archived, from the `test_completed` timestamp in the corresponding `Results/*.json` file. All evaluations were performed between 4 and 27 February 2026.

### Proprietary (API-based)

| Model | API model identifier | Provider | Evaluation period |
|---|---|---|---|
| Claude 4.5 Sonnet | `claude-sonnet-4-5-20250929` | Anthropic | 4–13 Feb 2026 |
| DeepSeek V3.2 | `deepseek-chat`* | DeepSeek | 11–12 Feb 2026 |
| GPT 5.2 | `gpt-5.2-2025-12-11` | OpenAI | 11 Feb 2026 |
| Grok 4.1 Fast (Reasoning) | `grok-4-1-fast-reasoning` | xAI | 14–16 Feb 2026 |
| Mistral Large 3 | `mistral-large-2512` | Mistral AI | 14–16 Feb 2026 |
| Qwen3-Max | `qwen3-max` | Alibaba Cloud (DashScope) | 14–16 Feb 2026 |

\* `deepseek-chat` is a provider alias; during the evaluation period it corresponded to DeepSeek-V3.2 in non-thinking mode.

### Open-Weight (Hugging Face Transformers)

| Model | Hugging Face ID | Precision | Evaluation period |
|---|---|---|---|
| Qwen3-1.7B | `Qwen/Qwen3-1.7B` | FP16 | 12–13 Feb 2026 |
| Ministral-3-3B-Instruct | `mistralai/Ministral-3-3B-Instruct-2512` | FP16 | 12 Feb 2026 |
| Qwen3-4B-Instruct | `Qwen/Qwen3-4B-Instruct-2507` | FP16 | 11 Feb 2026 |
| MedGemma 1.5 4B IT | `google/medgemma-1.5-4b-it` | BF16 | 13–27 Feb 2026 |
| Qwen3-8B | `Qwen/Qwen3-8B` | FP16 | 11–12 Feb 2026 |
| Meditron3-8B | `OpenMeditron/Meditron3-8B` | BF16 | 10 Feb 2026 |

### Quantized (GGUF via llama-cpp-python)

| Model | Quantization | GGUF file | Hugging Face repository | Evaluation period |
|---|---|---|---|---|
| MedGemma 1.5 4B IT | Q8_0 | `medgemma-1.5-4b-it.Q8_0.gguf` | `mradermacher/medgemma-1.5-4b-it-GGUF` | 14–27 Feb 2026 |
| MedGemma 1.5 4B IT | Q6_K | `medgemma-1.5-4b-it.Q6_K.gguf` | `mradermacher/medgemma-1.5-4b-it-GGUF` | 14–24 Feb 2026 |
| MedGemma 1.5 4B IT | Q4_K_M | `medgemma-1.5-4b-it.Q4_K_M.gguf` | `mradermacher/medgemma-1.5-4b-it-GGUF` | 13–23 Feb 2026 |
| Meditron3-8B | Q8_0 | `Meditron3-8B.Q8_0.gguf` | `QuantFactory/Meditron3-8B-GGUF` | 10 Feb 2026† |
| Meditron3-8B | Q6_K | `Meditron3-8B.Q6_K.gguf` | `QuantFactory/Meditron3-8B-GGUF` | 10 Feb 2026† |
| Meditron3-8B | Q4_K_M | `Meditron3-8B.Q4_K_M.gguf` | `QuantFactory/Meditron3-8B-GGUF` | 10 Feb 2026† |

† Execution logs for the Meditron3 GGUF runs were not archived; the date is the `test_completed` timestamp recorded in the corresponding `Results/*.json` file. All other outputs (consistency, metrics, raw responses, aggregated results) are available.

## Generation Parameters (summary)

Full details are given in `Official_testing_code/README.md` and in each script.

| Model group | Temperature | top-p | Max output tokens |
|---|---|---|---|
| Proprietary (API) | 0.1 | not set (provider default) | 50 |
| Qwen3, Ministral (Transformers) | 0.1 | 0.95 | 50 (2,048 if a reasoning block is detected) |
| Meditron3-8B (Transformers) | 0.1 | 0.95 | 100 (2,048 if a reasoning block is detected) |
| Meditron3-8B GGUF | 0.1 | 0.95 | 50 (2,048 if a reasoning block is detected) |
| MedGemma 1.5 4B IT (Transformers and GGUF) | greedy decoding (no sampling) | – | 2,048 |
| Retry after failed extraction (open-weight and quantized only) | 0.01 | 0.95 | 20 |

## Exploratory item-type analysis
The `Analysis/` folder contains the script used for the exploratory item-type analysis reported in the paper (ERQ, Supplementary Table S6) and its outputs. Run it from the repository root:
python Analysis/stratified_itemtype_analysis.py
The script reads the `Question type` and `Image` columns of the dataset and the `Consistency/` files of each model, computes the accuracy of each question across the 50 repetitions, and estimates the difference between knowledge-based and case-based items (text-only questions only) and between text-only and image-dependent items, with 95% bootstrap confidence intervals obtained by resampling questions (4,000 resamples, seed 2026). Outputs: `itemtype_results_by_model.csv` and `itemtype_results_by_group.csv`.

## Answer extraction audit
`Analysis/audit_extraction.py` audits answer extraction for all 630,000 responses. Run it from the repository root after `git lfs pull`: `python Analysis/audit_extraction.py`. For each response it compares (i) the letter produced by the original extraction function of the corresponding script in `Official_testing_code/`, re-executed on the archived raw output; (ii) the letter stored in the final dataset; and (iii) the letter obtained with a conservative re-extraction that accepts only an explicitly stated answer or the text of a single option. Outputs in `Analysis/audit_output/`: `summary_by_model.csv` (counts and accuracies per model), `audit_items.csv` (every response for which the three letters differ, with the beginning and end of the model output), `gpt_check_items.csv` (the re-administered GPT 5.2 responses, see below), and `log_check_by_model.csv` (per-run accuracy recomputed from the raw outputs compared with the accuracy printed in the execution logs). `Analysis/manual_resolution_review_R2.xlsx` contains the independent second review of all manually resolved responses.


## Notes on execution logs

Logs are archived exactly as produced at run time and have not been edited. The following points help their interpretation:

- **Header banners.** The banner line printed at start-up (`INIZIALIZZAZIONE TEST ...`) is a hard-coded string inherited from earlier versions of the scripts and was not always updated. In particular, the first header in `claude_medical_exam_test_v2.log` (an aborted start-up attempt on 4 Feb 2026, 15:41, before any question was sent) reads "CLAUDE 3.7 SONNET", and the headers in `gpt_medical_exam_test_v2.log` read "GPT-4.1". The model actually queried is the one reported in the line immediately below (`Inizializzato tester con modello ...`), i.e. `claude-sonnet-4-5-20250929` and `gpt-5.2-2025-12-11`, respectively.
- **Mistral model identifier.** At run time, the evaluation script contained the identifier string mistral-large-2411, which is therefore recorded in mistral_medical_exam_test_v2.log and results_mistral.json. The provider usage records for the evaluation period (14–16 February 2026) confirm that the requests were served by mistral-large-2512 (Mistral Large 3), the model reported in the paper. The script has since been corrected; the log and result files are kept as originally produced.
- **Aborted start-up attempts.** Some logs begin with short aborted attempts (e.g., `No such file or directory` errors caused by local dataset paths). These attempts did not produce any response; the evaluation starts at the first `Run 1/50` entry.
- **Language.** Closed-model logs are partly in Italian (e.g., *Anno* = year, *Processando domanda* = processing question, *completata* = completed).
- **GPT 5.2 re-administered items.** Seven items (2020: Q8, Q116; 2021: Q139; 2022: Q100; 2023: Q89, Q116, Q120) returned an empty output in all 50 repetitions, as recorded in `gpt_medical_exam_test_v2.log` ("Non è stato possibile estrarre una lettera dalla risposta: ''"), because internal reasoning consumed the 50-token output budget. Immediately after the main evaluation, these 350 responses were re-administered with the same prompt and option order and a larger maximum output length. The corresponding entries of `raw_responses_gpt.jsonl` contain the re-administered outputs and keep the timestamps of the original administration. The temporary script used for the re-administration was not archived.
- **Grok 4.1 re-extraction.** Grok frequently answered in a bracketed format (`[B]: …`) that the extraction function in `test_grok_medical_exam_v2.py` does not recognise. The function was updated during the evaluation, and all Grok responses were re-extracted with the final rule after completion (10,080 responses changed); the accuracy printed in the log therefore differs from the final results. The conservative re-extraction in `Analysis/audit_extraction.py` reproduces the final Grok letters for all 35,000 responses.
- **Checkpoint resumption.** When a run was interrupted, the questions after the last checkpoint (at most 10) were administered again on resumption. For this reason, the per-run accuracy printed in some logs can differ from the accuracy computed from the raw output files by a few responses (see `Analysis/audit_output/log_check_by_model.csv`).


## License

This repository is released for academic and research purposes.