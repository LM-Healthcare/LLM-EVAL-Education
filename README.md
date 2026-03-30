# LLM-EVAL-Education

Evaluation of Large Language Models on the Italian Medical Specialization Entrance Exam (SSM - Specializzazione in Medicina).

This repository contains the code and dataset used in our study, which benchmarks both closed-source (API-based) and open-weight LLMs on real multiple-choice questions from the Italian national medical residency entrance exams (2020–2024).

## Repository Structure

```
.
├── Dataset/                      # Exam questions (2020-2024)
│   ├── 2020_SSM_[checked].xlsx
│   ├── 2021_SSM_[checked].xlsx
│   ├── 2022_SSM_[checked].xlsx
│   ├── 2023_SSM_[checked].xlsx
│   └── 2024_SSM_[checked].xlsx
│
├── Official_testing_code/        # Evaluation scripts for all models
│   ├── Closed/                   # Closed-source / API-based models
│   ├── Open/                     # Open-weight models (HuggingFace)
│   └── Quantized/                # GGUF quantized models (llama-cpp)
│
├── Results/                      # Full evaluation outputs
│   ├── Closed Models/
│   ├── Open Models/
│   └── Quantized models/
│       ├── Consistency/          # Per-question answer tracking (.xlsx)
│       ├── Log/                  # Execution logs (.log)
│       ├── Metrics/              # Per-run accuracy & scores (.xlsx)
│       ├── Raw_Responses/        # Raw model outputs (.jsonl)
│       └── Results/              # Aggregated results (.json)
│
└── README.md
```

## Dataset

The `Dataset/` folder contains five Excel files, one per exam year (2020–2024). Each file includes the exam questions in Italian with five answer options (A–E), where **option A is always the correct answer** in the original dataset. The answer order is randomized at runtime by the testing scripts to eliminate positional bias.

Each row in the Excel files has the following columns:
- `Domanda` — question text
- `Risposta A` — answer option A (correct answer)
- `Risposta B` through `Risposta E` — distractor options

## Results

The `Results/` folder contains the complete evaluation outputs for all models, organized into three subcategories mirroring the code structure: `Closed Models/`, `Open Models/`, and `Quantized models/`. Each subcategory contains five subfolders:

| Subfolder | Format | Description |
|---|---|---|
| `Consistency/` | `.xlsx` | Per-question answer tracking across all 250 runs (5 years × 50 repetitions). Each row is a question; each column is a run. Values are `1` (correct), the original wrong letter, or `ND` (extraction failure). |
| `Log/` | `.log` | Full execution logs with timestamps, per-question details, and error traces. |
| `Metrics/` | `.xlsx` | Per-run summary metrics including accuracy (%), SSM score (with −0.25 penalty for wrong answers), total time, and average response time per question. |
| `Raw_Responses/` | `.jsonl` | One JSON object per question per run, containing the raw model output, the prompt sent, shuffled option mapping, extracted answer, timing, and correctness. |
| `Results/` | `.json` | Aggregated results per model with overall statistics across all years and runs. |

### File Naming Convention

Files follow the pattern `{type}_{model_name}.{ext}`, e.g.:
- `consistency_claude.xlsx`, `metrics_grok.xlsx` (closed models)
- `results_Qwen3_8B.json`, `raw_responses_Meditron3_8B_FP16.jsonl` (open models)
- `metrics_MedGemma_4B_Q4_K_M.xlsx`, `results_Meditron3_8B_Q6_K.json` (quantized models)

## Models Evaluated

### Closed-Source (API-based)
| Model | API Provider |
|---|---|
| GPT-5.2 | OpenAI |
| Claude 4.5 Sonnet | Anthropic |
| DeepSeek-Chat | DeepSeek |
| Grok-4.1 Fast Reasoning | xAI |
| Mistral Large (2411) | Mistral AI |
| Qwen3-Max | Alibaba (DashScope) |

### Open-Weight (HuggingFace Transformers)
| Model | HuggingFace ID |
|---|---|
| Meditron3-8B (FP16) | `OpenMeditron/Meditron3-8B` |
| MedGemma 1.5 4B IT | `google/medgemma-1.5-4b-it` |
| Qwen3-1.7B | `Qwen/Qwen3-1.7B` |
| Qwen3-4B-Instruct | `Qwen/Qwen3-4B-Instruct-2507` |
| Qwen3-8B | `Qwen/Qwen3-8B` |
| Ministral-3-3B-Instruct | `mistralai/Ministral-3-3B-Instruct-2512` |

### Quantized (GGUF via llama-cpp-python)
| Model | Quantization |
|---|---|
| MedGemma 1.5 4B IT | Q4_K_M |
| MedGemma 1.5 4B IT | Q6_K |
| MedGemma 1.5 4B IT | Q8_0 |
| Meditron3-8B | GGUF |

## License

This repository is released for academic and research purposes.
