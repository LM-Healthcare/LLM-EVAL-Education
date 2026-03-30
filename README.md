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
└── README.md
```

## Dataset

The `Dataset/` folder contains five Excel files, one per exam year (2020–2024). Each file includes the exam questions in Italian with five answer options (A–E), where **option A is always the correct answer** in the original dataset. The answer order is randomized at runtime by the testing scripts to eliminate positional bias.

Each row in the Excel files has the following columns:
- `Domanda` — question text
- `Risposta A` — answer option A (correct answer)
- `Risposta B` through `Risposta E` — distractor options

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

## How to Run

Each script is self-contained. See the [`Official_testing_code/README.md`](Official_testing_code/README.md) for detailed instructions on parameters, parsing logic, and usage.

### Quick Start (Closed-source example)

```bash
pip install openai pandas openpyxl
python Official_testing_code/Closed/test_gpt_medical_exam_v2.py
```

### Quick Start (Open-weight example)

```bash
pip install transformers torch accelerate pandas openpyxl
python Official_testing_code/Open/test_qwen3_4b_medical_exam.py
```

## Citation

If you use this code or dataset, please cite our paper:

```
@article{TODO,
  title={TODO},
  author={TODO},
  journal={TODO},
  year={2025}
}
```

## License

This repository is released for academic and research purposes.
