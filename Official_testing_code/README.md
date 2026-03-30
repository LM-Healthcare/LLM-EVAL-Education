# Official Testing Code

This folder contains the evaluation scripts used to benchmark LLMs on the Italian Medical Specialization Entrance Exam (SSM). All scripts share a common methodology but differ in how they interface with each model (API calls vs. local inference).

## Folder Structure

```
Official_testing_code/
├── Closed/          # API-based models (GPT, Claude, DeepSeek, Grok, Mistral, Qwen)
├── Open/            # Open-weight models via HuggingFace Transformers
├── Quantized/       # GGUF quantized models via llama-cpp-python
└── README.md
```

---

## Common Test Methodology

All scripts follow the same evaluation protocol:

- **Years tested**: 2020, 2021, 2022, 2023, 2024
- **Repetitions per year**: 50 runs
- **Total runs per model**: 250 (5 years × 50 runs)
- **Answer shuffling**: options are randomly shuffled at each run to eliminate positional bias (option A is always correct in the original dataset)
- **Checkpoint system**: progress is saved every 10 questions and can be resumed after interruptions
- **Output format**: JSON results, Excel files for consistency tracking and per-run metrics
- **Scoring**: `score = correct - (wrong × 0.25)` (matching the official SSM penalty-based scoring)

---

## Generation Parameters

### Closed-Source Models (API-based)

All closed-source models use **identical** generation parameters:

| Parameter | Value | Notes |
|---|---|---|
| `temperature` | `0.1` | Low temperature for near-deterministic output with slight variability across runs |
| `max_tokens` | `50` | Sufficient for the expected answer format (`LETTER: option text`) |

| Script | Model | API |
|---|---|---|
| `test_gpt_medical_exam_v2.py` | `gpt-5.2-2025-12-11` | OpenAI (`openai` SDK) |
| `test_claude_medical_exam_v2.py` | `claude-sonnet-4-5-20250929` | Anthropic (`anthropic` SDK) |
| `test_deepseek_medical_exam_v2.py` | `deepseek-chat` | DeepSeek REST API |
| `test_grok_medical_exam_v2.py` | `grok-4-1-fast-reasoning` | xAI REST API |
| `test_mistral_medical_exam_v2.py` | `mistral-large-2411` | Mistral AI REST API |
| `test_qwen_medical_exam_v2.py` | `qwen3-max` | DashScope (Alibaba) REST API |

**GPT** uses `max_completion_tokens=50` instead of `max_tokens=50` (OpenAI SDK naming convention).

### Open-Weight Models (HuggingFace Transformers)

Open-weight models use a **two-stage generation strategy**:

#### Stage 1 — Primary generation

| Parameter | Value | Notes |
|---|---|---|
| `temperature` | `0.1` | Low temperature, same rationale as closed models |
| `max_new_tokens` | `50` | Short answer expected |
| `top_p` | `0.95` | Nucleus sampling |
| `do_sample` | `True` | Enables stochastic sampling |

If the model enters **thinking mode** (detected via `<think>` or `<unused94>` tokens), the generation is retried with `max_new_tokens=2048` to allow the model to complete its reasoning chain.

#### Stage 2 — Retry on extraction failure

If letter extraction fails after Stage 1, a more constrained retry prompt is used:

| Parameter | Value |
|---|---|
| `temperature` | `0.01` |
| `max_new_tokens` | `20` |
| `top_p` | `0.95` |
| `do_sample` | `True` |

**Exception — MedGemma 1.5 4B IT (native precision):**

MedGemma uses full reasoning mode natively and is configured differently:

| Parameter | Value | Notes |
|---|---|---|
| `max_new_tokens` | `2048` | Allows full reasoning chain |
| `do_sample` | `False` | Greedy decoding |
| `temperature` | N/A | Not applicable with greedy decoding |

Retry uses `temperature=0.01`, `max_new_tokens=20`, `do_sample=True`.

| Script | Model | HuggingFace ID | Precision |
|---|---|---|---|
| `test_meditron3_fp16_medical_exam.py` | Meditron3-8B | `OpenMeditron/Meditron3-8B` | BF16 |
| `test_medgemma_4b_it_medical_exam.py` | MedGemma 1.5 4B IT | `google/medgemma-1.5-4b-it` | BF16 |
| `test_qwen3_1_7b_medical_exam.py` | Qwen3-1.7B | `Qwen/Qwen3-1.7B` | FP16 |
| `test_qwen3_4b_medical_exam.py` | Qwen3-4B-Instruct | `Qwen/Qwen3-4B-Instruct-2507` | FP16 |
| `test_qwen3_8b_medical_exam.py` | Qwen3-8B | `Qwen/Qwen3-8B` | FP16 |
| `test_ministral_3b_instruct_medical_exam.py` | Ministral-3-3B-Instruct | `mistralai/Ministral-3-3B-Instruct-2512` | FP16 |

### Quantized Models (GGUF via llama-cpp-python)

#### MedGemma GGUF variants (Q4_K_M, Q6_K, Q8_0)

| Parameter | Value | Notes |
|---|---|---|
| `temperature` | `0` | Fully deterministic primary generation |
| `max_tokens` | `2048` | Full reasoning chain supported |

Retry: `temperature=0.01`, `max_tokens=20`, `top_p=0.95`.

#### Meditron3 GGUF

| Parameter | Value | Notes |
|---|---|---|
| `temperature` | `0.1` | Same as other Meditron3 variants |
| `max_tokens` | `50` | Short answer expected |
| `top_p` | `0.95` | Nucleus sampling |

If thinking mode is detected, retried with `max_tokens=2048`.

| Script | Model | Quantization |
|---|---|---|
| `test_medgemma_4b_Q4_K_M_medical_exam.py` | MedGemma 1.5 4B IT | Q4_K_M |
| `test_medgemma_4b_Q6_K_medical_exam.py` | MedGemma 1.5 4B IT | Q6_K |
| `test_medgemma_4b_Q8_0_medical_exam.py` | MedGemma 1.5 4B IT | Q8_0 |
| `test_meditron3_gguf_medical_exam.py` | Meditron3-8B | GGUF |

---

## Prompt Design

All models receive the same core prompt structure. The specific formatting varies by model family to match their expected chat template.

### Closed-Source Models

**System message:**
```
You are an expert physician taking a medical specialization exam in Italy.
Carefully analyze the question and each possible answer.
Respond ONLY with the letter corresponding to the correct answer (A, B, C, D, or E).
Do not provide explanations or additional text.
```

**User message:**
```
You are a highly skilled medical doctor taking a specialization entrance exam in Italy.
Carefully analyze the following question and possible answers in Italian.
Choose the correct answer among the provided options.

QUESTION:
{question}

OPTIONS:
A: {option_A}
B: {option_B}
C: {option_C}
D: {option_D}
E: {option_E}

IMPORTANT: Provide your answer in this exact format:
[LETTER]: [content of the chosen option]

For example: "A: Fever and chills"

Do not provide additional explanations.
```

### Open-Weight Models

Open-weight models use their **native chat templates**:

- **Meditron3** — Llama 3 chat format (`<|begin_of_text|><|start_header_id|>...`)
- **MedGemma** — Gemma chat format via `AutoProcessor.apply_chat_template()`
- **Qwen3 family** — Qwen chat format (`<|im_start|>system\n...<|im_end|>`) with `/no_think` to disable reasoning mode
- **Ministral** — Mistral chat format

The models are asked to reply in the format: `The final answer is [LETTER]: [option text]`

---

## Answer Parsing Logic

All scripts use a **multi-stage regex-based parser** (`_extract_letter()`) to extract the selected answer letter from the model response. The parser is designed to be robust against varying response formats.

### Closed-Source Parser (6 stages)

1. **`^([A-E])\s*[:\-]`** — Priority: response starts with `A:` or `B-` (our requested format)
2. **`ANSWER\s*:\s*([A-E])\b`** — Pattern: `ANSWER: C`
3. **`(?:The\s+)?final\s+answer\s+is\s+([A-E])\b`** — Pattern: `The final answer is B`
4. **Explicit keyword patterns** — Matches `answer/option/choice/letter X`, `I choose X`, `^X$` (standalone letter)
5. **Isolated letter scan** — First occurrence of any letter A–E in the response text
6. **Numeric fallback** — Maps digits 1–5 to letters A–E

If no letter can be extracted, the answer is marked as `"ND"` (Not Determined).

### Open-Weight Parser (7+ stages)

The open-weight parser extends the closed-source one with additional stages for handling reasoning-mode outputs:

0. **Thinking block removal** — Strips `<think>...</think>` (Qwen3) and `<unused94>thought...</thought>` (MedGemma/Meditron3) blocks, extracting only the text after the reasoning
1. **`ANSWER\s*:\s*([A-E])\b`** — Explicit answer tag
2. **First-character check** — If response starts directly with a letter followed by `.`, `:`, `)`, or space
3. **`final\s+answer\s+is\s+([A-E])`** — Natural language answer statement
4. **`^([A-E])\s*[:\)]`** — Letter with punctuation at line start
5. **Extended pattern battery** — Includes Italian patterns (`la risposta è`, `risposta corretta è`), markdown bold (`**X**`), `opzione X`, etc.
6. **First-line letter scan** — If the first line is ≤3 characters, scan for any A–E letter
7. **Content-matching fallback** — When all regex approaches fail, the parser compares the response text against each option's keywords (words with 4+ characters). An option is matched only if it has >70% keyword overlap AND is at least 0.3 ahead of the second-best option, preventing ambiguous matches.

### Option Shuffling & Consistency Tracking

At each run, the five answer options are randomly shuffled to eliminate positional bias. The mapping between shuffled and original positions is tracked:

- **Correct answer** → recorded as `1`
- **Wrong answer** → recorded as the **original** letter (before shuffling), enabling cross-run consistency analysis
- **Extraction failure** → recorded as `"ND"`

---

## Dependencies

### Closed-source models
```
openai          # GPT
anthropic       # Claude
requests        # DeepSeek, Grok, Mistral, Qwen
pandas
openpyxl
```

### Open-weight models
```
torch
transformers
accelerate
pandas
openpyxl
```

### Quantized models
```
llama-cpp-python
pandas
openpyxl
```

---

## Output Files

Each script produces the following output files:

| File | Description |
|---|---|
| `results_{model}.json` | Full results with per-run metrics |
| `consistency_{model}.xlsx` | Per-question answer tracking across all 250 runs |
| `metrics_{model}.xlsx` | Per-run accuracy, score, time, and consistency |
| `raw_responses_{model}.jsonl` | Raw model responses with full context (question, options, timing) |
| `checkpoint_{model}.json` | Resumption checkpoint (deleted on completion) |
| `{model}_medical_exam_test.log` | Execution log |
