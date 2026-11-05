"""
DeepSeek V3.2 through OpenRouter, for the contamination analyses (second revision).

In the main study DeepSeek V3.2 was queried through the provider's alias `deepseek-chat`
(non-thinking mode, February 2026). That alias was retired by DeepSeek on 24 July 2026 and the
provider's API no longer serves V3.2. Its open weights are served by third-party providers; this
script queries `deepseek/deepseek-v3.2` through OpenRouter with reasoning disabled (non-thinking
mode, as `deepseek-chat`). Prompt, generation parameters (temperature 0.1, max_tokens 50) and answer
extraction are those of Official_testing_code/Closed/test_deepseek_medical_exam_v2.py: the tester
class is reused unchanged and only the endpoint, model identifier and API key differ.

Usage, from the repository root:
    python Contamination_analysis/run_deepseek_openrouter.py --task perturbation --runs 5
    python Contamination_analysis/run_deepseek_openrouter.py --task ts_guessing
Optional: --provider <name> pins one OpenRouter provider (no fallback to others).

API key: environment variable OPENROUTER_API_KEY, or asked at start-up.
Outputs (results/): perturbation_deepseek_v32_openrouter.jsonl, ts_guessing_deepseek_v32_openrouter.jsonl
and openrouter_calls.jsonl, which records for every call the model and the provider that served it.
"""
import argparse
import getpass
import importlib.util
import json
import os
import sys
import time

import requests

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_perturbation as rp  # noqa: E402
import run_ts_guessing as ts   # noqa: E402

NAME = "deepseek_v32_openrouter"
URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL_ID = "deepseek/deepseek-v3.2"
PROVIDER = None
_post = requests.post


def post_openrouter(url, headers=None, json=None, **kw):
    """requests.post used by the DeepSeek tester: adds the OpenRouter options and logs the provider."""
    if url == URL and json is not None:
        json = dict(json)
        json["reasoning"] = {"enabled": False}                 # non-thinking mode, as deepseek-chat
        if PROVIDER:
            json["provider"] = {"order": [PROVIDER], "allow_fallbacks": False}
    r = _post(url, headers=headers, json=json, **kw)
    if url == URL:
        try:
            body = r.json()
            rec = {"timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"), "status": r.status_code,
                   "model": body.get("model"), "provider": body.get("provider")}
        except ValueError:
            rec = {"timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"), "status": r.status_code}
        with open(os.path.abspath("openrouter_calls.jsonl"), "a", encoding="utf-8") as fh:
            fh.write(__import__("json").dumps(rec) + "\n")
    return r


def load_tester(name):
    script = os.path.join(rp.CODE, rp.MODELS["deepseek"][0])
    sys.path.insert(0, os.path.dirname(script))
    spec = importlib.util.spec_from_file_location("tester_deepseek_openrouter", script)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    key = os.environ.get("OPENROUTER_API_KEY") or getpass.getpass("OpenRouter API key (OPENROUTER_API_KEY): ")
    tester = mod.DeepseekMedicalExamTester(key)
    tester.api_url = URL
    tester.model_name = MODEL_ID
    tester.headers["Authorization"] = f"Bearer {key}"
    print(f"DeepSeek V3.2 via OpenRouter: {MODEL_ID}, reasoning disabled"
          + (f", provider {PROVIDER}" if PROVIDER else ", default provider routing"))
    return tester


def main():
    global PROVIDER
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=["perturbation", "ts_guessing"])
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--provider", default=None)
    a = ap.parse_args()
    PROVIDER = a.provider

    requests.post = post_openrouter                          # used by the tester and by run_ts_guessing
    rp.MODELS[NAME] = (rp.MODELS["deepseek"][0], "DeepseekMedicalExamTester", "api", "OPENROUTER_API_KEY")
    rp.load_tester = load_tester
    ts.load_tester = load_tester
    if a.task == "perturbation":
        sys.argv = [sys.argv[0], "--model", NAME, "--runs", str(a.runs)]
        rp.main()
    else:
        sys.argv = [sys.argv[0], "--model", NAME]
        ts.main()


if __name__ == "__main__":
    main()
