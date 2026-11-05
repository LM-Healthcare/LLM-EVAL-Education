"""
Proprietary models no longer reachable through their original API, queried through OpenRouter for
the contamination analyses (second revision).

  deepseek  DeepSeek V3.2. The alias `deepseek-chat` used in the main study was retired by DeepSeek
            on 24 July 2026. The same open-weight model (deepseek/deepseek-v3.2) is queried with
            reasoning disabled (non-thinking mode, as `deepseek-chat`), by default with OpenRouter's
            provider routing (--provider pins one provider).
  mistral   Mistral Large 3 (mistralai/mistral-large-2512), served by Mistral (provider pinned).
  qwen      Qwen3-Max (qwen/qwen3-max), served by Alibaba (provider pinned).

Prompt, generation parameters (temperature 0.1, max_tokens 50) and answer extraction are those of
the scripts in Official_testing_code/Closed/: the tester class of each model is reused unchanged and
only the endpoint, model identifier and API key differ.

Usage, from the repository root:
    python Contamination_analysis/run_openrouter.py --check                      # one test call per model
    python Contamination_analysis/run_openrouter.py --model mistral --task perturbation --runs 5
    python Contamination_analysis/run_openrouter.py --model mistral --task ts_guessing

API key: environment variable OPENROUTER_API_KEY, or asked at start-up.
Outputs (results/): perturbation_<model>_openrouter.jsonl, ts_guessing_<model>_openrouter.jsonl
(deepseek: *_deepseek_v32_openrouter.jsonl) and
openrouter_calls.jsonl, which records for every call the model and the provider that served it.
"""
import argparse
import getpass
import importlib.util
import json as jsonlib
import os
import sys
import time

import requests

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_perturbation as rp  # noqa: E402
import run_ts_guessing as ts   # noqa: E402

URL = "https://openrouter.ai/api/v1/chat/completions"
# name: (tester in run_perturbation.MODELS, OpenRouter model id, pinned provider, extra request fields)
TARGETS = {
    "deepseek": ("deepseek", "deepseek/deepseek-v3.2", None, {"reasoning": {"enabled": False}}),
    "mistral": ("mistral", "mistralai/mistral-large-2512", "Mistral", {}),
    "qwen": ("qwen", "qwen/qwen3-max", "Alibaba", {}),
}
STATE = {"target": None, "provider": None}
_post = requests.post


def post_openrouter(url, headers=None, json=None, **kw):
    """requests.post used by the testers: adds the OpenRouter options and logs model and provider."""
    if url == URL and json is not None:
        _, _, _, extra = TARGETS[STATE["target"]]
        json = {**json, **extra}
        if STATE["provider"]:
            json["provider"] = {"order": [STATE["provider"]], "allow_fallbacks": False}
    r = _post(url, headers=headers, json=json, **kw)
    if url == URL:
        rec = {"timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"), "target": STATE["target"], "status": r.status_code}
        try:
            body = r.json()
            rec.update(model=body.get("model"), provider=body.get("provider"))
        except ValueError:
            pass
        with open(os.path.abspath("openrouter_calls.jsonl"), "a", encoding="utf-8") as fh:
            fh.write(jsonlib.dumps(rec) + "\n")
    return r


def make_tester(target, key):
    base, model_id, _, _ = TARGETS[target]
    script, cls = rp.MODELS[base][0], rp.MODELS[base][1]
    path = os.path.join(rp.CODE, script)
    sys.path.insert(0, os.path.dirname(path))
    spec = importlib.util.spec_from_file_location(f"tester_{target}_openrouter", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    tester = getattr(mod, cls)(key)
    tester.api_url = URL
    tester.model_name = model_id
    tester.headers["Authorization"] = f"Bearer {key}"
    return tester


def get_key():
    return os.environ.get("OPENROUTER_API_KEY") or getpass.getpass("OpenRouter API key (OPENROUTER_API_KEY): ")


def check(key):
    """One short request per model; prints the model and provider that answered."""
    for target, (_, model_id, provider, extra) in TARGETS.items():
        body = {"model": model_id, "temperature": 0.1, "max_tokens": 20,
                "messages": [{"role": "user", "content": "Rispondi solo con la lettera: A, B, C, D o E? Scegli A."}],
                **extra}
        if provider:
            body["provider"] = {"order": [provider], "allow_fallbacks": False}
        try:
            r = _post(URL, headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
                      json=body, timeout=120)
            b = r.json()
            if r.status_code == 200:
                text = (b["choices"][0]["message"].get("content") or "").strip()
                print(f"OK   {target:<9} model={b.get('model')}  provider={b.get('provider')}  reply={text[:30]!r}")
            else:
                print(f"FAIL {target:<9} HTTP {r.status_code}: {jsonlib.dumps(b.get('error', b))[:300]}")
        except Exception as e:
            print(f"FAIL {target:<9} {e}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--model", choices=sorted(TARGETS))
    ap.add_argument("--task", choices=["perturbation", "ts_guessing"])
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--provider", default=None, help="pin an OpenRouter provider (default: see TARGETS)")
    a = ap.parse_args()
    if a.check:
        check(get_key())
        return
    if not (a.model and a.task):
        ap.error("--model and --task are required (or use --check)")

    STATE["target"] = a.model
    STATE["provider"] = a.provider or TARGETS[a.model][2]
    # same file name as the earlier run_deepseek_openrouter.py, so that a started run resumes
    name = "deepseek_v32_openrouter" if a.model == "deepseek" else f"{a.model}_openrouter"
    requests.post = post_openrouter                       # used by the testers and by run_ts_guessing
    base = TARGETS[a.model][0]
    rp.MODELS[name] = (rp.MODELS[base][0], rp.MODELS[base][1], "api", "OPENROUTER_API_KEY")

    def load(_name):
        t = make_tester(a.model, get_key())
        print(f"{a.model} via OpenRouter: {t.model_name}, provider "
              f"{STATE['provider'] or 'default routing'}, extra fields {TARGETS[a.model][3] or 'none'}")
        return t
    rp.load_tester = load
    ts.load_tester = load
    if a.task == "perturbation":
        sys.argv = [sys.argv[0], "--model", name, "--runs", str(a.runs)]
        rp.main()
    else:
        sys.argv = [sys.argv[0], "--model", name]
        ts.main()


if __name__ == "__main__":
    main()
