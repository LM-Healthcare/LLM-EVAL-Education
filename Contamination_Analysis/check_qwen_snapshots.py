"""
Check which Qwen3-Max versions are still served by Alibaba Model Studio (DashScope) with your key.
The alias `qwen3-max` used in the main study (February 2026) is no longer served; dated snapshots
may still be. Usage, from the repository root:
    python Contamination_Analysis/check_qwen_snapshots.py
API key: DASHSCOPE_API_KEY, or asked at start-up.
"""
import getpass
import os

import requests

ENDPOINTS = {
    "Singapore (used in the study)": "https://dashscope-intl.aliyuncs.com/compatible-mode/v1/chat/completions",
    "US (Virginia)": "https://dashscope-us.aliyuncs.com/compatible-mode/v1/chat/completions",
}
CANDIDATES = ["qwen3-max", "qwen3-max-2026-01-23", "qwen3-max-2025-09-23", "qwen3-max-preview", "qwen-max"]

key = os.environ.get("DASHSCOPE_API_KEY") or getpass.getpass("DashScope API key (DASHSCOPE_API_KEY): ")
for region, url in ENDPOINTS.items():
    print(f"== {region}")
    for m in CANDIDATES:
        try:
            r = requests.post(url, headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
                              json={"model": m, "temperature": 0.1, "max_tokens": 5,
                                    "messages": [{"role": "user", "content": "Rispondi solo: A"}]}, timeout=60)
            if r.status_code == 200:
                print(f"  OK    {m:<24} answered as model={r.json().get('model')}")
            else:
                print(f"  FAIL  {m:<24} HTTP {r.status_code}: {r.text[:120]}")
        except Exception as e:
            print(f"  FAIL  {m:<24} {e}")
