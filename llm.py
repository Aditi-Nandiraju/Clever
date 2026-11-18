"""LLM client with an on-disk cache.

Every reply is saved under cache/, keyed by provider, model, temperature,
prompt and a `tag` (config/seed/problem/attempt). Re-running an experiment
therefore costs nothing and gives identical results, and each attempt in the
resampling baseline is a distinct sample.
"""
import hashlib
import json
import os
import time

import config

_hf = None


def _seed_from(tag):
    return int(hashlib.sha256(tag.encode()).hexdigest()[:8], 16) % (2**31)


def _gemini(system, user, tag):
    from google import genai
    from google.genai import types
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    r = client.models.generate_content(
        model=config.MODEL, contents=user,
        config=types.GenerateContentConfig(system_instruction=system,
                                           temperature=config.TEMPERATURE,
                                           max_output_tokens=config.MAX_OUTPUT_TOKENS,
                                           seed=_seed_from(tag)))
    usage = getattr(r, "usage_metadata", None)
    return r.text or "", {"in": getattr(usage, "prompt_token_count", None),
                          "out": getattr(usage, "candidates_token_count", None)}


def _openai_compat(system, user, tag):
    """Groq, OpenRouter, OpenAI, Ollama, vLLM ... any /chat/completions endpoint."""
    import requests
    base = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
    r = requests.post(f"{base}/chat/completions", timeout=600,
                      headers={"Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY', '')}"},
                      json={"model": config.MODEL, "temperature": config.TEMPERATURE,
                            "max_tokens": config.MAX_OUTPUT_TOKENS, "seed": _seed_from(tag),
                            "messages": [{"role": "system", "content": system},
                                         {"role": "user", "content": user}]})
    r.raise_for_status()
    j = r.json()
    u = j.get("usage", {})
    return j["choices"][0]["message"]["content"] or "", {"in": u.get("prompt_tokens"),
                                                         "out": u.get("completion_tokens")}


def _hf_local(system, user, tag):
    """Open model on a GPU (e.g. Qwen/Qwen2.5-Coder-7B-Instruct on Colab)."""
    global _hf
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    if _hf is None:
        tok = AutoTokenizer.from_pretrained(config.MODEL)
        model = AutoModelForCausalLM.from_pretrained(config.MODEL, torch_dtype="auto", device_map="auto")
        _hf = (tok, model)
    tok, model = _hf
    msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    ids = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt").to(model.device)
    torch.manual_seed(_seed_from(tag))
    out = model.generate(ids, max_new_tokens=config.MAX_OUTPUT_TOKENS, do_sample=True,
                         temperature=config.TEMPERATURE)
    text = tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True)
    return text, {"in": int(ids.shape[1]), "out": int(out.shape[1] - ids.shape[1])}


MOCK_FN = None   # set by the smoke test: fn(system, user, tag) -> str


def _mock(system, user, tag):
    return MOCK_FN(system, user, tag), {"in": len(user) // 4, "out": 50}


PROVIDERS = {"gemini": _gemini, "openai_compat": _openai_compat,
             "hf_local": _hf_local, "mock": _mock}


def generate(system, user, tag):
    """Return (text, usage, cached). Retries with backoff on rate limits/errors."""
    key = hashlib.sha256(json.dumps([config.PROVIDER, config.MODEL, config.TEMPERATURE,
                                     system, user, tag]).encode()).hexdigest()
    path = config.CACHE_DIR / f"{key}.json"
    if path.exists() and config.PROVIDER != "mock":
        d = json.loads(path.read_text(encoding="utf-8"))
        return d["text"], d["usage"], True
    fn = PROVIDERS[config.PROVIDER]
    err = None
    for wait in (0, 15, 45, 90, 180):
        time.sleep(wait)
        try:
            text, usage = fn(system, user, tag)
            if config.PROVIDER != "mock":
                path.write_text(json.dumps({"tag": tag, "text": text, "usage": usage}), encoding="utf-8")
            return text, usage, False
        except Exception as e:          # rate limit, network, ...
            err = e
            print(f"  [llm] {type(e).__name__}: {str(e)[:120]} -> retry in {wait or 15}s")
    raise err
