"""One shared LLM interface for the whole team (owned by D, used by A, B, D).

    from llm import generate
    generate("Summarise: ...")                     # plain prompt
    generate(prompt, system="You are ...")         # with a system message
    generate(prompt, max_new_tokens=400, temperature=0.0)

Backend is chosen by the LLM_BACKEND env var (default "hf"):
  hf      local open model via transformers (default model Qwen/Qwen2.5-1.5B-Instruct,
          override with LLM_MODEL). Free and reproducible.
  openai  any OpenAI-compatible API (needs OPENAI_API_KEY; optional OPENAI_BASE_URL,
          LLM_MODEL). Use only for quality comparisons.
  echo    returns the prompt back; lets you develop a notebook with no model at all.

Every call is cached in artifacts/llm_cache.jsonl keyed on (backend, model, system,
prompt, params), so re-running a notebook is fast and gives identical outputs.
"""
import hashlib
import json
import os
from pathlib import Path

CACHE_PATH = Path(__file__).resolve().parent / "artifacts" / "llm_cache.jsonl"
BACKEND = os.environ.get("LLM_BACKEND", "hf")
MODEL = os.environ.get("LLM_MODEL", {"hf": "Qwen/Qwen2.5-1.5B-Instruct", "openai": "gpt-4o-mini"}.get(BACKEND, BACKEND))

_cache = None
_pipe = None


def _load_cache():
    global _cache
    if _cache is None:
        _cache = {}
        if CACHE_PATH.exists():
            for line in CACHE_PATH.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    rec = json.loads(line)
                    _cache[rec["key"]] = rec["output"]
    return _cache


def _key(**fields):
    return hashlib.sha256(json.dumps(fields, sort_keys=True).encode()).hexdigest()


def _hf_generate(messages, max_new_tokens, temperature):
    global _pipe
    if _pipe is None:
        import torch
        from transformers import pipeline
        device = 0 if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else -1)
        _pipe = pipeline("text-generation", model=MODEL, device=device, torch_dtype="auto")
    kwargs = dict(max_new_tokens=max_new_tokens, return_full_text=False)
    kwargs.update(dict(do_sample=True, temperature=temperature) if temperature > 0 else dict(do_sample=False))
    out = _pipe(messages, **kwargs)
    return out[0]["generated_text"].strip()


def _openai_generate(messages, max_new_tokens, temperature):
    from openai import OpenAI
    client = OpenAI(base_url=os.environ.get("OPENAI_BASE_URL"))
    r = client.chat.completions.create(model=MODEL, messages=messages,
                                       max_tokens=max_new_tokens, temperature=temperature)
    return r.choices[0].message.content.strip()


def generate(prompt, system=None, max_new_tokens=300, temperature=0.0, use_cache=True):
    """Return the model's reply as a string. Deterministic by default (temperature 0)."""
    messages = ([{"role": "system", "content": system}] if system else []) + [
        {"role": "user", "content": prompt}]
    key = _key(backend=BACKEND, model=MODEL, messages=messages,
               max_new_tokens=max_new_tokens, temperature=temperature)
    cache = _load_cache()
    if use_cache and key in cache:
        return cache[key]
    if BACKEND == "echo":
        text = f"[echo:{MODEL}] " + prompt[:200]
    elif BACKEND == "openai":
        text = _openai_generate(messages, max_new_tokens, temperature)
    else:
        text = _hf_generate(messages, max_new_tokens, temperature)
    cache[key] = text
    CACHE_PATH.parent.mkdir(exist_ok=True)
    with CACHE_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"key": key, "backend": BACKEND, "model": MODEL,
                            "prompt": prompt, "system": system, "output": text}) + "\n")
    return text


def describe():
    """Model name and backend, for the report's reproducibility section."""
    return {"backend": BACKEND, "model": MODEL}
