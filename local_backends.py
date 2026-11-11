"""Pluggable LOCAL extractor, used when the API (Gemini / Claude) fails or is not wanted.

Pick the backend in .env (no code changes needed):

  LOCAL_BACKEND=rules                         keyword rules (default, no model)
  LOCAL_BACKEND=ollama                        any model served by Ollama
      LOCAL_MODEL=qwen3:14b                   (LOCAL_URL defaults to http://localhost:11434)
  LOCAL_BACKEND=openai                        any OpenAI-compatible server: LM Studio, vLLM,
      LOCAL_MODEL=<model name>                llama.cpp server, a lab GPU server, Colab + tunnel
      LOCAL_URL=http://host:port              (LOCAL_API_KEY optional)
  LOCAL_BACKEND=python:my_module:my_function  your own code, e.g. a trained MatSciBERT model:
                                              my_function(title, text, fields) -> list[dict] or dict

Optional: LOCAL_MAX_CHARS (text sent to a model, default 12000), LOCAL_NUM_CTX (Ollama context,
default 8192), LOCAL_TIMEOUT (seconds, default 600).

Whatever the backend, its answers pass through the same safety net:
  * material / layer values that do not appear in the paper text are dropped (no invented values)
  * Device_Type is mapped onto the fixed list
  * if the backend errors or returns nothing usable, the keyword rules answer instead
"""
import importlib
import os
import re
import threading

import requests

import local_extractor
from schema import normalize_device_type

# text-valued features whose value must literally occur in the paper
GROUNDED = {"Main_Material", "Free_Layer_Material", "Tunnel_Barrier_Material", "Reference_Layer_Material",
            "Heavy_Metal_SOT_Layer", "Substrate", "Seed_Layer", "Capping_Layer", "SAF_Structure",
            "Interface_Insertion_Layer"}


def _squash(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", (s or "").lower())


def ground(rec: dict, title: str, text: str) -> dict:
    """Drop material values that are not written in the paper; keep the parts that are."""
    hay = _squash(local_extractor._compact(f"{title} {text}"))
    for k in GROUNDED & set(rec):
        v = rec.get(k)
        if not isinstance(v, str) or not v.strip():
            continue
        parts = [p.strip() for p in re.split(r";|,(?![^()]*\))", v) if p.strip()]
        kept = []
        for p in parts:
            core = _squash(re.sub(r"\(.*?\)", "", p)) or _squash(p)       # ignore "(001)", "(2 nm)"
            if core and core in hay:
                kept.append(p)
        rec[k] = "; ".join(kept) or None
    if "Device_Type" in rec:
        rec["Device_Type"] = normalize_device_type(rec["Device_Type"])
    return rec


class RulesBackend:
    name, label = "rules", "Local (rules)"

    def extract(self, title: str, text: str, fields: dict) -> list[dict]:
        vals = local_extractor.extract(title, text)
        return [{k: vals.get(k) for k in fields}]


class _ModelBackend:
    """Shared logic for model backends: prompt -> JSON -> safety net -> rules if unusable."""

    def __init__(self):
        self.max_chars = int(os.environ.get("LOCAL_MAX_CHARS", 12000))
        self.timeout = float(os.environ.get("LOCAL_TIMEOUT", 600))
        self.rules = RulesBackend()

    def _generate(self, prompt: str) -> str:
        raise NotImplementedError

    def raw_records(self, title: str, text: str, fields: dict) -> list[dict]:
        from llm_clients import build_prompt
        from run_extraction import parse_json
        prompt = build_prompt({"metadata": {"pdf_title": title}, "context": (text or "")[: self.max_chars]}, fields)
        data = parse_json(self._generate(prompt))
        recs = data.get("records") if isinstance(data, dict) else data
        return [r for r in (recs or []) if isinstance(r, dict)]

    def extract(self, title: str, text: str, fields: dict) -> list[dict]:
        try:
            recs = self.raw_records(title, text, fields)
        except Exception as e:
            print(f"[local:{self.name}] failed ({type(e).__name__}: {e}); using the rules", flush=True)
            recs = []
        recs = [ground({k: r.get(k) for k in fields}, title, text) for r in recs]
        recs = [r for r in recs if any(v not in (None, "") for k, v in r.items() if k not in ("Title", "Year"))]
        return recs or self.rules.extract(title, text, fields)


class OllamaBackend(_ModelBackend):
    def __init__(self, model: str, url: str):
        super().__init__()
        self.model, self.url = model, url.rstrip("/")
        self.name, self.label = "ollama", f"Local (ollama: {model})"

    def _generate(self, prompt: str) -> str:
        r = requests.post(f"{self.url}/api/generate", timeout=self.timeout, json={
            "model": self.model, "prompt": prompt, "stream": False, "format": "json", "think": False,
            "options": {"temperature": 0, "num_ctx": int(os.environ.get("LOCAL_NUM_CTX", 8192))}})
        r.raise_for_status()
        return r.json().get("response", "")


class OpenAICompatibleBackend(_ModelBackend):
    def __init__(self, model: str, url: str):
        super().__init__()
        self.model, self.url = model, url.rstrip("/")
        self.name, self.label = "openai", f"Local ({model})"

    def _generate(self, prompt: str) -> str:
        headers = {"Authorization": f"Bearer {os.environ['LOCAL_API_KEY']}"} if os.environ.get("LOCAL_API_KEY") else {}
        base = self.url if self.url.endswith("/v1") else self.url + "/v1"
        r = requests.post(f"{base}/chat/completions", headers=headers, timeout=self.timeout, json={
            "model": self.model, "temperature": 0, "response_format": {"type": "json_object"},
            "messages": [{"role": "user", "content": prompt}]})
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]


class PythonBackend(_ModelBackend):
    """LOCAL_BACKEND=python:my_module:my_function  ->  my_function(title, text, fields)"""

    def __init__(self, target: str):
        super().__init__()
        mod, _, fn = target.partition(":")
        self.fn = getattr(importlib.import_module(mod), fn or "extract")
        self.name, self.label = "python", f"Local ({mod}.{fn or 'extract'})"

    def raw_records(self, title, text, fields):
        out = self.fn(title, text, fields)
        return [out] if isinstance(out, dict) else list(out or [])


_backend, _lock = None, threading.Lock()


def get() -> RulesBackend | _ModelBackend:
    """The configured local backend (created once)."""
    global _backend
    with _lock:
        if _backend is None:
            kind = os.environ.get("LOCAL_BACKEND", "rules").strip()
            model = os.environ.get("LOCAL_MODEL", "").strip()
            try:
                if kind == "ollama":
                    _backend = OllamaBackend(model or "qwen3:4b", os.environ.get("LOCAL_URL", "http://localhost:11434"))
                elif kind == "openai":
                    _backend = OpenAICompatibleBackend(model, os.environ.get("LOCAL_URL", "http://localhost:1234"))
                elif kind.startswith("python:"):
                    _backend = PythonBackend(kind[len("python:"):])
                else:
                    _backend = RulesBackend()
            except Exception as e:
                print(f"[local] could not set up LOCAL_BACKEND={kind} ({e}); using the rules", flush=True)
                _backend = RulesBackend()
        return _backend


def is_local(label) -> bool:
    return isinstance(label, str) and label.startswith("Local")
