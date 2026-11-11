"""Gemini (REST, API key) and Claude (Claude Code subscription via headless CLI,
or Anthropic API key if set) clients. Both return the raw model text."""
import glob
import json
import os
import shutil
import subprocess
import tempfile
import threading
import time
from pathlib import Path

import requests

from schema import FIELDS

SYSTEM_PROMPT = """You are an expert in spintronics, magnetic tunnel junctions (MTJ), spin-orbit torque (SOT), \
magnetic thin films, 2D/van der Waals magnets and DFT. You extract structured data from research-paper excerpts.

Rules:
- Use ONLY information in the excerpts. Never guess. If a field is not stated, use null.
- A paper can report several distinct devices / material stacks / compounds. Return ONE record per distinct \
device or material system that has its own reported values (max 8; pick the main/best ones). If the paper is a \
review, return records only for concrete values it states, and set Source_Type="Review".
- Numeric fields must be plain numbers (no units, no strings) converted to the unit in the field name. \
If a range is given, use the best/headline value and put the range in Notes. "~300" -> 300.
- Stack materials: write compositions as in the paper (e.g. "Co20Fe60B20"). Full stack can go in Notes.
- Y/N fields: "Y", "N" or null.
- Extraction_Confidence: High (values explicitly stated in text/table), Medium (some inferred/converted), \
Low (sparse / ambiguous excerpts).
- Output strictly valid JSON, no markdown fences, no commentary."""


def build_prompt(parsed: dict, fields: dict | None = None) -> str:
    """fields: {name: meaning/unit}; defaults to the full schema."""
    field_lines = "\n".join(f'  "{k}": {v}' for k, v in (fields or FIELDS).items())
    return f"""{SYSTEM_PROMPT}

FIELDS (name: meaning / unit):
{field_lines}

Return JSON exactly in this shape:
{{"paper_summary": "<one sentence>", "records": [ {{<every field above as a key>}} , ... ]}}

PDF METADATA HINTS (may be noisy): {json.dumps(parsed["metadata"], ensure_ascii=False)}

PAPER EXCERPTS (pre-selected relevant parts; references removed):
<<<
{parsed["context"]}
>>>"""


def build_abstract_prompt(papers: list[dict], fields: dict) -> str:
    """Several papers in one call when only title + abstract are available.
    papers: [{"id", "title", "abstract"}]"""
    field_lines = "\n".join(f'  "{k}": {v}' for k, v in fields.items())
    items = "\n\n".join(f"[{p['id']}] TITLE: {p['title'] or '(unknown)'}\n"
                        f"ABSTRACT: {p['abstract'] or '(no abstract available)'}" for p in papers)
    return f"""{SYSTEM_PROMPT}

Only the TITLE and ABSTRACT of each paper are available (no full text). Fill a field only when the title or
abstract states it; otherwise null.

FIELDS (name: meaning / unit):
{field_lines}

Return JSON exactly in this shape, with one entry for EVERY paper id below (records may be empty):
{{"papers": [{{"id": "<id as given>", "records": [ {{<every field above as a key>}} , ... ]}} , ...]}}

PAPERS:
<<<
{items}
>>>"""


# ---------------------------------------------------------------- Gemini
class GeminiClient:
    name = "gemini"

    def __init__(self, model: str | None = None, min_interval: float | None = None,
                 api_key: str | None = None):
        self.key = api_key or os.environ.get("GEMINI_API_KEY")
        if not self.key:
            raise RuntimeError("GEMINI_API_KEY not set (put it in .env)")
        # Free tier has a small per-model DAILY quota (20 req/day on gemini-3.8-flash),
        # so we fall back through a chain of models as each one is exhausted.
        chain = model or os.environ.get("GEMINI_MODELS") or os.environ.get("GEMINI_MODEL", "gemini-3.8-flash")
        self.models = [m.strip() for m in chain.split(",") if m.strip()]
        self.exhausted: set[str] = set()       # out for the day: daily quota used up / model not offered (404)
        self.cooldown: dict[str, float] = {}   # model -> time it may be tried again (overloaded / per-minute limit)
        # free tier is rate limited per minute too -> space calls out
        self.min_interval = min_interval if min_interval is not None else float(os.environ.get("GEMINI_MIN_INTERVAL", 7))
        # give up on a paper if no model answered within this many seconds
        self.max_wait = float(os.environ.get("GEMINI_MAX_WAIT", 600))
        self.on_event = None                   # optional callback(str) for progress logs
        self._last = 0.0
        self._lock = threading.Lock()
        self._local = threading.local()

    @property
    def model(self) -> str:
        """Model that served the last call in this thread (recorded in the JSON)."""
        return getattr(self._local, "model", None) or self._current()

    def _current(self) -> str | None:
        return next((m for m in self.models if m not in self.exhausted), None)

    def _event(self, msg: str):
        print(f"[gemini] {msg}", flush=True)
        if self.on_event:
            self.on_event(msg)

    @staticmethod
    def _detail(r) -> tuple[str, float | None]:
        """One readable line for an error response, plus Google's suggested retry delay."""
        try:
            err = r.json()["error"]
            delay = None
            for d in err.get("details", []):
                if d.get("retryDelay"):
                    delay = float(str(d["retryDelay"]).rstrip("s") or 0)
            return f"{err.get('status', '')}: {err.get('message', '')}", delay
        except (ValueError, KeyError, TypeError, AttributeError):
            return " ".join(r.text.split())[:300], None

    def generate(self, prompt: str) -> str:
        body = {
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": 0.1, "responseMimeType": "application/json",
                                 "maxOutputTokens": 32768},
        }
        errors: dict[str, str] = {}           # last error per model, for the final message
        tried: set[str] = set()
        deadline = time.time() + self.max_wait

        def give_up() -> bool:
            # past the deadline, and no model is left that is ready and hasn't been tried for this paper
            now = time.time()
            return now >= deadline and not any(
                m not in tried and m not in self.exhausted and self.cooldown.get(m, 0) <= now for m in self.models)

        while True:
            now = time.time()
            live = [m for m in self.models if m not in self.exhausted]
            if not live:
                raise RuntimeError("Gemini: daily free-tier quota exhausted for all models ("
                                   + "; ".join(f"{m}: {e}" for m, e in errors.items()) + "); rerun tomorrow")
            ready = [m for m in live if self.cooldown.get(m, 0) <= now]
            if not ready:  # every model is busy: wait for the first one to cool down
                if now >= deadline:
                    break
                time.sleep(max(1.0, min(15.0, min(self.cooldown[m] for m in live) - now)))
                continue
            model = next((m for m in ready if m not in tried), ready[0])
            tried.add(model)
            with self._lock:  # space out request starts across worker threads
                wait = self.min_interval - (time.time() - self._last)
                if wait > 0:
                    time.sleep(wait)
                self._last = time.time()
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
            try:
                # a full paper reply normally takes 30-110 s; a model silent for 3 min is stuck
                r = requests.post(url, json=body, headers={"x-goog-api-key": self.key}, timeout=(15, 180))
            except requests.RequestException as e:
                errors[model] = f"network error: {type(e).__name__}"
                self.cooldown[model] = time.time() + 60
                self._event(f"{model} did not reply ({type(e).__name__}); trying the next model")
                if give_up():
                    break
                continue

            if r.status_code == 200:
                self._local.model = model
                data = r.json()
                try:
                    return "".join(p.get("text", "") for p in data["candidates"][0]["content"]["parts"])
                except (KeyError, IndexError):
                    raise RuntimeError(f"Gemini empty response from {model}: {json.dumps(data)[:300]}")

            detail, delay = self._detail(r)
            errors[model] = f"HTTP {r.status_code} {detail[:160]}"
            if r.status_code == 429 and "PerDay" in r.text:
                self.exhausted.add(model)
                self._event(f"daily quota used up for {model}; switching to {self._current()}")
            elif r.status_code == 404:
                self.exhausted.add(model)
                self._event(f"{model} is not available to this key (404); skipping it")
            elif r.status_code in (500, 502, 503, 504):
                # overloaded ("high demand"): don't hammer it, move on to the next model
                self.cooldown[model] = time.time() + 120
                self._event(f"{model} is overloaded (HTTP {r.status_code}); trying the next model")
            elif r.status_code == 429:
                self.cooldown[model] = time.time() + (delay or 30)
                self._event(f"{model} per-minute limit hit; retrying in {int(delay or 30)} s or using another model")
            else:  # 400/401/403...: retrying won't help
                raise RuntimeError(f"Gemini HTTP {r.status_code} {detail}")
            if give_up():
                break
        raise RuntimeError(f"Gemini: no model answered within {int(self.max_wait)} s ("
                           + "; ".join(f"{m}: {e}" for m, e in errors.items()) + ")")


# ---------------------------------------------------------------- Claude
def find_claude_cli() -> str | None:
    if os.environ.get("CLAUDE_CLI"):
        return os.environ["CLAUDE_CLI"]
    exe = shutil.which("claude")
    if exe:
        return exe
    hits = sorted(glob.glob(os.path.expanduser(
        "~/.vscode/extensions/anthropic.claude-code-*/resources/native-binary/claude*.exe")))
    return hits[-1] if hits else None


class ClaudeClient:
    """Uses the Anthropic API if ANTHROPIC_API_KEY is set, otherwise the
    Claude Code CLI in headless mode (`claude -p`), which bills to your
    Claude subscription."""
    name = "claude"

    def __init__(self, model: str | None = None, api_key: str | None = None, mode: str | None = None):
        """mode: "api" | "cli" | None (auto: api if a key is available)."""
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if mode == "api" and not self.api_key:
            raise RuntimeError("Anthropic API key required for Claude API mode")
        if self.api_key and mode != "cli":
            import anthropic
            self.client = anthropic.Anthropic(api_key=self.api_key)
            self.model = model or os.environ.get("CLAUDE_MODEL", "claude-sonnet-5-5")
            self.mode = "api"
        else:
            self.cli = find_claude_cli()
            if not self.cli:
                raise RuntimeError("Claude Code CLI not found; set CLAUDE_CLI or ANTHROPIC_API_KEY")
            self.model = model or os.environ.get("CLAUDE_MODEL", "sonnet")
            self.mode = "cli"
            self.workdir = tempfile.mkdtemp(prefix="claude_extract_")  # no CLAUDE.md / project context

    def generate(self, prompt: str) -> str:
        if self.mode == "api":
            msg = self.client.messages.create(
                model=self.model, max_tokens=16000, temperature=0.1,
                messages=[{"role": "user", "content": prompt}],
            )
            return "".join(b.text for b in msg.content if b.type == "text")

        env = {k: v for k, v in os.environ.items() if not k.startswith("CLAUDECODE")}
        cmd = [self.cli, "-p", "--output-format", "json", "--model", self.model, "--max-turns", "1",
               "--disallowedTools", "Bash,Edit,Write,Read,Glob,Grep,WebFetch,WebSearch,Task,Agent,NotebookEdit"]
        for attempt in range(3):
            p = subprocess.run(cmd, input=prompt, capture_output=True, text=True, encoding="utf-8",
                               cwd=self.workdir, env=env, timeout=900)
            if p.returncode == 0:
                try:
                    out = json.loads(p.stdout)
                    if out.get("is_error"):
                        raise RuntimeError(out.get("result"))
                    return out.get("result", "")
                except json.JSONDecodeError:
                    return p.stdout
            err = (p.stderr or p.stdout)[-800:]
            if "limit" in err.lower():
                time.sleep(60 * (attempt + 1))
                continue
            raise RuntimeError(f"claude CLI exit {p.returncode}: {err}")
        raise RuntimeError("claude CLI: retries exhausted")


def make_client(name: str, **kwargs):
    return {"gemini": GeminiClient, "claude": ClaudeClient}[name](**kwargs)
