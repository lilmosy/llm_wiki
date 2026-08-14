"""Thin Anthropic client wrapper shared by every phase.

Loads ANTHROPIC_API_KEY from the repo-root .env, exposes one text-completion
helper and one tool-use step. The SAME model is used for compile, agent, and
baseline so that measured differences reflect knowledge organization, not the
model (paper Section 4.4).

RESPONSE CACHE
--------------
Every completion is keyed by sha256(model | max_tokens | system | user) and
stored under runs/cache/. An identical call never reaches the API again. This
matters because most re-runs change one stage but re-execute the whole
pipeline: fixing the link-annotation prompt used to mean paying for all 239
compile calls again, even though 80 source digests and every unchanged
SELECTPAGES call would have produced byte-identical output.

The key includes the full prompt text, so any prompt edit is automatically a
cache miss -- there is no way to silently reuse a stale answer for a changed
question. Cached calls are counted separately in USAGE["cached"] so reports can
still state the true API cost of a cold run.

Scope: every call that goes through complete()/complete_json() -- all index
building and all one-shot baseline answers, 950 of the 989 calls in the last
full run. The ReAct agent in retrieval/agent.py talks to the SDK directly
because it needs tool-use blocks; those 39 calls stay uncached rather than
introduce a serialisation layer for tool blocks that cannot be tested while the
API is unavailable.

Set LLMWIKI_NO_CACHE=1 to bypass, or delete runs/cache/ to force a cold run.
"""
import hashlib
import json
import os
import re
import time

import anthropic

# ---- load .env ------------------------------------------------------------
# Project root first, so a standalone clone works with its own .env. Then the
# workspace root, because several projects here share one key file.
# Already-set environment variables always win (setdefault).
_PROJECT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
for _env in (os.path.join(_PROJECT, ".env"),
             os.path.join(_PROJECT, "..", "..", ".env")):
    if not os.path.exists(_env):
        continue
    for line in open(_env):
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))

# The SDK default can wait for many minutes when an upstream connection stalls.
# A bounded timeout lets the retry loop below recover rather than freezing a
# long indexing run at one request.
_CLIENT = anthropic.Anthropic(timeout=60.0)

# rough token accounting so the report can quote compile vs query cost.
# "calls" counts real API calls only; cache hits are tallied separately so a
# warm re-run cannot make the reported build cost look cheaper than it was.
USAGE = {"calls": 0, "input_tokens": 0, "output_tokens": 0, "cached": 0}

# Anchor on the project root, not on this file's directory. When llm.py moved
# into core/ the old "next to me" form silently followed it to core/runs/cache
# and started an empty cache -- no error, just paid-for calls made twice.
_CACHE_DIR = os.path.join(_PROJECT, "runs", "cache")
_CACHE_ON = os.environ.get("LLMWIKI_NO_CACHE", "") not in ("1", "true", "True")


def _cache_path(model, max_tokens, system, user):
    h = hashlib.sha256("\x00".join(
        [model, str(max_tokens), system or "", user or ""]).encode()).hexdigest()
    return os.path.join(_CACHE_DIR, h[:2], h + ".json")


def _cache_get(p):
    if _CACHE_ON and os.path.exists(p):
        try:
            return json.load(open(p))["text"]
        except Exception:
            return None
    return None


def _cache_put(p, model, system, user, text):
    if not _CACHE_ON:
        return
    os.makedirs(os.path.dirname(p), exist_ok=True)
    # the prompt is stored alongside the answer so the cache stays auditable --
    # you can read any entry and see exactly which call produced it.
    json.dump({"model": model, "system": system, "user": user, "text": text},
              open(p, "w"), ensure_ascii=False, indent=1)


def _acct(resp):
    USAGE["calls"] += 1
    try:
        USAGE["input_tokens"] += resp.usage.input_tokens
        USAGE["output_tokens"] += resp.usage.output_tokens
    except Exception:
        pass


def complete(system, user, model, max_tokens=2000):
    """Single text completion with basic retry, served from cache when possible."""
    path = _cache_path(model, max_tokens, system, user)
    hit = _cache_get(path)
    if hit is not None:
        USAGE["cached"] += 1
        return hit
    for attempt in range(4):
        try:
            resp = _CLIENT.messages.create(
                model=model,
                max_tokens=max_tokens,
                system=system,
                messages=[{"role": "user", "content": user}],
            )
            _acct(resp)
            text = "".join(b.text for b in resp.content if b.type == "text")
            _cache_put(path, model, system, user, text)
            return text
        except (anthropic.RateLimitError, anthropic.APIStatusError,
                anthropic.APIConnectionError, anthropic.APITimeoutError) as e:
            if attempt == 3:
                raise
            time.sleep(2 ** attempt + 1)


def complete_json(system, user, model, max_tokens=2000):
    """Completion that must return JSON; robustly extract the first JSON value."""
    txt = complete(system + "\nReturn ONLY valid JSON, no prose, no code fences.",
                   user, model, max_tokens)
    return extract_json(txt)


def extract_json(txt):
    txt = txt.strip()
    txt = re.sub(r"^```(?:json)?", "", txt).strip()
    txt = re.sub(r"```$", "", txt).strip()
    # find first { or [ and matching close by json.loads on progressively trimmed text
    start = min([i for i in (txt.find("{"), txt.find("[")) if i != -1], default=-1)
    if start == -1:
        raise ValueError(f"no JSON found in: {txt[:200]}")
    for end in range(len(txt), start, -1):
        try:
            return json.loads(txt[start:end])
        except Exception:
            continue
    raise ValueError(f"could not parse JSON: {txt[:200]}")
