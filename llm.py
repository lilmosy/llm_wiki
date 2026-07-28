"""Thin Anthropic client wrapper shared by every phase.

Loads ANTHROPIC_API_KEY from the repo-root .env, exposes one text-completion
helper and one tool-use step. The SAME model is used for compile, agent, and
baseline so that measured differences reflect knowledge organization, not the
model (paper Section 4.4).
"""
import json
import os
import re
import time

import anthropic

# ---- load .env (repo root: two levels up from this file) -------------------
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_ENV = os.path.join(_ROOT, ".env")
if os.path.exists(_ENV):
    for line in open(_ENV):
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))

_CLIENT = anthropic.Anthropic()

# rough token accounting so the report can quote compile vs query cost
USAGE = {"calls": 0, "input_tokens": 0, "output_tokens": 0}


def _acct(resp):
    USAGE["calls"] += 1
    try:
        USAGE["input_tokens"] += resp.usage.input_tokens
        USAGE["output_tokens"] += resp.usage.output_tokens
    except Exception:
        pass


def complete(system, user, model, max_tokens=2000):
    """Single text completion with basic retry."""
    for attempt in range(4):
        try:
            resp = _CLIENT.messages.create(
                model=model,
                max_tokens=max_tokens,
                system=system,
                messages=[{"role": "user", "content": user}],
            )
            _acct(resp)
            return "".join(b.text for b in resp.content if b.type == "text")
        except (anthropic.RateLimitError, anthropic.APIStatusError, anthropic.APIConnectionError) as e:
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
