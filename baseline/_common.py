"""Shared answer-generation step for every baseline arm.

CRITICAL for the controlled comparison (paper §4.4): all retrieval arms end in
the SAME final prompt with the SAME answer format. Only the retrieved context
differs. If each arm had its own answer prompt, score gaps would mix "how the
knowledge was organised" with "how the answer was asked for".
"""
import os
import re

from core.llm import complete

# Identical across arms. Terse-span enforcement removes the verbosity artifact
# that deflates token-overlap F1 for agentic arms.
ANSWER_SYS = (
    "Answer the question using ONLY the retrieved context below. "
    "Reply with exactly one line: 'ANSWER: <x>' where <x> is the SHORTEST exact "
    "answer span only -- a name, a date, or a title -- with NO explanation, "
    "NO parentheses, NO extra clauses."
)


def parse_answer(raw):
    m = re.search(r"ANSWER:\s*(.+)", raw or "")
    return (m.group(1) if m else (raw or "")).strip().split("\n")[0]


def answer_from_context(question, blocks, cfg, sys=ANSWER_SYS):
    """blocks: list[str] of retrieved text. Returns {"pred", "raw"}."""
    ctx = "\n\n".join(f"[{i + 1}] {b}" for i, b in enumerate(blocks))
    raw = complete(sys, f"Context:\n{ctx}\n\nQuestion: {question}", cfg["model"], 500)
    return {"pred": parse_answer(raw), "raw": raw}


# Index namespace: lets one checkout hold several datasets side by side
# (indexes/<arm>/ for the hand-authored set, indexes_2wiki/<arm>/ for the real
# benchmark slice) so a swap never clobbers the other run's artifacts.
_NS = ""


def set_index_namespace(ns):
    global _NS
    _NS = ns or ""


def index_dir(arm):
    """Where an arm's offline build artifacts live: indexes[_<ns>]/<arm>/."""
    base = "indexes" + (f"_{_NS}" if _NS else "")
    root = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        base, arm)
    os.makedirs(root, exist_ok=True)
    return root
