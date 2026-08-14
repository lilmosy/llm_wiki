"""B0. None (Closed-book) -- paper §4.2 lower bound.

No retrieval at all: the question goes straight to the LLM, which must answer
from parametric knowledge.

WHY THIS ARM MATTERS MORE THAN ITS SCORE:
It is a CONTAMINATION PROBE. Our corpus is built from real Wikipedia entities
(Herschell Gordon Lewis, Pasquale Festa Campanile, John V of Anhalt-Zerbst), so
the model may already know the answers. Any question closed-book gets right is
a question that does NOT measure retrieval. The closed-book score therefore
tells us how many of the 8 questions are valid retrieval experiments at all.

No offline index -- the corpus is never read.
"""
from core.llm import complete
from baseline._common import parse_answer

_SYS = (
    "Answer the question from your own knowledge. You have NO access to any "
    "documents. If you do not know, reply 'ANSWER: unknown'. "
    "Otherwise reply with exactly one line: 'ANSWER: <x>' where <x> is the "
    "SHORTEST exact answer span only -- a name, a date, or a title -- with NO "
    "explanation, NO parentheses, NO extra clauses."
)


class ClosedBook:
    needs_build = False

    def __init__(self, cfg):
        self.cfg = cfg

    def answer(self, question):
        raw = complete(_SYS, f"Question: {question}", self.cfg["model"], 300)
        return {"pred": parse_answer(raw), "raw": raw,
                "retrieved": [], "retrieved_pids": []}
