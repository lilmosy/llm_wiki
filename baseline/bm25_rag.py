"""Baseline: Vanilla RAG (BM25) -- top-k flat passages + one LLM answer.
Same model, same corpus as LLM-Wiki (controlled comparison, paper Section 4.4).
Flat chunks: NO structure, NO links -- the ablation the paper calls 'w/o Wiki Structure'.
"""
import re

from rank_bm25 import BM25Okapi
from llm import complete

_WORD = re.compile(r"[a-z0-9]+")


class BM25Rag:
    def __init__(self, corpus, cfg):
        self.corpus = corpus
        self.cfg = cfg
        self.docs = [f"{x['title']}: {x['text']}" for x in corpus]
        self.bm = BM25Okapi([_WORD.findall(d.lower()) for d in self.docs])

    def answer(self, question):
        scores = self.bm.get_scores(_WORD.findall(question.lower()))
        order = sorted(range(len(self.docs)), key=lambda i: scores[i], reverse=True)
        topk = [self.docs[i] for i in order[:self.cfg["baseline_topk"]]]
        ctx = "\n\n".join(f"[{i+1}] {d}" for i, d in enumerate(topk))
        sys = ("Answer the question using ONLY the retrieved passages. Reply with exactly one line: "
               "'ANSWER: <x>' where <x> is the SHORTEST exact answer span only -- a name, date, or title -- "
               "with NO explanation, NO parentheses, NO extra clauses.")
        raw = complete(sys, f"Passages:\n{ctx}\n\nQuestion: {question}", self.cfg["model"], 500)
        m = re.search(r"ANSWER:\s*(.+)", raw)
        pred = (m.group(1) if m else raw).strip().split("\n")[0]
        return {"pred": pred, "raw": raw, "retrieved": [order[i] for i in range(min(self.cfg["baseline_topk"], len(order)))]}
