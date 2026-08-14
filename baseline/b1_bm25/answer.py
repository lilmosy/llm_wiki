"""Baseline: Vanilla RAG (BM25) -- top-k flat passages + one LLM answer.
Same model, same corpus as LLM-Wiki (controlled comparison, paper Section 4.4).
Flat chunks: NO structure, NO links -- the ablation the paper calls 'w/o Wiki Structure'.
"""
import re

from rank_bm25 import BM25Okapi
from core.llm import complete

# \w, not [a-z0-9]: the ASCII class split every accented name in the corpus
# ("Zulawski" -> u/awski, "Fernan Gomez" -> fern/n/g/mez), which handicapped the
# two lexical arms on 15 of 16 questions for a reason unrelated to how they
# organize knowledge. The embedding arms never had this problem, so leaving it
# in would have confounded "dense beats sparse" with "our tokenizer cannot read
# Polish".
_WORD = re.compile(r"\w+", re.UNICODE)


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
        pids = [self.corpus[i]["pid"] for i in order[:self.cfg["baseline_topk"]]]
        return {"pred": pred, "raw": raw, "retrieved": pids, "retrieved_pids": pids}
