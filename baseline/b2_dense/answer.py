"""B2. Vanilla RAG (Dense) -- paper §4.2.

Identical to the BM25 arm except the retriever: cosine similarity over sentence
embeddings instead of lexical overlap. Still ONE-SHOT -- retrieve top-k once,
answer once, no second look.

OFFLINE BUILD: yes, but LIGHT -- encode the 16 passages once. No LLM involved,
so the build costs 0 LLM calls. Artifacts land in indexes/dense/.
"""
import json
import os

import numpy as np

from core.embed import encode_cached, encode_query, top_k
from baseline._common import answer_from_context, index_dir


class DenseRag:
    needs_build = True
    arm = "dense"

    def __init__(self, corpus, cfg):
        self.corpus = corpus
        self.cfg = cfg
        self.docs = [f"{x['title']}: {x['text']}" for x in corpus]
        self.mat = None

    # ---- offline build ----------------------------------------------------
    def build(self):
        d = index_dir(self.arm)
        # Same cache as the graph arms, so all four invalidate on the same rule
        # (model name + text hash). Dense used to write a bare .npy that carried
        # no record of which model produced it, which is how a 384-dimension
        # matrix outlived the move to a 1024-dimension embedder.
        self.mat = encode_cached(self.docs, self.cfg, os.path.join(d, "vectors.npz"))
        json.dump({"pids": [x["pid"] for x in self.corpus],
                   "model": self.cfg["embed_model"], "dim": int(self.mat.shape[1])},
                  open(os.path.join(d, "meta.json"), "w"), indent=1)
        with open(os.path.join(d, "INDEX.md"), "w") as f:
            f.write(f"# Dense index (B2)\n\n"
                    f"- embedding model: `{self.cfg['embed_model']}` "
                    f"(paper uses Qwen3-Embedding-8B — substituted)\n"
                    f"- vectors: {self.mat.shape[0]} passages x {self.mat.shape[1]} dims\n"
                    f"- LLM calls to build: **0** (embedding only)\n\n"
                    "One vector per raw passage. No summarisation, no structure — "
                    "this is the flat-chunk arm with a semantic retriever.\n")
        return {"nodes": int(self.mat.shape[0])}

    def load(self):
        d = index_dir(self.arm)
        self.mat = encode_cached(self.docs, self.cfg, os.path.join(d, "vectors.npz"))

    # ---- query ------------------------------------------------------------
    def answer(self, question):
        if self.mat is None:
            self.load()
        qv = encode_query([question], self.cfg)[0]
        hits = top_k(qv, self.mat, self.cfg["baseline_topk"])
        blocks = [self.docs[i] for i, _ in hits]
        out = answer_from_context(question, blocks, self.cfg)
        out["retrieved"] = [self.corpus[i]["pid"] for i, _ in hits]
        out["retrieved_pids"] = list(out["retrieved"])
        return out
