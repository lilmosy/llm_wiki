"""B3. RAPTOR -- QUERY TIME.

"Collapsed tree" retrieval: flatten every level (raw passages AND every summary
node) into one pool and take the top-k by cosine. A summary node can win, which
is the whole point -- facts scattered across several passages are already merged
inside one summary, so a single retrieval step can carry a multi-hop answer.
"""
import json
import os

from core.embed import encode_cached, encode_query, top_k
from baseline._common import answer_from_context, index_dir


class Raptor:
    needs_build = True
    arm = "raptor"

    def __init__(self, cfg):
        self.cfg = cfg
        self.nodes = None
        self.mat = None

    def load(self):
        d = index_dir(self.arm)
        self.nodes = json.load(open(os.path.join(d, "tree.json")))["nodes"]
        self.mat = encode_cached([n["text"] for n in self.nodes], self.cfg,
                                 os.path.join(d, "vectors.npz"))

    def answer(self, question):
        if self.nodes is None:
            self.load()
        qv = encode_query([question], self.cfg)[0]
        hits = top_k(qv, self.mat, self.cfg["baseline_topk"])
        blocks = [self.nodes[i]["text"] for i, _ in hits]
        out = answer_from_context(question, blocks, self.cfg)
        # A retrieved summary may stand for many raw passages. Preserve both the
        # node identity and its leaf-passage provenance for evidence evaluation.
        by_id = {n["id"]: n for n in self.nodes}

        def leaves(node_id):
            node = by_id[node_id]
            return [node_id] if node["level"] == 0 else sum(
                (leaves(child) for child in node["children"]), [])

        node_ids = [self.nodes[i]["id"] for i, _ in hits]
        pids = list(dict.fromkeys(sum((leaves(node_id) for node_id in node_ids), [])))
        out["retrieved"] = node_ids
        out["retrieved_pids"] = pids
        return out
