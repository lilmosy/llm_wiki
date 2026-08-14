"""B6. HippoRAG 2 -- QUERY TIME (Personalized PageRank).

  1. link the question to graph nodes by embedding similarity  (no LLM)
  2. run PPR seeded on those nodes                             (no LLM)
  3. score passages by the mass their entities accumulated
  4. one LLM call to answer from the top-k passages

Step 2 is the trick: a passage that shares NO words with the question can still
score high because its entities sit one relation away from a seed. That is how a
single retrieval step covers a multi-hop chain -- the thing BM25 structurally
cannot do.
"""
import json
import os
from collections import defaultdict

import networkx as nx

from core.embed import encode_cached, encode_query, top_k
from baseline._common import answer_from_context, index_dir


class HippoRag:
    needs_build = True
    arm = "hipporag"

    def __init__(self, corpus, cfg):
        self.cfg = cfg
        self.by_pid = {x["pid"]: f"{x['title']}: {x['text']}" for x in corpus}
        self.G = None

    def load(self):
        d = index_dir(self.arm)
        triples = json.load(open(os.path.join(d, "triples.json")))["triples"]
        G = nx.Graph()
        self.ent_pids = defaultdict(set)
        for t in triples:
            s, o, p = t["subject"], t["object"], t["pid"]
            G.add_node(s, kind="entity")
            G.add_node(o, kind="entity")
            G.add_edge(s, o, relation=t["relation"])
            self.ent_pids[s].add(p)
            self.ent_pids[o].add(p)
        self.G = G
        self.nodes = [n for n in G.nodes if G.nodes[n].get("kind") == "entity"]
        self.node_mat = (encode_cached(self.nodes, self.cfg,
                                       os.path.join(d, "vectors.npz"))
                         if self.nodes else None)

    def answer(self, question):
        if self.G is None:
            self.load()
        if not self.nodes:
            return {"pred": "", "raw": "", "retrieved": [], "retrieved_pids": []}

        # 1. question -> seed nodes (embedding linking, no LLM)
        qv = encode_query([question], self.cfg)[0]
        seeds = {self.nodes[i]: max(s, 0.0) for i, s in top_k(qv, self.node_mat, 5)}
        if not any(seeds.values()):
            seeds = {n: 1.0 for n in list(seeds)[:1]} or None

        # 2. Personalized PageRank
        pr = nx.pagerank(self.G, alpha=0.85, personalization=seeds)

        # 3. passage scores = summed mass of the entities they contain
        pscore = defaultdict(float)
        for ent, mass in pr.items():
            for p in self.ent_pids.get(ent, ()):
                pscore[p] += mass
        ranked = sorted(pscore.items(), key=lambda kv: -kv[1])[:self.cfg["baseline_topk"]]

        blocks = [self.by_pid[p] for p, _ in ranked if p in self.by_pid]
        out = answer_from_context(question, blocks, self.cfg)
        out["retrieved"] = [p for p, _ in ranked]
        out["retrieved_pids"] = list(out["retrieved"])
        out["seeds"] = list(seeds)
        out["ppr_top"] = [[p, round(s, 4)] for p, s in ranked]
        return out
