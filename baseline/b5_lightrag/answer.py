"""B5. LightRAG -- QUERY TIME (dual-level retrieval).

1. LLM splits the question into two keyword sets:
     low-level  = concrete entity names        ("The Gamecock", "Monster a Go-Go")
     high-level = abstract themes/relations    ("film director", "birth date")
2. low-level  -> match against the ENTITY surface, then expand ONE hop to
                 neighbours (this is where the multi-hop material arrives)
3. high-level -> match against the RELATION surface
4. merge both into one context -> answer

Contrast with LLM-Wiki: here the hop expansion is a fixed, code-driven single
step. LLM-Wiki instead lets the agent read a page and DECIDE the next hop, up to
T_max times. Same graph idea, different amount of agency -- that contrast is the
paper's core claim.
"""
import json
import os

from core.llm import complete, complete_json
from core.embed import encode_cached, encode_query, top_k
from baseline._common import answer_from_context, index_dir

_KW_SYS = "You split a question into retrieval keywords. Return ONLY JSON."
_KW_USER = """Question: {q}

Return JSON:
{{"low_level": ["concrete entity names mentioned or clearly implied"],
  "high_level": ["abstract topics / relation types the question is about"]}}"""


class LightRag:
    needs_build = True
    arm = "lightrag"

    def __init__(self, corpus, cfg):
        self.cfg = cfg
        self.by_pid = {x["pid"]: f"{x['title']}: {x['text']}" for x in corpus}
        self.ents = None

    def load(self):
        d = index_dir(self.arm)
        g = json.load(open(os.path.join(d, "graph.json")))
        self.ents, self.rels = g["entities"], g["relations"]
        self.ent_mat = encode_cached(
            [f"{e['name']}: {'; '.join(e['descriptions'])}" for e in self.ents],
            self.cfg, os.path.join(d, "ent_vectors.npz"))
        self.rel_mat = encode_cached(
            [f"{r['source']} {' '.join(r['keywords'])} {r['target']}: {r['description']}"
             for r in self.rels],
            self.cfg, os.path.join(d, "rel_vectors.npz")) if self.rels else None

    def answer(self, question):
        if self.ents is None:
            self.load()
        try:
            kw = complete_json(_KW_SYS, _KW_USER.format(q=question), self.cfg["model"], 400)
        except Exception:
            kw = {"low_level": [question], "high_level": [question]}
        low = " ".join(kw.get("low_level") or [question])
        high = " ".join(kw.get("high_level") or [question])

        k = self.cfg["baseline_topk"]
        # low-level: entity match + one-hop neighbour expansion.
        # The membership test reads `seeds`, which never changes, while the
        # results accumulate in `names`. Testing against `names` instead would
        # let a node added early in this pass match a later relation, chaining
        # outward for as many hops as the relation order happens to allow --
        # measured at up to 4 hops on q4, which is LightRAG's cost argument
        # (one hop, no community summaries) quietly undone.
        seeds = {e["name"] for e in
                 [self.ents[i] for i, _ in top_k(encode_query([low], self.cfg)[0], self.ent_mat, k)]}
        names = set(seeds)
        for r in self.rels:
            if r["source"] in seeds:
                names.add(r["target"])
            elif r["target"] in seeds:
                names.add(r["source"])
        blocks, pids = [], []
        for e in self.ents:
            if e["name"] in names:
                blocks.append(f"ENTITY {e['name']} ({e['type']}): {'; '.join(e['descriptions'])}")
                pids += e["sources"]
        # high-level: relation match
        if self.rel_mat is not None:
            for i, _ in top_k(encode_query([high], self.cfg)[0], self.rel_mat, k):
                r = self.rels[i]
                blocks.append(f"RELATION {r['source']} -> {r['target']} [{', '.join(r['keywords'])}]: {r['description']}")
                pids.append(r["pid"])
        # ground with the source passages the two surfaces pointed at
        for p in dict.fromkeys(pids):
            if p in self.by_pid:
                blocks.append(f"SOURCE {p}: {self.by_pid[p]}")

        out = answer_from_context(question, blocks[:24], self.cfg)
        out["retrieved"] = sorted(set(pids))
        out["retrieved_pids"] = list(out["retrieved"])
        out["keywords"] = kw
        return out
