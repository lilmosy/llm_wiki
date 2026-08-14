"""B6. HippoRAG 2 -- OFFLINE BUILD (Gutierrez et al., 2025).

Builds an open KG of (subject, relation, object) triples, then a bipartite-ish
graph where entity nodes are linked to each other by relations AND to the
passage nodes they occur in. Query time runs Personalized PageRank from the
question's entities; probability mass flows along relations and lands on
passages the question never named.

WHY THIS MUST BE AN OFFLINE BUILD:
triple extraction is the LLM stage (~16 calls). The PPR itself is pure
computation (no LLM), which is exactly HippoRAG's efficiency claim: pay once
offline, then multi-hop retrieval costs one matrix iteration.

Artifacts -> indexes/hipporag/{triples.json, GRAPH.md}
"""
import json
import os

from core.llm import complete_json
from baseline._common import index_dir

_SYS = "You are an open information extraction system. Return ONLY JSON."
_USER = """Passage (id = {pid}):
{text}

Extract every factual statement as a triple.

Return JSON:
{{"triples": [{{"subject": "...", "relation": "...", "object": "..."}}]}}

Rules:
- Use the full surface form of names as they appear in the passage.
- Dates and years are valid objects (e.g. {{"subject": "X", "relation": "born on", "object": "15 June 1926"}}).
- Extract relations between named things, not generic commentary."""


def build(corpus, cfg):
    d = index_dir("hipporag")
    triples = []
    extraction_failures = []
    for x in corpus:
        try:
            r = complete_json(_SYS, _USER.format(pid=x["pid"], text=f"{x['title']}: {x['text']}"),
                              cfg["model"], 1200)
        except Exception as e:
            r = {"triples": []}
            extraction_failures.append({"pid": x["pid"],
                                        "error": f"{type(e).__name__}: {e}"})
        for t in r.get("triples", []):
            s, rel, o = ((t.get("subject") or "").strip(), (t.get("relation") or "").strip(),
                         (t.get("object") or "").strip())
            if s and o:
                triples.append({"subject": s, "relation": rel, "object": o, "pid": x["pid"]})

    json.dump({"triples": triples, "extraction_failures": extraction_failures},
              open(os.path.join(d, "triples.json"), "w"), indent=1)

    ents = sorted({t["subject"] for t in triples} | {t["object"] for t in triples})
    L = ["# HippoRAG 2 open KG (B6)\n",
         f"- triples: {len(triples)} | distinct nodes: {len(ents)}",
         f"- LLM calls to build: **{len(corpus)}** (triple extraction)",
         "- query-time PPR uses NO LLM — that is HippoRAG's efficiency claim\n",
         "\n## Triples\n"]
    for t in triples:
        L.append(f"- `{t['pid']}`  **{t['subject']}** —{t['relation']}→ *{t['object']}*")
    open(os.path.join(d, "GRAPH.md"), "w").write("\n".join(L))

    return {"triples": len(triples), "nodes": len(ents),
            "extraction_failures": len(extraction_failures)}
