"""B5. LightRAG -- OFFLINE BUILD (Guo et al., 2025).

Same entity/relation extraction as GraphRAG, but deliberately NO community
summarisation -- dropping that stage is LightRAG's whole cost argument. Instead
it embeds two separate retrieval surfaces:

  * entity surface   -> answers "which concrete thing is this about?" (low level)
  * relation surface -> answers "what kind of connection is being asked about?"
                        (high level)

WHY THIS MUST BE AN OFFLINE BUILD:
extraction is the LLM stage (~16 calls) and both embedding surfaces have to
exist before any query can be routed against them.

Artifacts -> indexes/lightrag/{graph.json, ENTITIES.md}
"""
import json
import os

from core.llm import complete_json
from baseline._common import index_dir

_SYS = ("You are an information extraction system for a knowledge graph. Extract "
        "entities and their relationships. Return ONLY JSON.")

_USER = """Passage (id = {pid}):
{text}

Return JSON:
{{"entities": [{{"name": "...", "type": "person|film|place|organization|other", "description": "one line with concrete attributes (dates, places)"}}],
  "relations": [{{"source": "...", "target": "...", "keywords": ["..."], "description": "the nature of the relation"}}]}}

Rules:
- Entity names must be the full surface form used in the passage.
- Put concrete attributes (birth date, death date, release year) in the description.
- "keywords" are 1-3 abstract terms describing the relation type (e.g. "direction", "parenthood")."""


def build(corpus, cfg):
    d = index_dir("lightrag")
    entities = {}     # name -> {name, type, descriptions[], sources[]}
    relations = []    # {source, target, keywords[], description, pid}
    extraction_failures = []

    for x in corpus:
        try:
            r = complete_json(_SYS, _USER.format(pid=x["pid"], text=f"{x['title']}: {x['text']}"),
                              cfg["model"], 1200)
        except Exception as e:
            r = {"entities": [], "relations": []}
            extraction_failures.append({"pid": x["pid"],
                                        "error": f"{type(e).__name__}: {e}"})
        for e in r.get("entities", []):
            name = (e.get("name") or "").strip()
            if not name:
                continue
            ent = entities.setdefault(name, {"name": name, "type": e.get("type", "other"),
                                             "descriptions": [], "sources": []})
            desc = (e.get("description") or "").strip()
            if desc and desc not in ent["descriptions"]:
                ent["descriptions"].append(desc)
            if x["pid"] not in ent["sources"]:
                ent["sources"].append(x["pid"])
        for rel in r.get("relations", []):
            s, t = (rel.get("source") or "").strip(), (rel.get("target") or "").strip()
            if s and t and s != t:
                relations.append({"source": s, "target": t,
                                  "keywords": rel.get("keywords", []),
                                  "description": (rel.get("description") or "")[:300],
                                  "pid": x["pid"]})

    json.dump({"entities": list(entities.values()), "relations": relations,
               "extraction_failures": extraction_failures},
              open(os.path.join(d, "graph.json"), "w"), indent=1)

    L = ["# LightRAG dual-level index (B5)\n",
         f"- entities: {len(entities)} | relations: {len(relations)}",
         f"- LLM calls to build: **{len(corpus)}** (extraction only — NO community summaries, "
         "which is LightRAG's cost saving vs GraphRAG)\n",
         "\n## Entity surface (low-level retrieval target)\n"]
    for e in sorted(entities.values(), key=lambda z: z["name"]):
        L.append(f"- **{e['name']}** ({e['type']}) — {'; '.join(e['descriptions'])[:200]}  "
                 f"`{', '.join(e['sources'])}`")
    L.append("\n## Relation surface (high-level retrieval target)\n")
    for r in relations:
        L.append(f"- {r['source']} → {r['target']}  *[{', '.join(r['keywords'])}]* — {r['description'][:120]}")
    open(os.path.join(d, "ENTITIES.md"), "w").write("\n".join(L))

    return {"entities": len(entities), "relations": len(relations),
            "extraction_failures": len(extraction_failures)}
