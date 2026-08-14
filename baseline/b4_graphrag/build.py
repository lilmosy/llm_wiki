"""B4. GraphRAG -- OFFLINE BUILD (Edge et al., 2024).

Three stages, all offline:
  1. per-passage LLM extraction of entities and relationships
  2. graph assembly, then COMMUNITY DETECTION over that graph
     (paper uses Leiden; this MVP substitutes networkx greedy modularity --
      the simplification is reported)
  3. per-community LLM summary ("community report")

WHY THIS MUST BE AN OFFLINE BUILD:
extraction + community reports are the expensive LLM stages (~20 calls for 16
passages). At query time GraphRAG only reads the pre-written community reports.
Rebuilding per question would multiply that by the question count.

Artifacts -> indexes/graphrag/{graph.json, COMMUNITIES.md}
"""
import json
import os

import networkx as nx

from core.llm import complete, complete_json
from baseline._common import index_dir

_EXTRACT_SYS = ("You are an information extraction system. Extract entities and the "
                "relationships between them from the passage. Return ONLY JSON.")

_EXTRACT_USER = """Passage (id = {pid}):
{text}

Return JSON:
{{"entities": [{{"name": "...", "type": "person|film|place|organization|other", "description": "one line"}}],
  "relations": [{{"source": "<entity name>", "target": "<entity name>", "description": "how they relate"}}]}}

Rules:
- Entity names must be the full surface form used in the passage.
- Include factual attributes (birth dates, death dates, years) inside the entity description.
- Only relations between entities you listed."""

_REPORT_SYS = ("You write a community report for a knowledge graph community. "
               "Preserve every concrete fact: full names, full dates, places, and "
               "who-did-what relations. Do not generalise or omit details.")


def build(corpus, cfg):
    d = index_dir("graphrag")
    G = nx.Graph()
    ent_src = {}      # entity -> set of pids
    extraction_failures = []

    for x in corpus:
        try:
            r = complete_json(_EXTRACT_SYS,
                              _EXTRACT_USER.format(pid=x["pid"], text=f"{x['title']}: {x['text']}"),
                              cfg["model"], 1200)
        except Exception as e:
            r = {"entities": [], "relations": []}
            # Do not silently turn an unavailable LLM response into apparent
            # negative evidence. The graph is still built so a long run can
            # finish, but the exact missing source is surfaced in its artifact.
            extraction_failures.append({"pid": x["pid"],
                                        "error": f"{type(e).__name__}: {e}"})
        for e in r.get("entities", []):
            name = (e.get("name") or "").strip()
            if not name:
                continue
            if name not in G:
                G.add_node(name, type=e.get("type", "other"), descriptions=[])
            desc = (e.get("description") or "").strip()
            if desc and desc not in G.nodes[name]["descriptions"]:
                G.nodes[name]["descriptions"].append(desc)
            ent_src.setdefault(name, set()).add(x["pid"])
        for rel in r.get("relations", []):
            s, t = (rel.get("source") or "").strip(), (rel.get("target") or "").strip()
            if s and t and s in G and t in G and s != t:
                G.add_edge(s, t, description=(rel.get("description") or "")[:200])

    # ---- community detection (Leiden -> greedy modularity substitution) ----
    if G.number_of_nodes():
        comms = [sorted(c) for c in nx.community.greedy_modularity_communities(G)]
    else:
        comms = []

    reports = []
    for ci, members in enumerate(comms):
        lines = []
        for m in members:
            lines.append(f"- {m} ({G.nodes[m].get('type')}): " + "; ".join(G.nodes[m]["descriptions"]))
        for a, b, data in G.edges(members, data=True):
            if a in members and b in members:
                lines.append(f"- RELATION {a} -> {b}: {data.get('description', '')}")
        body = "\n".join(lines)
        # Communities are wildly uneven (34 members down to 1), so a flat output
        # budget truncates exactly the big ones -- and the big ones are where
        # multi-entity questions live. C0 carried Ernest I with "died Dessau 12
        # June 1516" in its node data and the report was cut before reaching him;
        # at query time only the report is read, so the fact was unrecoverable.
        budget = max(700, min(4000, len(body) // 4))
        rep = complete(_REPORT_SYS,
                       f"Community members and relations:\n{body}\n\n"
                       "Write the community report as one dense paragraph.",
                       cfg["model"], budget).strip()
        if rep and rep[-1] not in ".!?\"')":
            print(f"    ! C{ci} report may be truncated ({len(members)} members, "
                  f"budget {budget}, ends {rep[-30:]!r})", flush=True)
        reports.append({"id": f"C{ci}", "members": members, "report": rep,
                        "sources": sorted({p for m in members for p in ent_src.get(m, [])})})

    json.dump({"nodes": [{"name": n, **{k: v for k, v in G.nodes[n].items()}} for n in G.nodes],
               "edges": [{"source": a, "target": b, "description": dta.get("description", "")}
                         for a, b, dta in G.edges(data=True)],
               "communities": reports, "extraction_failures": extraction_failures},
              open(os.path.join(d, "graph.json"), "w"), indent=1)

    L = ["# GraphRAG communities (B4)\n",
         f"- entities: {G.number_of_nodes()} | relations: {G.number_of_edges()}",
         f"- communities detected: {len(comms)} (greedy modularity; paper uses Leiden)",
         f"- LLM calls to build: **{len(corpus) + len(comms)}** "
         f"({len(corpus)} extraction + {len(comms)} community reports)\n"]
    for r in reports:
        L.append(f"\n## {r['id']}  ({len(r['members'])} entities, sources: {', '.join(r['sources'])})")
        L.append(f"**members:** {', '.join(r['members'])}\n")
        L.append(r["report"])
    open(os.path.join(d, "COMMUNITIES.md"), "w").write("\n".join(L))

    return {"entities": G.number_of_nodes(), "relations": G.number_of_edges(),
            "communities": len(comms), "extraction_failures": len(extraction_failures)}
