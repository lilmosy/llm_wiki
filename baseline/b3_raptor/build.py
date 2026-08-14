"""B3. RAPTOR -- OFFLINE BUILD (Sarthi et al., 2024).

Recursive clustering + summarisation into a tree:
    level 0 (leaves) = the 16 raw passages
    level 1          = cluster them, LLM-summarise each cluster
    level 2          = cluster the level-1 summaries, summarise again
    ...until one node remains or max levels reached.

WHY THIS MUST BE AN OFFLINE BUILD:
every internal node is an LLM-written summary. Building at query time would
re-summarise the whole corpus for each of the 8 questions (8x the cost) and,
worse, give each question a DIFFERENT tree -- so score differences could no
longer be attributed to the method.

Artifacts -> indexes/raptor/{tree.json, TREE.md}
"""
import json
import math
import os

from sklearn.cluster import KMeans

from core.llm import complete
from core.embed import encode
from baseline._common import index_dir

_SUM_SYS = ("You are summarising a cluster of related documents for a retrieval index. "
            "Write a single dense paragraph that PRESERVES every concrete fact: names, "
            "full dates, places, and who-did-what relations. Do not generalise, do not "
            "drop details, do not add anything not present.")


def _cluster(vecs, target_size, seed=0):
    n = len(vecs)
    k = max(1, math.ceil(n / target_size))
    if k >= n:
        return list(range(n))
    km = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(vecs)
    return km.labels_.tolist()


def build(corpus, cfg):
    d = index_dir("raptor")
    nodes = []           # {id, level, text, children}
    for x in corpus:
        nodes.append({"id": x["pid"], "level": 0,
                      "text": f"{x['title']}: {x['text']}", "children": []})

    cur = list(nodes)
    for level in range(1, cfg["raptor_levels"] + 1):
        if len(cur) <= 1:
            break
        vecs = encode([n["text"] for n in cur], cfg)
        labels = _cluster(vecs, cfg["raptor_cluster_size"], seed=level)
        groups = {}
        for n, lab in zip(cur, labels):
            groups.setdefault(lab, []).append(n)

        new = []
        for lab, members in sorted(groups.items()):
            joined = "\n\n".join(f"- {m['text']}" for m in members)
            # Output budget scales with the cluster, because clusters do: a flat
            # 700 truncated 9 of 20 summaries mid-sentence, and what fell off the
            # end was whole passages -- L1-C10 stopped at "4 February 1", losing
            # every fact from the last three of its ten children. A summary that
            # drops its tail is not a lossy summary, it is a missing one.
            budget = max(700, min(4000, len(joined) // 4))
            summary = complete(_SUM_SYS, f"Documents:\n{joined}\n\nWrite the summary.",
                               cfg["model"], budget).strip()
            if summary and summary[-1] not in ".!?\"')":
                print(f"    ! L{level}-C{lab} summary may be truncated "
                      f"(budget {budget}, ends {summary[-30:]!r})", flush=True)
            nid = f"L{level}-C{lab}"
            node = {"id": nid, "level": level, "text": summary,
                    "children": [m["id"] for m in members]}
            new.append(node)
            nodes.append(node)
        cur = new

    json.dump({"nodes": nodes}, open(os.path.join(d, "tree.json"), "w"), indent=1)

    # human-readable artifact
    L = ["# RAPTOR tree (B3)\n",
         f"- levels built: {max(n['level'] for n in nodes)}",
         f"- total nodes: {len(nodes)} (leaves {sum(1 for n in nodes if n['level'] == 0)}, "
         f"summaries {sum(1 for n in nodes if n['level'] > 0)})",
         f"- LLM calls to build: **{sum(1 for n in nodes if n['level'] > 0)}** (one summary per cluster)\n"]
    for lv in sorted({n["level"] for n in nodes}, reverse=True):
        L.append(f"\n## Level {lv}" + (" (raw passages)" if lv == 0 else " (LLM summaries)"))
        for n in [x for x in nodes if x["level"] == lv]:
            L.append(f"\n### `{n['id']}`" + (f"  ← {', '.join(n['children'])}" if n["children"] else ""))
            L.append(f"{n['text'][:600]}")
    open(os.path.join(d, "TREE.md"), "w").write("\n".join(L))

    return {"nodes": len(nodes),
            "summaries": sum(1 for n in nodes if n["level"] > 0)}
