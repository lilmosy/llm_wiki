"""SELECTPAGES (Algorithm 1, step 1): pick which EXISTING wiki pages a new
passage should update -- narrowing the compile scope before anything is written.
One LLM call per passage; COMPILEWIKIPAGES (compile.py) then does the writing.
"""
from llm import complete_json


def select_pages(passage, wiki, model, k):
    if not wiki.pages:
        return {"update": [], "create": []}
    idx = "\n".join(wiki.index_lines())
    sys = "You perform SELECTPAGES for a wiki compiler: pick which EXISTING pages a new passage should update."
    user = (f"CURRENT WIKI INDEX:\n{idx}\n\nNEW PASSAGE ({passage['pid']}):\n"
            f"{passage['title']}: {passage['text']}\n\n"
            f"Return JSON {{\"update\": [<=%d existing page slugs relevant to this passage>]}}." % k)
    try:
        r = complete_json(sys, user, model, 500)
        return {"update": [s for s in r.get("update", []) if s in wiki.pages][:k], "create": []}
    except Exception:
        return {"update": [], "create": []}
