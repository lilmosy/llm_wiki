"""Query-time compositional traversal (Section 3.2): a ReAct tool-calling loop
over wiki_search / wiki_read, with sufficiency check and paper stop conditions
(T_max tool calls, patience P consecutive empty searches, >=1 read before answer).
"""
import json
import re

import anthropic
from llm import _CLIENT, _acct

_TOOLS = [
    {"name": "wiki_search",
     "description": "Search the Wiki index by page names, aliases, tags, and summaries. Returns candidate pages.",
     "input_schema": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}},
    {"name": "wiki_read",
     "description": "Batch-read wiki pages (by slug) or indices. Returned page content includes [[links]] to related pages for the next hop.",
     "input_schema": {"type": "object", "properties": {"paths": {"type": "array", "items": {"type": "string"}}}, "required": ["paths"]}},
]

_SYS = ("You answer multi-hop questions by traversing a compiled Wiki.\n"
        "Loop: wiki_search -> wiki_read -> follow [[links]] in the pages you read -> "
        "check if the evidence is sufficient -> answer.\n"
        "Rules: you MUST call wiki_read at least once before answering; follow inter-page "
        "[[links]] to reach bridge entities (e.g. a person's father, a film's director); "
        "when evidence is sufficient, reply with exactly one final line: 'ANSWER: <x>' "
        "where <x> is the SHORTEST exact answer span only -- a name, a date, or a title -- "
        "with NO explanation, NO parentheses, NO extra clauses.")


def answer_question(question, wiki, cfg):
    model, tmax, patience = cfg["model"], cfg["agent_tmax"], cfg["agent_patience"]
    messages = [{"role": "user", "content": f"Question: {question}"}]
    trace, empty_streak, reads = [], 0, 0
    final = ""
    for step in range(tmax):
        resp = _CLIENT.messages.create(model=model, max_tokens=cfg["max_tokens"],
                                       system=_SYS, tools=_TOOLS, messages=messages)
        _acct(resp)
        if resp.stop_reason != "tool_use":
            final = "".join(b.text for b in resp.content if b.type == "text")
            break
        messages.append({"role": "assistant", "content": resp.content})
        results = []
        for b in resp.content:
            if b.type != "tool_use":
                continue
            if b.name == "wiki_search":
                hits = wiki.search(b.input.get("query", ""), cfg["select_k"])
                empty_streak = empty_streak + 1 if not hits else 0
                trace.append({"tool": "wiki_search", "arg": b.input.get("query", ""),
                              "hits": [h["slug"] for h in hits]})
                out = json.dumps(hits)
            else:  # wiki_read
                paths = b.input.get("paths", [])
                reads += 1
                out = wiki.read(paths)
                trace.append({"tool": "wiki_read", "arg": paths})
            results.append({"type": "tool_result", "tool_use_id": b.id, "content": out})
        messages.append({"role": "user", "content": results})
        if empty_streak >= patience:
            trace.append({"stop": f"patience {patience} empty searches"})
            # force a final answer
            resp = _CLIENT.messages.create(model=model, max_tokens=cfg["max_tokens"],
                                           system=_SYS + "\nEvidence gathering ended; answer now.",
                                           messages=messages)
            _acct(resp)
            final = "".join(b.text for b in resp.content if b.type == "text")
            break
    m = re.search(r"ANSWER:\s*(.+)", final)
    pred = (m.group(1) if m else final).strip().split("\n")[0]
    return {"pred": pred, "raw": final, "trace": trace, "tool_calls": len(trace), "reads": reads}
