"""Query-time compositional traversal (Section 3.2): a ReAct tool-calling loop
over wiki_search / wiki_read, with sufficiency check and paper stop conditions
(T_max tool calls, patience P consecutive empty searches, >=1 read before answer).
"""
import json
import re

import anthropic
from core.llm import _CLIENT, _acct
from baseline.llm_wiki.tools import TOOLS, wiki_search, wiki_read

_SYS = ("You answer multi-hop questions by traversing a compiled Wiki.\n"
        "Loop: wiki_search -> wiki_read -> follow [[links]] in the pages you read -> "
        "check if the evidence is sufficient -> answer.\n"
        "Each page's Related Pages section annotates every [[link]] with HOW that page "
        "relates to the current one ('father of John V', 'director of this film') -- use "
        "those annotations to choose the next hop.\n"
        "The Wiki also has directory indices you can open with wiki_read: 'index.md' is the "
        "directory catalog, and '<category>/_index.md' lists every page in one category. "
        "For open-ended or enumeration questions, browse those indices first; if wiki_search "
        "returns hits unrelated to the question, read an index rather than guessing.\n"
        "Rules: you MUST call wiki_read at least once before answering; follow inter-page "
        "[[links]] to reach bridge entities (e.g. a person's father, a film's director); "
        "ground the answer in what you actually read -- never state a fact (a date, a name, "
        "a number) that did not appear in a page you opened; if the value you need is absent, "
        "keep traversing to find it rather than filling it in; "
        "when evidence is sufficient, reply with exactly one final line: 'ANSWER: <x>' "
        "where <x> is the SHORTEST exact answer span only -- a name, a date, or a title -- "
        "with NO explanation, NO parentheses, NO extra clauses.")


def answer_question(question, wiki, cfg):
    model, tmax, patience = cfg["model"], cfg["agent_tmax"], cfg["agent_patience"]
    messages = [{"role": "user", "content": f"Question: {question}"}]
    trace, empty_streak, reads = [], 0, 0
    read_pages, retrieved_pids = [], []
    final = ""
    for step in range(tmax):
        resp = _CLIENT.messages.create(model=model, max_tokens=cfg["max_tokens"],
                                       system=_SYS, tools=TOOLS, messages=messages)
        _acct(resp)
        if resp.stop_reason != "tool_use":
            text = "".join(b.text for b in resp.content if b.type == "text")
            # The model often narrates its next hop ("Rebane's birth date isn't
            # listed, let me check Lewis too") without emitting the tool call in
            # the same turn. Taking that as the final answer truncates traversal
            # mid-hop and leaves the model to fill the gap from memory. Only stop
            # once it has actually produced the ANSWER line.
            if text.strip() and "ANSWER:" not in text and step < tmax - 1:
                trace.append({"nudge": text.strip()[:160]})
                messages.append({"role": "assistant", "content": resp.content})
                messages.append({"role": "user", "content":
                                 "You have not answered yet. Either call a tool to fetch the "
                                 "evidence you still need, or output the final 'ANSWER: <x>' line."})
                continue
            final = text
            break
        messages.append({"role": "assistant", "content": resp.content})
        results = []
        for b in resp.content:
            if b.type != "tool_use":
                continue
            if b.name == "wiki_search":
                hits = wiki_search(wiki, b.input.get("query", ""), cfg["select_k"])
                empty_streak = empty_streak + 1 if not hits else 0
                trace.append({"tool": "wiki_search", "arg": b.input.get("query", ""),
                              "hits": [h["slug"] for h in hits]})
                out = json.dumps(hits)
            else:  # wiki_read
                paths = b.input.get("paths", [])
                reads += 1
                out = wiki_read(wiki, paths)
                # A Wiki page is a compiled view, not the raw evidence itself.
                # Keep its source pids so 2Wiki gold evidence recall can distinguish
                # "page was read but wrong source" from "page was never reached".
                pages, pids = [], []
                for path in paths:
                    slug = path.split("/")[-1].replace(".md", "")
                    if slug in wiki.pages:
                        pages.append(slug)
                        pids.extend(wiki.pages[slug].get("sources", []))
                    elif ("digests" in path or "articles" in path) and slug in wiki.sources:
                        pids.append(slug)
                read_pages.extend(pages)
                retrieved_pids.extend(pids)
                trace.append({"tool": "wiki_read", "arg": paths,
                              "pages": pages, "source_pids": list(dict.fromkeys(pids))})
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
    return {"pred": pred, "raw": final, "trace": trace, "reads": reads,
            "tool_calls": sum(1 for t in trace if "tool" in t),
            "nudges": sum(1 for t in trace if "nudge" in t),
            "read_pages": list(dict.fromkeys(read_pages)),
            "retrieved_pids": list(dict.fromkeys(retrieved_pids))}
