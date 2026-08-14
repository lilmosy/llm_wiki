"""B4. GraphRAG -- QUERY TIME (map-reduce over community reports).

MAP:    ask EVERY community report the question independently -> partial answers
REDUCE: hand all partial answers to the LLM -> final answer

This is the most expensive arm per question: 1 LLM call per community + 1 reduce.
GraphRAG was designed for global sensemaking queries ("what are the themes?"),
so a 16-passage corpus with 2-3 communities does not show its intended strength
-- noted as a limitation in REPORT.md.
"""
import json
import os

from core.llm import complete
from baseline._common import parse_answer, index_dir, ANSWER_SYS

_MAP_SYS = ("You are given ONE community report from a knowledge graph. Answer the "
            "question using ONLY that report. If the report does not contain the "
            "needed information, reply exactly: IRRELEVANT. Otherwise give the answer "
            "plus the supporting facts in at most 3 sentences.")


class GraphRag:
    needs_build = True
    arm = "graphrag"

    def __init__(self, cfg):
        self.cfg = cfg
        self.comms = None

    def load(self):
        d = index_dir(self.arm)
        self.comms = json.load(open(os.path.join(d, "graph.json")))["communities"]

    def answer(self, question):
        if self.comms is None:
            self.load()
        partials, map_trace, selected_pids = [], [], []
        for c in self.comms:
            out = complete(_MAP_SYS, f"Community report:\n{c['report']}\n\nQuestion: {question}",
                           self.cfg["model"], 400).strip()
            relevant = "IRRELEVANT" not in out.upper()
            map_trace.append({"community_id": c["id"], "source_pids": c.get("sources", []),
                              "relevant": relevant})
            if relevant:
                partials.append(f"[{c['id']}] {out}")
                selected_pids.extend(c.get("sources", []))
        if not partials:
            partials = ["(no community produced a relevant partial answer)"]
        ctx = "\n\n".join(partials)
        raw = complete(ANSWER_SYS,
                       f"Context (partial answers from graph communities):\n{ctx}\n\n"
                       f"Question: {question}",
                       self.cfg["model"], 500)
        return {"pred": parse_answer(raw), "raw": raw,
                "retrieved": [c["id"] for c in self.comms],
                "retrieved_pids": list(dict.fromkeys(selected_pids)),
                "map_trace": map_trace, "partials": partials}
