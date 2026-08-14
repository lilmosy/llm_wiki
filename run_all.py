"""End-to-end runner over N arms.

  Phase 1  build   -- every arm that needs an OFFLINE INDEX builds it once
  Phase 2  query   -- every arm answers every question
  Phase 3  score   -- shared harness, then REPORT.md

Adding an arm = write the module, add ONE entry to ARMS below. Nothing else in
this file (or in make_report.py) is arm-specific.

    python3 run_all.py            full run (rebuilds every index)
    python3 run_all.py --reuse    reuse existing indexes/ (query only)
    python3 run_all.py --dry-run  wire everything up, load indexes, make 0 API calls
    python3 run_all.py --only llm_wiki,bm25    restrict to some arms
    python3 run_all.py --questions q2,q7,q3,q12,q1   query a diagnostic subset
"""
import json
import hashlib
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import yaml
from core.llm import USAGE
from core.evaluate import score

HERE = os.path.dirname(os.path.abspath(__file__))
cfg = yaml.safe_load(open(os.path.join(HERE, "config.yaml")))

# --data lets datasets live beneath data/. Everything an arm writes (indexes,
# wiki tree, results) gets a stable dataset namespace so a swap never clobbers
# another run. `data/2wiki` -> ns "2wiki"; `data/musique` -> ns "musique".
DATA = "data/2wiki"
if "--data" in sys.argv:
    DATA = sys.argv[sys.argv.index("--data") + 1].rstrip("/")
if DATA.startswith("data/"):
    NS = DATA[len("data/"):].replace("/", "_")
elif DATA.startswith("data_"):  # compatibility with earlier flat layout
    NS = DATA[5:]
else:
    NS = ""
SFX = f"_{NS}" if NS else ""

corpus = [json.loads(l) for l in open(os.path.join(HERE, DATA, "corpus.jsonl")) if l.strip()]
questions = json.load(open(os.path.join(HERE, DATA, "questions.json")))
ALL_QUESTION_IDS = [q["id"] for q in questions]
QUESTION_FILTER = None
if "--questions" in sys.argv:
    QUESTION_FILTER = [q.strip() for q in sys.argv[sys.argv.index("--questions") + 1].split(",") if q.strip()]
    unknown = [q for q in QUESTION_FILTER if q not in ALL_QUESTION_IDS]
    if unknown:
        raise SystemExit(f"unknown question id(s): {', '.join(unknown)}; available: {', '.join(ALL_QUESTION_IDS)}")
    wanted = set(QUESTION_FILTER)
    # Preserve the canonical file order rather than the command-line order so
    # rows remain directly comparable with a later all-question run.
    questions = [q for q in questions if q["id"] in wanted]

# Reusing an index compiled from another corpus or configuration silently
# invalidates every comparison. Keep one stable fingerprint for this run and
# write it beside each offline artifact after a successful build.
_corpus_bytes = open(os.path.join(HERE, DATA, "corpus.jsonl"), "rb").read()
_questions_bytes = open(os.path.join(HERE, DATA, "questions.json"), "rb").read()
DATA_FINGERPRINT = hashlib.sha256(_corpus_bytes + b"\0" + _questions_bytes).hexdigest()
CONFIG_FINGERPRINT = hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()

from baseline._common import set_index_namespace
set_index_namespace(NS)

# Every arm now writes under indexes/<arm>/, LLM-Wiki included -- the md tree
# is that arm's index, not a separate kind of artifact.
WIKI_DIR = os.path.join(HERE, "indexes" + SFX, "llm_wiki", "wiki")
RESULTS = os.path.join(HERE, f"runs/results{SFX}.json")
COMPILE_INFO = os.path.join(HERE, f"runs/compile_info{SFX}.json")

# Guard: this file is an entry-point script, not an importable module. Importing
# it must NOT run the pipeline (hundreds of API calls + overwrites results.json).
if __name__ != "__main__":
    raise SystemExit("run_all.py is a script; run `python3 run_all.py`, do not import it.")

REUSE = "--reuse" in sys.argv
# --dry-run wires up every arm and loads every on-disk index, then stops before
# the first API call. It is how a refactor gets verified when the pipeline
# itself costs hundreds of calls: if the registry, the imports and the artifact
# paths are wrong, this fails in seconds and for free.
DRY = "--dry-run" in sys.argv
ONLY = None
if "--only" in sys.argv:
    ONLY = set(sys.argv[sys.argv.index("--only") + 1].split(","))


def mean(xs):
    return sum(xs) / len(xs) if xs else 0.0


def calls():
    return USAGE["calls"]


# ===========================================================================
# ARM REGISTRY  -- the whole point of the refactor.
#   name      : key used in results.json / REPORT.md
#   label     : display name
#   paradigm  : paper §4.2 grouping
#   build     : callable() -> stats dict, or None if the arm needs no index
#   make      : callable() -> object with .answer(question)
# ===========================================================================
def _arms():
    from baseline.llm_wiki.build import run_compile
    from baseline.llm_wiki.wiki import Wiki
    from baseline.llm_wiki.answer import answer_question
    from baseline.b1_bm25.answer import BM25Rag
    from baseline.b0_closed_book.answer import ClosedBook
    from baseline.b2_dense.answer import DenseRag
    from baseline.b3_raptor.build import build as raptor_build
    from baseline.b3_raptor.answer import Raptor
    from baseline.b4_graphrag.build import build as graphrag_build
    from baseline.b4_graphrag.answer import GraphRag
    from baseline.b5_lightrag.build import build as lightrag_build
    from baseline.b5_lightrag.answer import LightRag
    from baseline.b6_hipporag.build import build as hipporag_build
    from baseline.b6_hipporag.answer import HippoRag

    state = {}

    def wiki_build():
        w, eb, cinfo = run_compile(corpus, cfg, WIKI_DIR,
                                   os.path.join(HERE, f"indexes{SFX}", "llm_wiki", "error_book.yaml"))
        state["wiki"], state["eb"], state["cinfo"] = w, eb, cinfo
        pages_by_source = {x["pid"]: [] for x in corpus}
        for slug, page in w.pages.items():
            for pid in page.get("sources", []):
                if pid in pages_by_source:
                    pages_by_source[pid].append(slug)
        gold_pids = {pid for q in questions for pid in q.get("gold_pids", [])}
        covered = {pid for pid, pages in pages_by_source.items() if pages}
        cinfo["coverage"] = {
            "source_passages": len(corpus),
            "passages_with_compiled_page": len(covered),
            "missing_source_pids": sorted(set(pages_by_source) - covered),
            "gold_passages": len(gold_pids),
            "gold_passages_with_compiled_page": len(gold_pids & covered),
            "missing_gold_pids": sorted(gold_pids - covered),
            "pages_by_source": pages_by_source,
        }
        json.dump(cinfo, open(COMPILE_INFO, "w"), indent=1, default=str)
        return {"pages": len(w.pages), "sources": len(w.sources),
                "structural_errors": len(cinfo["final_errors"]),
                "passage_coverage": f"{len(covered)}/{len(corpus)}",
                "gold_coverage": f"{len(gold_pids & covered)}/{len(gold_pids)}"}

    def wiki_load():
        state["wiki"] = Wiki.load(WIKI_DIR)
        state["cinfo"] = json.load(open(COMPILE_INFO)) if os.path.exists(COMPILE_INFO) else None
        return {"pages": len(state["wiki"].pages)}

    class _WikiArm:
        def answer(self, q):
            return answer_question(q, state["wiki"], cfg)

    dense = DenseRag(corpus, cfg)
    lightrag = LightRag(corpus, cfg)
    hipporag = HippoRag(corpus, cfg)

    # "artifact" = the file whose existence proves the offline index is already
    # built. --reuse only skips a build when this file is present, so a partial
    # run can be resumed without silently querying a missing index.
    return [
        {"name": "llm_wiki", "label": "LLM-Wiki (ours)", "paradigm": "agent-native wiki",
         "artifact": f"indexes{SFX}/llm_wiki/wiki/_manifest.json",
         "build": wiki_build, "load": wiki_load, "make": lambda: _WikiArm()},

        {"name": "closed_book", "label": "None (Closed-book)", "paradigm": "no retrieval",
         "artifact": None, "build": None, "load": None, "make": lambda: ClosedBook(cfg)},

        {"name": "bm25", "label": "Vanilla RAG (BM25)", "paradigm": "flat sparse",
         "artifact": None, "build": None, "load": None, "make": lambda: BM25Rag(corpus, cfg)},

        {"name": "dense", "label": "Vanilla RAG (Dense)", "paradigm": "flat dense",
         "artifact": f"indexes{SFX}/dense/vectors.npz",
         "build": dense.build, "load": dense.load, "make": lambda: dense},

        {"name": "raptor", "label": "RAPTOR", "paradigm": "hierarchical summary tree",
         "artifact": f"indexes{SFX}/raptor/tree.json",
         "build": lambda: raptor_build(corpus, cfg), "load": None, "make": lambda: Raptor(cfg)},

        {"name": "graphrag", "label": "GraphRAG", "paradigm": "graph + community reports",
         "artifact": f"indexes{SFX}/graphrag/graph.json",
         "build": lambda: graphrag_build(corpus, cfg), "load": None, "make": lambda: GraphRag(cfg)},

        {"name": "lightrag", "label": "LightRAG", "paradigm": "graph dual-level",
         "artifact": f"indexes{SFX}/lightrag/graph.json",
         "build": lambda: lightrag_build(corpus, cfg), "load": lightrag.load,
         "make": lambda: lightrag},

        {"name": "hipporag", "label": "HippoRAG 2", "paradigm": "graph + PPR",
         "artifact": f"indexes{SFX}/hipporag/triples.json",
         "build": lambda: hipporag_build(corpus, cfg), "load": hipporag.load,
         "make": lambda: hipporag},
    ]


ARMS = [a for a in _arms() if not ONLY or a["name"] in ONLY]
os.makedirs(os.path.join(HERE, "runs", "history"), exist_ok=True)


def _build_meta_path(arm):
    return os.path.join(HERE, f"indexes{SFX}", arm["name"], "_build_meta.json")


def _reuse_is_compatible(arm):
    path = _build_meta_path(arm)
    try:
        meta = json.load(open(path))
        return (meta.get("data_fingerprint") == DATA_FINGERPRINT and
                meta.get("config_fingerprint") == CONFIG_FINGERPRINT)
    except (OSError, ValueError, json.JSONDecodeError):
        return False


def _write_build_meta(arm):
    path = _build_meta_path(arm)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    json.dump({"data": DATA, "data_fingerprint": DATA_FINGERPRINT,
               "config_fingerprint": CONFIG_FINGERPRINT}, open(path, "w"), indent=1)

# ---------------------------------------------------------------- Phase 1
q_note = "" if QUESTION_FILTER is None else f" | query filter={','.join(QUESTION_FILTER)}"
print(f"[Phase 1] data={DATA} ({len(corpus)} passages, {len(questions)} q) | {len(ARMS)} arms, reuse={REUSE}{q_note}", flush=True)
meta = []
for a in ARMS:
    before = calls()
    t0 = time.time()
    stats, note = {}, ""
    have = a["artifact"] and os.path.exists(os.path.join(HERE, a["artifact"]))
    compatible = have and _reuse_is_compatible(a)
    if DRY:
        stats = a["load"]() if (a["load"] and have) else {}
        note = "dry-run: " + ("loaded" if have else "no artifact")
    elif a["build"] is None:
        note = "no index"
    elif REUSE and compatible:
        stats = a["load"]() if a["load"] else {}
        note = "reused (fingerprint match)"
    else:
        stats = a["build"]() or {}
        if a["artifact"]:
            _write_build_meta(a)
        note = "built" + ("" if not REUSE else
                           (" (no artifact to reuse)" if not have else " (fingerprint mismatch)"))
    spent = calls() - before
    meta.append({"name": a["name"], "label": a["label"], "paradigm": a["paradigm"],
                 "build_calls": spent, "build_stats": stats, "build_note": note,
                 "build_sec": round(time.time() - t0, 1)})
    print(f"  {a['name']:<12} {note:<24} {spent:>3} LLM calls  {stats}", flush=True)

RUNNERS = {a["name"]: a["make"]() for a in ARMS}

if DRY:
    print("\n[dry-run] arms constructed and indexes loaded; stopping before any API call")
    for a in ARMS:
        art = a["artifact"]
        ok = "-" if not art else ("있음" if os.path.exists(os.path.join(HERE, art)) else "없음")
        print(f"  {a['name']:<12} {type(RUNNERS[a['name']]).__name__:<12} artifact={art or '(없음)'} [{ok}]")
    print(f"\nAPI 호출: {calls()}회")
    raise SystemExit(0)

# ---------------------------------------------------------------- Phase 2
print(f"\n[Phase 2] answering {len(questions)} questions x {len(ARMS)} arms", flush=True)
rows = []
for q in questions:
    per = {}
    for a in ARMS:
        before = calls()
        query_t0 = time.time()
        try:
            out = RUNNERS[a["name"]].answer(q["question"])
        except Exception as e:
            out = {"pred": "", "raw": f"ERROR: {type(e).__name__}: {e}", "error": True}
        out["query_sec"] = round(time.time() - query_t0, 3)
        out["score"] = score(out.get("pred", ""), q["answer"])
        out["calls"] = calls() - before
        # Evidence recall is measured against raw-passage provenance. Each arm
        # exposes retrieved_pids even when its answer context was a compiled page,
        # summary node, or community report. `retrieved` remains a method-native
        # trace (page slug, RAPTOR node, community id, etc.).
        gold = set(q.get("gold_pids") or [])
        if "retrieved_pids" in out:
            got = set(out["retrieved_pids"])
            out["evidence_recall"] = len(gold & got) / len(gold) if gold else None
        else:
            # Compatibility for a third-party arm that has not yet adopted the
            # standard trace field. Its recall is intentionally left unknown.
            out["evidence_recall"] = None
        per[a["name"]] = out
    rows.append({**q, "arms": per})
    flags = "  ".join(f"{a['name'][:8]}:{'O' if per[a['name']]['score']['cover'] else 'X'}" for a in ARMS)
    print(f"  {q['id']} hop{q['hop']}  {flags}", flush=True)

# --only re-runs a subset. Writing the payload as-is would drop every arm that
# was not selected, so merge this run's arms over the previous results file.
if ONLY and os.path.exists(RESULTS):
    prev = json.load(open(RESULTS))
    order = [m["name"] for m in prev.get("arms", [])]
    by_name = {m["name"]: m for m in prev.get("arms", []) if m["name"] not in ONLY}
    by_name.update({m["name"]: m for m in meta})
    meta = [by_name[n] for n in order if n in by_name] + \
           [m for m in meta if m["name"] not in order]
    prev_rows = {r["id"]: r for r in prev.get("rows", [])}
    for r in rows:                       # same questions, so a plain per-row merge
        old = prev_rows.get(r["id"])
        if old:
            r["arms"] = {**old["arms"], **r["arms"]}

payload = {"arms": meta, "rows": rows, "usage": dict(USAGE), "config": cfg,
           "data": {"path": DATA, "fingerprint": DATA_FINGERPRINT,
                    "passages": len(corpus), "questions": len(questions),
                    "question_filter": QUESTION_FILTER},
           "compile": None}
if ONLY:
    payload["usage_note"] = (f"partial re-run (--only {','.join(sorted(ONLY))}); "
                             "'usage' counts only these arms. Per-arm build/query "
                             "call counts in arms[]/rows[] stay accurate.")
if os.path.exists(COMPILE_INFO):
    payload["compile"] = json.load(open(COMPILE_INFO))

json.dump(payload, open(RESULTS, "w"), indent=1, default=str)
stamp = time.strftime("%Y%m%d_%H%M%S")
json.dump(payload, open(os.path.join(HERE, f"runs/history/{stamp}{SFX}.json"), "w"), indent=1, default=str)
print(f"\n  -> {RESULTS}  (+ runs/history/{stamp}{SFX}.json)", flush=True)

# ---------------------------------------------------------------- Phase 3
os.system(f'python3 "{os.path.join(HERE, "analysis", "make_report.py")}" {NS}')
print(f"[done] {USAGE['calls']} LLM calls | in {USAGE['input_tokens']:,} tok | out {USAGE['output_tokens']:,} tok",
      flush=True)
