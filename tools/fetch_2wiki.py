"""Build a balanced real 2WikiMultihopQA slice (corpus.jsonl + questions.json).

Why this exists
---------------
The old `data/` set was hand-authored in an earlier session (distractors literally say
"This page is a geographic distractor"). That is fine as a teaching example but
invalid as evidence. This script swaps in the REAL benchmark at the same scale.

What we gain over the hand-authored set
  * real Wikipedia passages (mean ~58 words vs our 31; distractors don't confess)
  * `supporting_facts` -> gold evidence annotation, so evidence-recall becomes a
    first-class metric instead of my manual annotation
  * `evidences` -> gold (subject, relation, object) triples per question
  * the paper's Appendix H Case 1 exists verbatim in the benchmark:
      "Which film has the director who is older, The Gamecock (Film) or Monster A Go-Go?"

Source: framolfese/2WikiMultihopQA (HotpotQA-style repack of Alab-NII/2wikimultihop)

    python3 tools/fetch_2wiki.py --n 16 --out data/2wiki

``--n`` must be a multiple of four.  The sampler selects the same number of
questions for each official 2Wiki type, while always retaining the LLM-Wiki
paper's Appendix H ``Monster A Go-Go`` case as the first question.
"""
import json
import os
import sys
import time
import urllib.parse
import urllib.error
import urllib.request

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
API = "https://datasets-server.huggingface.co"
DS = urllib.parse.quote("framolfese/2WikiMultihopQA", safe="")

# The paper's Appendix H Case 1, verbatim in the benchmark.
ANCHOR_ID = "f1145dc6084f11ebbd56ac1f6bf848b6"

# 2Wiki question types -> the hop label our harness reports.
HOP = {"comparison": 2, "compositional": 2, "inference": 2, "bridge_comparison": 4}


def _get(path):
    # The public dataset-server occasionally returns a transient 5xx response.
    # Retrying here keeps a dataset refresh reproducible without changing the
    # selected question order.
    for attempt in range(3):
        try:
            with urllib.request.urlopen(f"{API}/{path}", timeout=60) as r:
                return json.load(r)
        except (urllib.error.HTTPError, urllib.error.URLError):
            if attempt == 2:
                raise
            time.sleep(attempt + 1)


def rows(offset, length):
    return [x["row"] for x in _get(
        f"rows?dataset={DS}&config=default&split=validation"
        f"&offset={offset}&length={length}")["rows"]]


def candidates(limit=500, page_size=100):
    """Read a deterministic validation prefix, paging the dataset API."""
    out = []
    for offset in range(0, limit, page_size):
        out.extend(rows(offset, min(page_size, limit - offset)))
    return out


def search(q, length=20):
    return [x["row"] for x in _get(
        f"search?dataset={DS}&config=default&split=validation"
        f"&query={urllib.parse.quote(q)}&offset=0&length={length}")["rows"]]


def main():
    n = 16
    if "--n" in sys.argv:
        n = int(sys.argv[sys.argv.index("--n") + 1])
    out_name = "data/2wiki"
    if "--out" in sys.argv:
        out_name = sys.argv[sys.argv.index("--out") + 1].rstrip("/")
    out = os.path.join(HERE, out_name)

    if n < 4 or n % 4:
        raise SystemExit("--n must be a multiple of four (one balanced quota per question type)")

    picked, seen_ids = [], set()

    # 1) anchor: the Appendix H case
    for r in search("The Gamecock", 20):
        if r["id"] == ANCHOR_ID:
            picked.append(r)
            seen_ids.add(r["id"])
            break
    if not picked:
        raise SystemExit("anchor question not found; the dataset mirror may have changed")

    # 2) Fill exact, balanced quotas.  Keep the anchor as q1 so the paper's
    # Appendix H trace remains easy to compare with the local reproduction.
    per_type = n // 4
    want = {t: per_type for t in HOP}
    want[picked[0]["type"]] -= 1
    for r in candidates():
        if r["id"] in seen_ids:
            continue
        t = r["type"]
        if want.get(t, 0) > 0:
            want[t] -= 1
            picked.append(r)
            seen_ids.add(r["id"])
    if any(want.values()):
        raise SystemExit(f"not enough rows to fill balanced sample: remaining quotas={want}")

    # 3) pool passages, dedup by title (titles are the join key for supporting_facts)
    title2pid, corpus = {}, []
    for r in picked:
        c = r["context"]
        for title, sents in zip(c["title"], c["sentences"]):
            if title in title2pid:
                continue
            pid = f"p{len(corpus) + 1:03d}"
            title2pid[title] = pid
            corpus.append({
                "pid": pid,
                "title": title,
                "text": " ".join(sents).strip(),
                # Keep source boundaries for sentence-level evidence audits.
                "sentences": sents,
            })

    # 4) questions + GOLD EVIDENCE from supporting_facts (this is the real payoff)
    questions = []
    for i, r in enumerate(picked, 1):
        # Preserve the official, per-question candidate set as well as the
        # pooled corpus.  The former lets us audit the original 10-context
        # setting; the latter is what the RAG arms actually index.
        context_pids = [title2pid[title] for title in r["context"]["title"]]
        gold_titles = list(dict.fromkeys(r["supporting_facts"]["title"]))
        gold_pids = [title2pid[t] for t in gold_titles if t in title2pid]
        gold_supporting_facts = [
            {"pid": title2pid[title], "title": title, "sent_id": sent_id}
            for title, sent_id in zip(r["supporting_facts"]["title"],
                                      r["supporting_facts"]["sent_id"])
            if title in title2pid
        ]
        questions.append({
            "id": f"q{i}",
            "question": r["question"],
            "answer": r["answer"],
            "hop": HOP.get(r["type"], 2),
            "type": r["type"],
            "context_pids": context_pids,                 # official 10 candidates, in order
            "gold_pids": gold_pids,                        # ← evidence-recall label
            "distractor_pids": [pid for pid in context_pids if pid not in set(gold_pids)],
            "gold_supporting_facts": gold_supporting_facts, # ← sentence-level audit label
            "evidences": r.get("evidences"),               # ← gold triples
            "src_id": r["id"],
            "note": "2WikiMultihopQA validation (real benchmark)"
            + (" — paper Appendix H Case 1" if r["id"] == ANCHOR_ID else ""),
        })

    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "corpus.jsonl"), "w") as f:
        for x in corpus:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")
    json.dump(questions, open(os.path.join(out, "questions.json"), "w"),
              ensure_ascii=False, indent=1)

    words = [len(x["text"].split()) for x in corpus]
    print(f"wrote {out}/")
    print(f"  corpus    : {len(corpus)} passages | words min {min(words)} "
          f"mean {sum(words) / len(words):.1f} max {max(words)}")
    print(f"  questions : {len(questions)}")
    for q in questions:
        print(f"    {q['id']} hop{q['hop']:<2} [{q['type']:<18}] {q['question'][:64]}")
        print(f"         answer={q['answer']!r} gold={q['gold_pids']}")


if __name__ == "__main__":
    main()
