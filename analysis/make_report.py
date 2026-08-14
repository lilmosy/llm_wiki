"""Offline report builder: runs/results.json -> REPORT.md. Makes NO API calls.

Handles both schemas:
  new: {"arms": [...], "rows": [{..., "arms": {name: {...}}}]}
  old: {"rows": [{..., "wiki": {}, "base": {}, "wiki_score": {}, "base_score": {}}]}
so the pre-refactor run (the Appendix H reproduction) still renders.
"""
import json
import os
from collections import defaultdict

import sys
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NS = sys.argv[1] if len(sys.argv) > 1 and sys.argv[1] else ""
SFX = f"_{NS}" if NS else ""
D = json.load(open(os.path.join(HERE, f"runs/results{SFX}.json")))
rows = D["rows"]


def mean(xs):
    return sum(xs) / len(xs) if xs else 0.0


# ---- schema migration (old 2-arm format -> new N-arm format) ---------------
if "arms" not in D or not isinstance(D.get("arms"), list):
    for r in rows:
        r["arms"] = {
            "llm_wiki": {**r.get("wiki", {}), "score": r.get("wiki_score", {}),
                         "calls": r.get("wiki", {}).get("tool_calls", 0)},
            "bm25": {**r.get("base", {}), "score": r.get("base_score", {}), "calls": 1},
        }
    D["arms"] = [{"name": "llm_wiki", "label": "LLM-Wiki (ours)", "paradigm": "agent-native wiki",
                  "build_calls": D.get("usage", {}).get("calls", 0), "build_stats": {},
                  "build_note": "legacy run"},
                 {"name": "bm25", "label": "Vanilla RAG (BM25)", "paradigm": "flat sparse",
                  "build_calls": 0, "build_stats": {}, "build_note": "no index"}]

META = D["arms"]
NAMES = [m["name"] for m in META]
LABEL = {m["name"]: m["label"] for m in META}
N = len(rows)
DATA_PATH = D.get("data", {}).get("path", "")
DATASET_LABEL = "MuSiQue-Ans dev slice" if DATA_PATH.endswith("musique") else "2WikiMultihopQA slice"


def agg(name, metric):
    return mean([r["arms"][name]["score"].get(metric, 0) for r in rows if name in r["arms"]])


def hits(name):
    return sum(1 for r in rows if r["arms"].get(name, {}).get("score", {}).get("cover") == 1)


L = []
A = L.append
A(f"# LLM-Wiki 재현 MVP — {len(NAMES)}개 방법 비교 리포트\n")
A("> 논문 *Retrieval as Reasoning: Self-Evolving Agent-Native Retrieval via LLM-Wiki* (arXiv 2605.25480) §4.2의")
A(f"> **이번에 선택한 {len(NAMES)}개 방법**을 동일 조건에서 **간이 재현(simplified reimplementation)**.")
NPASS = len({r for t in (D.get("compile") or {}).get("trace", []) for r in [t["pid"]]}) or "?"
A(f"> 코퍼스 {len(rows)}문항 / {NPASS} passage ({DATASET_LABEL}). "
  f"답변 LLM = `{D.get('config', {}).get('model', '?')}` **전 arm 동일**.")
A(f"> 임베딩 = `{D.get('config', {}).get('embed_model', '-')}` (논문은 Qwen3-Embedding-8B — 대체).\n")
A("> ⚠️ **논문 수치 재현이 아니다.** 논문은 3개 데이터셋 각 500문항, GLM-5.1 백본. 목표는")
A("> 절대 수치 복제가 아니라 **각 방식이 멀티홉에서 어디서 무너지는지의 경향 확인**이다.\n")

# ---------------------------------------------------------------- 0
A("## 0. 한눈에 보기 (cover 내림차순)\n")
A("| arm | paradigm | **cover** | F1 | EM | 질의 LLM 호출/문항 | 질의 시간/문항 |")
A("|---|---|---|---|---|---|---|")
order = sorted(NAMES, key=lambda n: (-agg(n, "cover"), -agg(n, "f1")))
for n in order:
    qc = mean([r["arms"][n].get("calls", 0) for r in rows if n in r["arms"]])
    qt = mean([r["arms"][n].get("query_sec", 0) for r in rows if n in r["arms"]])
    star = "**" if n == "llm_wiki" else ""
    A(f"| {star}{LABEL[n]}{star} | {next(m['paradigm'] for m in META if m['name'] == n)} | "
      f"{star}{agg(n, 'cover'):.3f} ({hits(n)}/{N}){star} | {agg(n, 'f1'):.3f} | {agg(n, 'em'):.3f} | {qc:.1f} | {qt:.1f}s |")
A("\n- **cover** = 정답이 예측에 포함되는가 (장황한 답변에 강건한 정확도 지표). 주 판정 기준.")
A("- F1/EM = 토큰 overlap / 완전일치. 답변을 최단 스팬으로 강제했으나 여전히 표기 변형에 민감.\n")

# ---------------------------------------------------------------- 0b
_has_ev = any(r["arms"].get(n, {}).get("evidence_recall") is not None
              for r in rows for n in NAMES)
if _has_ev:
    A("## 0b. 근거 회수율 — 정답 근거 문단을 실제로 회수했나\n")
    A("데이터셋의 `supporting_facts`(gold_pids)를 정답 라벨로 쓴 자동 채점.")
    A("**정답률과 어긋나는 arm이 진짜 성질을 드러낸다.**\n")
    case_for_evidence = max(rows, key=lambda r: r.get("hop", 0), default=None)
    case_id = case_for_evidence["id"] if case_for_evidence else "–"
    A(f"| arm | 평균 근거 회수율 | {case_id}(최고 hop) | 정답률 |")
    A("|---|---|---|---|")
    for n in order:
        vs = [r["arms"][n]["evidence_recall"] for r in rows
              if r["arms"].get(n, {}).get("evidence_recall") is not None]
        if not vs:
            A(f"| {LABEL[n]} | – (문단 id 없음) | – | {hits(n)}/{N} |")
            continue
        case_ev = next((r["arms"][n]["evidence_recall"] for r in rows if r["id"] == case_id), None)
        A(f"| {LABEL[n]} | {mean(vs):.3f} | {'-' if case_ev is None else f'{case_ev:.2f}'} | {hits(n)}/{N} |")
    A("\n> `–`는 문단 id가 아니라 요약/커뮤니티를 회수하는 arm(GraphRAG·RAPTOR 일부·LLM-Wiki)이라 "
      "문단 단위 측정이 정의되지 않는 경우다.\n")

# ---------------------------------------------------------------- 1
A("## 1. 오프라인 인덱스 빌드 비용 (논문의 '컴파일 비용' 논점)\n")
A("정확도만 비교하면 인덱스를 미리 만든 arm의 선불 비용이 안 보인다. 빌드/질의 비용을 분리해 기록한다.\n")
A("| arm | 오프라인 빌드 | 빌드 LLM 호출 | 빌드 산출물 | 질의 호출/문항 |")
A("|---|---|---|---|---|")
ART = {"llm_wiki": "`wiki/` (md 트리)", "dense": "`indexes/dense/`", "raptor": "`indexes/raptor/TREE.md`",
       "graphrag": "`indexes/graphrag/COMMUNITIES.md`", "lightrag": "`indexes/lightrag/ENTITIES.md`",
       "hipporag": "`indexes/hipporag/GRAPH.md`", "bm25": "–", "closed_book": "–"}
for m in META:
    n = m["name"]
    qc = mean([r["arms"][n].get("calls", 0) for r in rows if n in r["arms"]])
    need = "❌ 없음" if m.get("build_note") == "no index" else (
        "⚠️ 임베딩만" if m.get("build_calls", 0) == 0 else "✅ 필수")
    st = ", ".join(f"{k}={v}" for k, v in (m.get("build_stats") or {}).items())
    A(f"| {LABEL[n]} | {need} | {m.get('build_calls', 0)} | {ART.get(n, '–')} {st} | {qc:.1f} |")
A(f"\n- 총 LLM 호출: **{D.get('usage', {}).get('calls', 0)}회** "
  f"(입력 {D.get('usage', {}).get('input_tokens', 0):,} tok / 출력 {D.get('usage', {}).get('output_tokens', 0):,} tok)")
_cached = D.get('usage', {}).get('cached', 0)
if _cached:
    A(f"- 응답 캐시 히트: **{_cached}회** (API로 나가지 않음). "
      f"위 호출 수는 실제 API 호출만 센 것이므로 콜드 런 비용과 비교 가능하다.")
A("- 빌드는 **1회**, 질의는 **문항마다**. 코퍼스가 커지면 빌드 비용은 상각되고 질의 비용이 지배한다.\n")

# ---------------------------------------------------------------- 2
A("## 2. 오염 점검 — Closed-book이 몇 개 맞히나\n")
if "closed_book" in NAMES:
    cb = [r for r in rows if r["arms"].get("closed_book", {}).get("score", {}).get("cover") == 1]
    A(f"**Closed-book cover = {hits('closed_book')}/{N}.**\n")
    if cb:
        A("검색 없이 맞힌 문항 (= 검색 능력을 측정하지 못하는 문항):\n")
        for r in cb:
            A(f"- `{r['id']}` (hop{r['hop']}) {r['question']} → 예측 `{r['arms']['closed_book']['pred']}`")
        A(f"\n→ 나머지 **{N - len(cb)}문항**이 검색을 실제로 요구하는 유효 문항이다. "
          "arm 간 비교는 이 점을 감안해 읽어야 한다.\n")
    else:
        A("→ 한 문항도 맞히지 못했다. **전 문항이 검색을 요구하는 유효 문항**이다.\n")
else:
    A("(closed_book arm 미실행)\n")

# ---------------------------------------------------------------- 3
A("## 3. hop 깊이별 정확도 (논문 핵심 주장: hop↑ → 구조화 이득↑)\n")
by_hop = defaultdict(lambda: defaultdict(list))
for r in rows:
    for n in NAMES:
        if n in r["arms"]:
            by_hop[r["hop"]][n].append(r["arms"][n]["score"].get("cover", 0))
A("| hop | " + " | ".join(LABEL[n] for n in NAMES) + " |")
A("|---" * (len(NAMES) + 1) + "|")
for h in sorted(by_hop):
    cells = " | ".join(f"{mean(by_hop[h][n]):.2f}" for n in NAMES)
    A(f"| {h}-hop ({len(by_hop[h][NAMES[0]])}문항) | {cells} |")
A("")

# ---------------------------------------------------------------- 4
A("## 4. 문항별 정답 매트릭스 (O = cover 성공)\n")
A("| id | hop | 질문 | 정답 | " + " | ".join(LABEL[n] for n in NAMES) + " |")
A("|---|---|---|---" + "|---" * len(NAMES) + "|")
for r in rows:
    cells = " | ".join("**O**" if r["arms"].get(n, {}).get("score", {}).get("cover") else "X" for n in NAMES)
    A(f"| {r['id']} | {r['hop']} | {r['question'][:52]} | {r['answer']} | {cells} |")
A("")

# ---------------------------------------------------------------- 4b
A("## 4b. 채점 아티팩트 점검 (오답으로 찍혔지만 실제로는 맞은 것)\n")
A("`cover`는 정답 문자열이 예측에 **포함**되는지를 본다. 그래서 예측이 정답보다 **짧으면**")
A("(예: 정답 `Karl I, Prince of Anhalt-Zerbst` vs 예측 `Karl I`) 내용이 맞아도 오답으로 찍힌다.")
A("최단 스팬 강제 지시가 과하게 먹은 경우다. 자동 검출:\n")
import sys as _s
_s.path.insert(0, HERE)
from core.evaluate import normalize as _nrm
near = []
for r in rows:
    for n in NAMES:
        a = r["arms"].get(n, {})
        p, g = _nrm(str(a.get("pred", ""))), _nrm(r["answer"])
        if a.get("score", {}).get("cover") == 0 and p and p in g:
            near.append((r["id"], n, a.get("pred"), r["answer"]))
if near:
    A("| 문항 | arm | 예측 | 정답 | 판정 |")
    A("|---|---|---|---|---|")
    for qid, n, p, g in near:
        A(f"| {qid} | {LABEL[n]} | `{p}` | `{g}` | 내용 일치, **표기 절단**으로 오답 처리 |")
    adj = defaultdict(int)
    for _, n, _, _ in near:
        adj[n] += 1
    A("\n**절단분을 정답으로 보정하면:** " + ", ".join(
        f"{LABEL[n]} {hits(n)}/{N} → **{hits(n) + adj[n]}/{N}**" for n in adj) + "\n")
    A("> 이건 방법의 성능 차이가 아니라 **채점기와 답변 형식의 상호작용**이다. "
      "스케일업 시 정답 별칭(alias) 목록을 두거나 LLM judge로 판정해야 한다.\n")
else:
    A("→ 절단형 근접 오답 없음.\n")

# ---------------------------------------------------------------- 5
q1 = next((r for r in rows if r["id"] == "q1"), None)
case = q1 or max(rows, key=lambda r: r.get("hop", 0), default=None)
case_title = "4-hop 케이스 상세 (논문 Appendix H Case 1)" if q1 else "최고-hop 순차 chain 사례 상세"
A(f"## 5. {case_title}\n")
if case:
    A(f"**{case['question']}**  (정답: `{case['answer']}`)\n")
    A("| arm | 예측 | cover | 회수한 근거 |")
    A("|---|---|---|---|")
    for n in NAMES:
        a = case["arms"].get(n, {})
        ev = a.get("retrieved") or []
        if n == "llm_wiki":
            ev = [f"{s.get('tool')}({s.get('arg')})" for s in (a.get("trace") or [])][:4]
        A(f"| {LABEL[n]} | `{str(a.get('pred', ''))[:38]}` | "
          f"{'✅' if a.get('score', {}).get('cover') else '❌'} | {str(ev)[:110]} |")
    gold = ", ".join(case.get("gold_pids", []))
    if q1:
        A("\n이 문항은 **영화 → 감독 → 생년 → 비교**의 4단계다. 감독의 생년은 영화 문단에 없고 "
          f"별도 gold 문단({gold})에 있으므로, 한 번의 lexical 검색으로는 재료가 빠질 수 있다.\n")
    else:
        decomp = case.get("question_decomposition") or []
        steps = " → ".join(str(x.get("answer", "")) for x in decomp)
        A(f"\n공식 decomposition의 중간 답 경로는 **{steps}**다. gold paragraph는 `{gold}`이며, "
          "각 단계의 중간 답이 다음 검색·추론의 입력으로 이어져야 한다.\n")

# ---------------------------------------------------------------- 6
A("## 6. 컴파일(LLM-Wiki 인덱스) 상세\n")
C = D.get("compile")
if C:
    A(f"- passage별 SELECTPAGES + COMPILEWIKIPAGES 트레이스 {len(C.get('trace', []))}건")
    A(f"- 최종 미해결 오류: {len(C.get('final_errors', []))}건 / 코드 자동수정 {C.get('final_fixed', 0)}건")
    errs = [e for t in C.get("trace", []) for e in t.get("errors", [])]
    A(f"- 컴파일 중 감지된 오류: **{len(errs)}건**"
      + ("" if errs else " → **Error Book 자기교정 루프가 한 번도 발동하지 않았다** "
                        "(구현돼 있으나 미발화. 스케일업 시 관측 예정).") + "\n")
else:
    A("(compile 정보 없음 — `--reuse` 실행이었거나 legacy 결과)\n")

# ---------------------------------------------------------------- 7
best_base = max((n for n in NAMES if n != "llm_wiki"), key=lambda n: agg(n, "cover"), default=None)
is_2wiki = "2wiki" in str(D.get("data", {}).get("path", "")).lower()
A("## 7. " + ("논문과의 대조" if is_2wiki else "MuSiQue slice 해석 한계") + "\n")
if is_2wiki:
    A(f"| | 논문 (2WikiMHQA 500문항, GLM-5.1) | 이 MVP ({N}문항, {D.get('config', {}).get('model', '?')}) |")
    A("|---|---|---|")
    A(f"| 방법 수 | 7 baseline + LLM-Wiki | {len(NAMES)}개 arm |")
    A(f"| 최강 baseline | HippoRAG 2 | {LABEL.get(best_base, '-')} (cover {agg(best_base, 'cover'):.3f}) |")
    A(f"| LLM-Wiki 우위폭 | +2.0~8.1 F1p | cover {agg('llm_wiki', 'cover'):.3f} vs "
      f"차순위 {agg(best_base, 'cover'):.3f} |")
    A("| hop 효과 | 2→4hop 격차 5.7→8.3 F1p 증가 | 위 3절 표 참조 |")
A("\n> ⚠️ **해석 한계**")
A(f"> - 문항 {N}개는 통계적 유의성을 논할 규모가 아니다. arm 간 1문항 차이 = cover {1 / N:.3f}.")
A(f"> - 코퍼스 {NPASS} passage에서 top-5는 전체의 {5/int(NPASS)*100:.1f}%다. 이 slice는 메커니즘 해부용이지 벤치마크 성능 추정용이 아니다.")
A("> - 이 slice의 질문은 전역 요약이 아니라 국소·정확 멀티홉 질문이라, GraphRAG의 전역 질의 설계 의도가 직접 검증되지는 않는다.")
A("> - 임베딩 모델을 소형으로 대체했다. 전 arm 동일 적용이라 arm 간 비교는 유효하나 절대 성능은 논문보다 낮다.")
A("> - 다음 단계는 유형·hop별 문항을 늘려 같은 분석을 반복하는 것이다.\n")

A("## 8. 산출물 지도\n")
A("```")
_artifact_ns = NS or "2wiki"
A(f"indexes_{_artifact_ns}/llm_wiki/wiki/              LLM-Wiki 컴파일 결과 (md 트리)")
A(f"indexes_{_artifact_ns}/dense/INDEX.md              임베딩 행렬 메타")
A(f"indexes_{_artifact_ns}/graphrag/COMMUNITIES.md     커뮤니티 리포트")
A(f"runs/results_{_artifact_ns}.json                   이번 실행 결과")
A(f"runs/compile_info_{_artifact_ns}.json              LLM-Wiki 컴파일 감사 정보")
A("```")

# 실행 산출물이므로 다른 run artifact와 같은 runs/ 아래에 둔다.
OUT = os.path.abspath(os.path.join(HERE, "runs", f"REPORT{SFX}.md"))
open(OUT, "w").write("\n".join(L))
print(f"wrote {OUT}")
for n in order:
    print(f"  {LABEL[n]:<22} cover {agg(n, 'cover'):.3f} ({hits(n)}/{N})  F1 {agg(n, 'f1'):.3f}")
