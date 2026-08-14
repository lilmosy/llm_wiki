# llm_wiki 작업 기준

> 최신 상태 기준: **2026-08-15**. `AGENTS.md`는 이 파일을 가리킨다. 과거 8문항/80문단 실행 지침보다 아래를 우선한다.

## 목적과 해석 범위

- 이 프로젝트는 LLM-Wiki 논문의 **간이 재구현**이다. 공식 코드나 논문 수치의 재현물이라고 쓰지 않는다.
- 현재 목표는 방법 순위가 아니라 **build → retrieval/traversal → answer** 중 실패 위치를 trace로 해부하는 것이다.
- 7개 baseline/8문항·80문단은 초기 탐색 snapshot이고, 현재는 `llm_wiki`, `bm25`, `dense`, `graphrag` 네 arm의 실제 trace를 깊게 읽는 후속 단계다. 나머지 arm 구현은 남아 있지만, 현재 slice의 같은 수준 비교 결과가 아니다.

## 데이터와 현재 실행 상태

| dataset | 위치 | corpus | 실행/해석 상태 |
|---|---|---:|---|
| 2Wiki | `data/2wiki/` | 16문항, 156 passage | q1/q2/q3/q7/q12를 네 arm으로 실행. 다섯 문항 모두 trace 분석 가능. |
| MuSiQue | `data/musique/` | 3문항, 60 paragraph | m1만 clean 사례다. **m2·m3은 공식 gold chain이 원문에 grounded되지 않아 성능 순위에서 제외한다**. |

- 실행 원본은 `runs/results_2wiki.json`, `runs/results_musique.json`이고, LLM-Wiki 컴파일 감사 정보는 대응하는 `compile_info_*.json`이다.
- 자동 집계 리포트는 `runs/REPORT_2wiki.md`, `runs/REPORT_musique.md`다. 둘 다 실행 당시의 raw 결과 snapshot이므로, 새 실행 결과를 의도적으로 남길 때 외에는 재생성하거나 편집하지 않는다.
- 구현이 지금 형태에 이른 경위와 과거에 틀렸던 판단은 `devlog.md`에 기록한다. 구조나 기본 설정을 바꾸면 여기에 한 항목 남긴다.
- 이 저장소는 자체 완결이다. 레포 밖 문서를 근거나 필수 참조로 걸지 않는다.

## 파일·실행 규칙

- `config.yaml`은 현재 모델, embedding, top-k의 **단일 기준**이다. 문서나 코드에 다른 모델을 기준으로 쓰지 않는다.
- 데이터·모델·하이퍼파라미터가 바뀌면 인덱스를 다시 빌드한다. `--reuse`는 artifact의 data/config fingerprint가 모두 일치할 때만 안전하다.
- `--questions`는 **질의 문항만** 줄인다. build corpus는 `--data` 아래의 전체 `corpus.jsonl`을 유지한다.
- 사용자 요청 없이 `data/2wiki/`나 `data/musique/`를 다시 생성하거나 덮어쓰지 않는다.
- build 비용과 query 비용을 분리해 기록한다. `results_*.json`의 arm metadata는 build, 각 row의 arm trace는 query다.
- LLM-Wiki artifact는 `indexes_<dataset>/llm_wiki/wiki/`, Dense는 `indexes_<dataset>/dense/`, GraphRAG는 `indexes_<dataset>/graphrag/` 아래에 둔다. dataset namespace를 섞지 않는다.
- 커밋 기준: **LLM이 만든 산출물(wiki page, graph, community report, ErrorBook, REPORT)은 커밋한다.** 재생성이 공짜인 것(`*.npz` 임베딩 벡터, `runs/cache/`, `runs/history/`)은 커밋하지 않는다.
- 경로는 프로젝트 루트를 기준으로 계산한다. `os.path.dirname(__file__)` 옆에 산출물을 두면 파일을 옮길 때 산출물도 조용히 따라가고, 없으면 새로 만들어져 에러 없이 어긋난다(`core/runs/cache` 사례).

```bash
# API 0회 검증
python3 run_all.py --dry-run --data data/2wiki \
  --only llm_wiki,bm25,dense,graphrag \
  --questions q1,q2,q3,q7,q12

# 현재 2Wiki 진단 run
python3 run_all.py --data data/2wiki \
  --only llm_wiki,bm25,dense,graphrag \
  --questions q1,q2,q3,q7,q12

```

## 방법별 trace를 읽는 위치

| arm | build artifact | query trace |
|---|---|---|
| LLM-Wiki | wiki page, `_manifest.json`, `compile_info_*.json`, ErrorBook | `trace`, `read_pages`, `retrieved_pids` |
| BM25 | 영속 artifact 없음 | `retrieved`, `retrieved_pids` |
| Dense | `meta.json`, `INDEX.md`, `vectors.npz`(비커밋, 재빌드 시 생성) | `retrieved`, `retrieved_pids` |
| GraphRAG | `graph.json`, `COMMUNITIES.md`, `_build_meta.json` | `map_trace`, `partials`, `retrieved_pids` |

실패 판정 순서는 고정한다.

1. gold/decomposition이 원문에 grounded되는가? 아니라면 data-grounding 문제이고, 그 문항은 성능 비교에서 제외한다. 확인된 사례는 MuSiQue m2·m3이며 README 해석 경계에 대조를 남겼다.
2. valid gold가 인덱스에 존재하는가? 아니라면 원문/분할 또는 인덱싱 실패다.
3. 존재하지만 질의에서 읽지 않았는가? 검색·ranking·link/community 탐색 실패다.
4. 모두 읽고도 틀렸는가? 추론·비교·최종 답변 실패다.
5. gold를 읽지 않고 맞혔는가? grounded 성공이 아니라 shortcut/parametric-knowledge 가능성으로 표시한다.

## 구현 경계

- LLM-Wiki는 passage를 entity-centric wiki page + wikilink + source PID로 컴파일하고 agent가 search/read를 반복한다.
- GraphRAG는 현재 Leiden이 아니라 `greedy_modularity` community detection을 쓴다.
- embedding은 논문의 Qwen3-Embedding-8B가 아니라 `Qwen/Qwen3-Embedding-0.6B`이다.
- ErrorBook은 compile 중 발견된 오류를 기록·수정/재검증하는 빌드 단계 기능이다. `--reuse` 질의에서는 새 repair를 수행하지 않는다.
