# LLM-Wiki 메커니즘 해부 실험

> 상태 기준: **2026-08-15**. 이 저장소는 논문 _Retrieval as Reasoning: Self-Evolving Agent-Native Retrieval via LLM-Wiki_ (arXiv:2605.25480)의 공개되지 않은 구현을, 논문의 알고리즘·스키마 설명을 바탕으로 만든 **간이 재구현**이다. 논문 수치의 재현물이 아니다.

## 무엇을 하는가

**LLM-Wiki**는 검색을 "질문과 비슷한 문단 고르기"가 아니라 **추론 과정**으로 다루는 방법이다. 원문 passage를 미리 **엔티티 단위 wiki page**로 컴파일해 두고(page 안에는 핵심 사실, 다른 page로 가는 wikilink, 출처 PID가 들어간다), 질의 시점에는 에이전트가 page를 읽고 **다음에 어디로 갈지 스스로 정하며** 링크를 따라간다. 컴파일 중 발견된 오류는 ErrorBook에 기록·수정된다.

이 저장소는 그 방법을 논문의 baseline들과 **같은 조건에서 나란히 돌려보고, 실패가 어느 단계에서 나는지 확인**한다.

```text
인덱싱  →  근거 회수  →  추론/답변
   ↑           ↑            ↑
   어디에서 틀렸는지를 trace로 분리해서 본다
```

정답률만 보면 이 셋이 구분되지 않는다. 그래서 **gold 근거를 실제로 읽었는지**를 함께 기록하고, 근거를 읽지 않고 답만 맞힌 경우는 성공으로 세지 않는다.

비교 대상은 논문 §4.2의 baseline이다. `b0` closed-book · `b1` BM25 · `b2` Dense · `b3` RAPTOR · `b4` GraphRAG · `b5` LightRAG · `b6` HippoRAG.

## 현재 상태

목표는 방법의 순위를 확정하는 것이 아니라, 작은 실제 코퍼스에서 실패 위치를 해부하는 것이다.

| 데이터 | 준비된 데이터 | 실제 실행 | 현재 해석 상태 |
|---|---:|---|---|
| **2WikiMultihopQA** | 공식 validation에서 고른 16문항 / 고유 passage 156개 | q1, q2, q3, q7, q12의 진단 실행 | 5문항 모두 성능·근거 trace 분석 가능 |
| **MuSiQue-Ans** | 공식 dev에서 고른 3문항 / paragraph 60개 | m1, m2, m3 실행 | m1만 clean end-to-end 사례. **m2·m3은 공식 gold chain이 원문에 grounded되지 않아** 성능 순위에서 제외 |

이번 실행에서 실제로 비교·해부한 arm은 **LLM-Wiki, BM25, Dense, GraphRAG** 네 개다. 나머지 세 arm은 구현은 있으나 이 slice에서 같은 수준으로 해석한 결과가 아직 없다.

### 관찰된 것

- **LLM-Wiki** — 2Wiki q1·q7·q12와 MuSiQue m1에서 page/link를 따라 gold 근거를 모두 읽는 경로가 실제 trace로 확인됐다.
- **BM25 / Dense** — 일부 2-hop 문항에서는 gold를 한 번에 회수했지만, q1·q12·m1처럼 **뒤 hop의 근거**가 필요한 경우에 누락이 생겼다. 뒤 hop은 질문에 단어가 안 나오므로 질문과의 유사도로는 닿지 않는다.
- **GraphRAG** — q7·m1에서는 필요한 관계가 community report에 남아 있었다. 다만 작은 국소 QA에서 모든 community를 map하는 비용이 크고, extraction·map 실패도 있었다.

⚠️ 2Wiki 5문항 / MuSiQue 3문항의 **탐색 사례**다. 이 수로 방법의 일반 성능을 주장할 수 없다. 자세한 한계는 아래 [해석 경계](#해석-경계).

## 문서

| 문서 | 용도 |
|---|---|
| [runs/REPORT_2wiki.md](runs/REPORT_2wiki.md) | 2Wiki 5문항 실행의 자동 집계표. 정답률·근거 회수율·빌드 비용·hop별 정확도 |
| [runs/REPORT_musique.md](runs/REPORT_musique.md) | MuSiQue 3문항 실행의 **raw 결과 snapshot**. m2·m3의 grounding 해석은 아래 "해석 경계"와 함께 읽는다 |
| [devlog.md](devlog.md) | 이 저장소가 지금 형태에 이른 경위. 무엇을 왜 바꿨고 무엇이 틀렸었는지 |
| `runs/results_*.json` | 문항×arm별 예측, retrieved PID, agent/community trace의 원본 |
| `runs/compile_info_*.json` | LLM-Wiki 컴파일 coverage, validation, repair, final error 감사 로그 |
| `indexes_*/` | 실제 생성된 index artifact. wiki page, community report, ErrorBook을 직접 읽을 수 있다 |

## 현재 디렉터리 구조

폴더 이름이 논문의 어느 단계인지를 그대로 가리킨다.

```text
llm_wiki/
├── README.md · CLAUDE.md · devlog.md
├── AGENTS.md                    →  CLAUDE.md 로의 심볼릭 링크
├── requirements.txt · .env.example
├── config.yaml                     모델·임베딩·top-k 의 단일 기준
├── run_all.py                      실행기: build → query → score
│
├── core/                           모든 arm 이 공유하는 부품 (§4.4 통제)
│   ├── llm.py                      LLM 호출 + 응답 캐시 + 토큰 집계
│   ├── embed.py                    임베딩 + 벡터 캐시 + top_k
│   └── evaluate.py                 채점 (F1 / EM / cover)
│
├── baseline/                       방법 구현 (§4.2)
│   ├── _common.py                  전 arm 공통 답변 프롬프트
│   ├── b0_closed_book/answer.py    문맥 없이 답 (오염 점검)
│   ├── b1_bm25/answer.py           어휘 검색
│   ├── b2_dense/answer.py          벡터 검색
│   ├── b3_raptor/                  요약 트리          build.py + answer.py
│   ├── b4_graphrag/                엔티티 그래프·커뮤니티
│   ├── b5_lightrag/                이중 검색 (엔티티 + 관계)
│   ├── b6_hipporag/                Personalized PageRank
│   └── llm_wiki/                   제안 방법 (§3)
│       ├── select_pages.py         SELECTPAGES
│       ├── build.py                COMPILEWIKIPAGES (Algorithm 1)
│       ├── wiki.py                 페이지 저장·wikilink·source 관리
│       ├── validators.py           구조/grounding 검증
│       ├── error_book.py           ErrorBook — 오류 기록·수정·재검증
│       ├── tools.py                에이전트 도구 (search / read)
│       └── answer.py               ReAct 순회 (T_max, patience)
│
├── analysis/make_report.py         results_*.json → runs/REPORT_*.md (API 0회)
├── tools/                          공식 데이터셋에서 slice 추출
│   ├── fetch_2wiki.py
│   └── fetch_musique.py
│
├── data/                           입력
│   ├── 2wiki/{corpus.jsonl, questions.json}       156 passage / 16문항
│   └── musique/{corpus.jsonl, questions.json}      60 para / 3문항
│
├── indexes_2wiki/                  빌드 산출물 — LLM 이 만든 것
│   ├── llm_wiki/{wiki/, error_book.yaml, _build_meta.json}
│   ├── graphrag/{graph.json, COMMUNITIES.md, _build_meta.json}
│   ├── dense/{meta.json, INDEX.md, _build_meta.json}
│   │   └── vectors.npz          ✕  질의 시 자동 생성
│   ├── raptor/  ← 미실행         ✕  vectors.npz
│   ├── lightrag/ ← 미실행        ✕  ent_vectors.npz, rel_vectors.npz
│   └── hipporag/ ← 미실행        ✕  vectors.npz
├── indexes_musique/                동일 구조
│
└── runs/                           실행 기록
    ├── results_<dataset>.json      문항×arm 예측·retrieved PID·trace
    ├── compile_info_<dataset>.json LLM-Wiki 컴파일 감사 로그
    ├── REPORT_<dataset>.md         자동 집계표
    ├── cache/ ✕                    LLM 응답 캐시
    └── history/ ✕                  과거 실행 snapshot
```

`✕` 는 저장소에 없는 것이다. 기준은 **LLM 이 만든 산출물은 커밋하고, 재생성이 공짜인 것은 커밋하지 않는다**.

`*.npz` 는 임베딩 벡터다. **clone 직후에는 없고, 따로 받아올 필요도 없다** — 해당 arm 을 처음 실행할 때 로컬 임베딩 모델이 API 비용 없이 만들어 위 위치에 캐시한다(`core/embed.py` 의 `encode_cached`). 바이너리라 diff 가 남지 않고 재빌드마다 통째로 히스토리에 쌓이기 때문에 제외했다. 벡터는 `(문서 텍스트, 임베딩 모델)` 만의 함수라 둘 중 하나가 바뀌면 자동으로 다시 만든다.

`runs/cache/` 는 LLM 응답 캐시로, 재실행 비용을 줄일 뿐 결과의 근거가 아니다. `runs/history/` 는 과거 실행 snapshot이고 그 역할은 git 이 대신한다.

`b3_raptor`·`b5_lightrag`·`b6_hipporag` 는 구현은 있으나 현재 slice 에서는 실행하지 않았다. `b0`~`b2` 에 `build.py` 가 없는 것은 오프라인 인덱스가 필요 없는 방법이기 때문이다.

### LLM-Wiki의 실제 산출물

LLM-Wiki는 원문 passage를 source archive와 entity-centric wiki page로 컴파일한다. wiki page에는 `Key Facts`, `Related Pages`의 wikilink, source PID가 남는다.

- 2Wiki: 156 source passage에서 485 page 생성, 146 passage가 page로 반영. 현재 5문항의 gold 14/14는 coverage에 포함.
- MuSiQue: 60 source paragraph에서 261 page 생성, 54 paragraph가 page로 반영. gold 9/9는 coverage에 포함.

빈 생성·구조/grounding 검증·repair 기록은 `runs/compile_info_<dataset>.json`과 `indexes_<dataset>/llm_wiki/error_book.yaml`에 남는다.

## 실행

`config.yaml`이 모델·embedding·top-k의 단일 기준이다. 데이터·config를 바꿨다면 기존 index를 재사용하지 않는다.

```bash
pip install -r requirements.txt
cp .env.example .env        # ANTHROPIC_API_KEY 를 채운다
```

임베딩은 API가 아니라 로컬 모델이므로 키가 필요 없다. 다만 첫 실행 때 모델 약 1.2GB를 내려받는다.

```bash
# API 호출 없이 배선/기존 artifact 확인
python3 run_all.py --dry-run --data data/2wiki \
  --only llm_wiki,bm25,dense,graphrag \
  --questions q1,q2,q3,q7,q12

# 현재 2Wiki 진단 slice를 새로 build하고 질의
python3 run_all.py --data data/2wiki \
  --only llm_wiki,bm25,dense,graphrag \
  --questions q1,q2,q3,q7,q12

# data/config fingerprint가 같은 기존 index만 재사용하여 질의
python3 run_all.py --reuse --data data/2wiki \
  --only llm_wiki,bm25,dense,graphrag \
  --questions q1,q2,q3,q7,q12

```

`--questions`는 **질의 문항만** 제한한다. 인덱싱 corpus는 해당 dataset directory의 전체 corpus를 유지한다.

현재 `runs/REPORT_*.md`는 이미 실행한 결과의 snapshot이므로, 새로운 실행 결과를 의도적으로 남길 때가 아니라면 재생성하지 않는다.

## 채점

세 지표를 함께 기록한다. 정규화(소문자화·문장부호 제거·관사 제거)는 SQuAD 공식 eval 방식이고, HotpotQA·2Wiki·MuSiQue 공식 채점기가 공유하는 표준이다.

| 지표 | 정의 | 성격 |
|---|---|---|
| **EM** | 정규화 후 완전 일치 | 가장 엄격 |
| **F1** | 단어 겹침의 조화평균 | 부분 정답에 점수를 준다 |
| **cover** | gold가 예측 안에 포함되는가 | 장황한 답이 손해 보지 않는다 |

`cover`는 논문에 없는 **이 저장소의 추가 지표**다. 에이전트 계열은 문장으로 답하는 경향이 있어, `gold "1928"` / `pred "The film was released in 1928."` 같은 경우 F1이 0.33까지 떨어진다. 대신 `cover`는 반대 방향으로 관대하므로(길수록 우연히 포함될 수 있다) 단독으로 성공 판정에 쓰지 않고, 항상 근거 회수율과 함께 읽는다.

## 해석 경계

- 이 저장소의 구현은 논문의 공식 코드가 아니라 **간이 재구현**이다. 논문 수치의 재현물이 아니다.
- 현재 2Wiki 5문항, MuSiQue 3문항의 탐색 사례뿐이므로 **방법의 일반 성능 순위를 주장할 수 없다.** arm 간 1문항 차이가 2Wiki에서 0.2, MuSiQue에서 0.333이다.
- MuSiQue **m2·m3은 공식 gold chain이 원문에 grounded되지 않는다.** MuSiQue가 chain을 distant supervision으로 구성해서, 정답 문자열은 gold 문단에 있지만 decomposition이 주장하는 **관계**는 그 문단에 없다.

  | | 공식 decomposition | gold 문단이 실제로 말하는 것 |
  |---|---|---|
  | m2 step3 | `Trey Parker >> place of birth → Denver` (`m030`) | `m030`은 **Dian Bachar**가 Denver 출생이라고 말한다. Trey Parker는 친구로만 언급되고, 그의 출생지는 코퍼스에 없다 |
  | m3 step2 | `Ralph Rapson >> place of death → Minneapolis` (`m042`) | `m042`는 Rapson이 **설계한 건물**이 Minneapolis에 있다고 말한다. 사망 언급은 코퍼스에 없다 |

  근거를 다 읽어도 맞힐 수 없는 문항이므로 **성능 순위에서 제외**하고, 여기서의 정답은 shortcut이나 parametric knowledge로 본다. m1은 이 문제가 없는 clean 사례다.
- GraphRAG는 논문의 Leiden 대신 `greedy_modularity` community detection을 쓴다.
- embedding은 논문의 `Qwen3-Embedding-8B` 대신 `Qwen/Qwen3-Embedding-0.6B`를 쓴다.
- 결과를 읽을 때는 항상 **정답 → 근거 회수 → build/trace** 순서로 확인한다.
