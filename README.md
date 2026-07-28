# LLM-Wiki 재현 MVP

논문 **_Retrieval as Reasoning: Self-Evolving Agent-Native Retrieval via LLM-Wiki_** (arXiv 2605.25480) 의 핵심 파이프라인을 처음부터 재구현한 최소 동작본(MVP)입니다.

> **한 줄 요약** — 문서를 chunk-벡터로 저장하는 대신 **LLM으로 위키 페이지(+링크)로 컴파일**하고, 에이전트가 링크를 따라가며 답을 찾습니다("검색을 조회가 아니라 추론으로"). 멀티홉 질문에서 기존 RAG가 왜 깨지는지, 구조화된 위키가 그걸 얼마나 해결하는지를 작은 코퍼스에서 end-to-end로 확인합니다.

논문에는 공개 코드가 없어(의사코드 Algorithm 1 + §3.2 서술 + Appendix E 위키 스키마만 존재), 이 저장소는 **논문 설계를 근거로 재구현**한 것입니다. 파일명은 논문의 정확한 파일 구조가 아니라 **논문 구성요소에 대응**하도록 지었습니다(아래 디렉터리 구조에 줄별 주석).

---

## 목적

- **재현**: (a) 파이프라인 end-to-end 동작, (b) 논문 **Appendix H 트레이스 재현**, (c) 핵심 패턴 재현 — *위키 > flat-RAG*, *hop이 깊을수록 격차↑*.
- **고도화 탐구(진행 중)**: 논문이 스스로 인정한 한계(컴파일 비용 / 확장성 / Single-doc 열세)를 출발점으로 기여 축 탐색. → 상세는 [REPORT.md](REPORT.md).

---

## 한눈에 보기 (결과)

| 지표 | Flat-RAG (BM25) baseline | LLM-Wiki (ours) |
|---|---|---|
| **정확도 cover** (장황함에 강건) | 0.875 (7/8) | **1.000 (8/8)** |
| 평균 F1 (토큰 overlap) | 0.875 | **1.000** |
| 평균 EM | 0.875 | **1.000** |

- **세 지표 모두 LLM-Wiki 우위.** LLM-Wiki는 전 문항 정답(8/8), baseline은 **q1(4-hop)에서 오답** — 논문 Appendix H Case 1의 실패를 그대로 재현.
- 코퍼스는 2Wiki 스타일 큐레이션 16 passage / 8문항(**논문 Appendix H 정답 케이스 2건 포함**). 전 비교군 동일 모델 `claude-opus-4-8`(논문 §4.4 통제), 임베딩 없이 BM25.

### F1/EM에 대한 메모 (terse 교정)

초기 실행에서는 F1/EM만 보면 LLM-Wiki가 낮게 보였는데, 지식 부족이 아니라 **에이전트가 문장형으로 답해 토큰 겹침이 깎이는 채점(harness) 아티팩트**였습니다(예: `Monster a Go-Go` 대신 `"감독이 더 나이 많은 영화는 Monster a Go-Go입니다"`). **양쪽 답변을 terse-span(짧은 정답 스팬)으로 강제하는 교정**을 적용해 재실행한 결과, 아티팩트가 사라지고 위 표처럼 **F1·EM도 위키 우위**가 됐습니다. 정확도 1차 지표는 여전히 장황함에 강건한 **cover**로 보되, 교정 후 세 지표가 같은 방향을 가리킵니다.

---

## 디렉터리 구조 (논문 대응 · 주석)

> 논문엔 공개 코드가 없어 파일 배치는 재구현 선택입니다. 각 줄에 **논문 대응**과 **역할**을 달고, LLM이 자동 생성하는 부분은 `(LLM 생성)`으로 표시했습니다. 파이프라인 단계는 ①입력 → ②컴파일 → ③위키 → ④질의 → ⑤채점 순.

```text
llm_wiki/
│   ── 실행 인프라 (논문 밖 배관) ──
├── run_all.py            전체 엔트리: Phase1 컴파일 → Phase2 질의 → Phase3 평가
├── make_report.py        오프라인(API無): runs/results.json → REPORT.md 재생성
├── llm.py                Anthropic(claude-opus-4-8) 래퍼 · .env 로드 · 토큰 집계
├── wiki.py               Wiki 저장/렌더 + search/read 구현 (툴 래퍼는 retrieval/tools.py)
├── config.yaml           논문 하이퍼파라미터: Tmax=15 · patience=3 · SELECTPAGES k=5
│
│   ── ① 입력 ──
├── data/
│   ├── corpus.jsonl      16 passage 원문 (2Wiki 스타일, 논문 Appendix H 케이스 2건 포함)
│   └── questions.json    8문항 = 질문+정답+hop/type 라벨   ← 채점 기준(정답이 주어짐)
│
│   ── ② 인덱스 타임: 문서 → 위키 컴파일 (질문 전 1회) ──
├── indexing/
│   ├── select_pages.py   SELECTPAGES (Algorithm 1 ①): 갱신할 기존 페이지 고르기
│   ├── compile.py        COMPILEWIKIPAGES (Algorithm 1 ②) + 컴파일 루프 진행자
│   ├── validators.py     Error Book Layer 1: 깨진 링크 등 구조오류를 코드로 자동수정  ← §3.3
│   └── error_book.py     오류→규칙화→다음 컴파일 프롬프트에 주입 (자기교정 장부)      ← §3.3
│
│   ── ③ 컴파일 산출물 = 검색 대상 위키 (전부 LLM 생성) ──   ← Appendix E 위키 구조
├── wiki/
│   ├── index.md          전체 목차                                         (LLM 생성)
│   ├── _manifest.json    페이지 메타 인덱스(제목·별칭·태그·facts) = wiki_search가 뒤짐  (LLM 생성)
│   │
│   │   ▸ 아래 카테고리 6개도 LLM이 스스로 분류 (people/media/… 하드코딩 아님)
│   ├── people/
│   │   ├── _index.md         카테고리 목차: 페이지 목록 + 별칭·태그          (LLM 생성)
│   │   ├── John-V-Prince-of-Anhalt-Zerbst.md
│   │   │       └ 원문을 재조직: YAML(type·aliases·tags) + 한 줄 요약
│   │   │         + ## Key Facts + ## Related Pages([[링크]]=hop 다리) + ## Related Sources
│   │   └── …  Karl-I · Ernest-I · Margarete …  (인물 10페이지)
│   ├── media/           영화 6페이지 (Monster-a-Go-Go · The-Gamecock-film …)
│   ├── geography/       Anhalt-Zerbst · Chicago · Melfi …
│   ├── organizations/   House-of-Ascania …
│   ├── concepts/        Godfather-of-Gore …
│   └── sources/         원문 보관(provenance) — 위키가 아니라 근거 원본
│       ├── digests/     p01–p16.md  passage별 요약 (빠른 대조용)            (LLM 생성)
│       └── articles/    p01–p16.md  passage 원문 전체 (최종 근거)
│
│   ── ④ 쿼리 타임: 질문마다 위키를 링크 추론 ──
├── retrieval/
│   └── agent.py         ReAct 루프(§3.2): wiki_search→wiki_read→[[링크]] 따라가기
│                          →충분성 판단→답. 툴 스키마는 여기, 실제 검색·읽기는 wiki.py
│
│   ── ⑤ 대조군 + 채점 ──
├── baseline/
│   └── bm25_rag.py      Vanilla RAG (BM25 top-k, 위키 없음)  ← §4 baseline
├── harness/
│   └── evaluate.py      각 답 ↔ 정답 대조 → F1 · EM · cover
│
│   ── 산출물 ──
├── error_book.yaml      컴파일 중 잡힌 오류 장부                            (생성물)
└── runs/results.json    문항별 예측·트레이스·점수 원본 (run1)               (생성물)
```

---

## 실행

```bash
# 환경변수로 API 키 지정 (또는 llm.py가 레포 루트 .env에서 로드)
export ANTHROPIC_API_KEY=...

python3 run_all.py      # 컴파일 → 질의(LLM-Wiki + BM25) → 평가. wiki/, runs/results.json 생성
python3 make_report.py  # 오프라인: runs/results.json → REPORT.md 재생성
```

- 의존성: `anthropic`, `rank_bm25`, `pyyaml`
- 하이퍼파라미터(`config.yaml`): `Tmax=15`, `patience=3`, `SELECTPAGES k=5`
- 스케일업(실제 2Wiki dev 앞 50개): `data/` 파일 교체만으로 가능(로더 동일)

---

## 한계 / 다음 단계

- 코퍼스 8문항 **소규모** — 목표는 절대 F1 복제가 아니라 파이프라인 동작 + 트레이스/패턴 재현.
- 컴파일은 LLM 비결정성으로 실행마다 페이지 수가 약간 변동.
- 다음: 2Wiki dev 앞 50개로 스케일업 + 고도화 기여 축 확정. → [REPORT.md](REPORT.md) 참조.
