# LLM-Wiki 재현 MVP

논문 **_Retrieval as Reasoning: Self-Evolving Agent-Native Retrieval via LLM-Wiki_** (arXiv 2605.25480) 의 핵심 파이프라인을 처음부터 재구현한 최소 동작본(MVP)입니다.

> **한 줄 요약** — 문서를 chunk-벡터로 저장하는 대신 **LLM으로 위키 페이지(+링크)로 컴파일**하고, 에이전트가 링크를 따라가며 답을 찾습니다("검색을 조회가 아니라 추론으로"). 멀티홉 질문에서 기존 RAG가 왜 깨지는지, 구조화된 위키가 그걸 얼마나 해결하는지를 작은 코퍼스에서 end-to-end로 확인합니다.

논문에는 공개 코드가 없어(의사코드 Algorithm 1 + §3.2 서술 + Appendix E 위키 스키마만 존재), 이 저장소는 **논문 설계를 근거로 재구현**한 것입니다. 파일명은 논문의 정확한 파일 구조가 아니라 **논문 구성요소에 대응**하도록 지었습니다(아래 대응표).

---

## 목적

- **재현**: (a) 파이프라인 end-to-end 동작, (b) 논문 **Appendix H 트레이스 재현**, (c) 핵심 패턴 재현 — *위키 > flat-RAG*, *hop이 깊을수록 격차↑*.
- **고도화 탐구(진행 중)**: 논문이 스스로 인정한 한계(컴파일 비용 / 확장성 / Single-doc 열세)를 출발점으로 기여 축 탐색. → 상세는 [REPORT.md](REPORT.md).

---

## 한눈에 보기 (결과)

| 지표 | Flat-RAG (BM25) baseline | LLM-Wiki (ours) |
|---|---|---|
| **정확도 cover** (장황함에 강건) | 0.875 (7/8) | **1.000 (8/8)** |
| 평균 F1 (토큰 overlap) | 0.750 | 0.554 |
| 평균 EM | 0.625 | 0.375 |

- **cover 기준 LLM-Wiki가 전 문항 정답(8/8)**, baseline은 **q1(4-hop)에서 오답** — 논문 Appendix H Case 1의 실패를 그대로 재현.
- 코퍼스는 2Wiki 스타일 큐레이션 16 passage / 8문항(**논문 Appendix H 정답 케이스 2건 포함**). 전 비교군 동일 모델 `claude-opus-4-8`(논문 §4.4 통제), 임베딩 없이 BM25.

### ⚠️ 쟁점 — F1/EM vs cover, 무엇으로 판정할 것인가

F1/EM에서는 LLM-Wiki가 **낮게** 보입니다. 이유는 지식 부족이 아니라 **에이전트가 문장형으로 답해 토큰 겹침(F1)이 깎이는 채점(harness) 아티팩트**입니다. 예: 정답 `Monster a Go-Go`를 baseline은 그대로 답하지만, LLM-Wiki는 `"감독이 더 나이 많은 영화는 Monster a Go-Go입니다"`처럼 답합니다(내용은 정답).

이를 없애려면 **양쪽 프롬프트를 terse-span(짧은 정답 스팬)으로 강제하는 교정**이 필요하고, 코드엔 반영돼 있으나 **교정본 재실행 전**입니다. 따라서 위 F1/EM은 교정 전(verbose) 수치이며, 이 세팅에서 **F1/EM과 cover 중 무엇을 정확도 기준으로 삼을지가 열린 논점**입니다. 본 리포트는 장황함에 강건한 **cover를 1차 지표**로 봅니다.

---

## 디렉터리 ↔ 논문 대응

```text
llm_wiki/
├── data/              입력: 코퍼스(passage) + 질문/정답
├── indexing/          [인덱스 타임] 문서 → 위키 컴파일 (질문 전 1회)
├── wiki/              컴파일 산출물 = md 위키 트리 (검색 대상)
├── retrieval/         [쿼리 타임] 에이전트가 위키를 링크 추론
├── baseline/          BM25 flat-RAG 대조군
├── harness/           평가 (F1/EM/cover)
├── run_all.py         전체 파이프라인 실행 엔트리
├── make_report.py     오프라인(API無)으로 REPORT.md 재생성
├── llm.py · wiki.py   Anthropic 래퍼 · 위키 저장/검색 툴
└── config.yaml        논문 하이퍼파라미터
```

| 코드 | 논문 대응 | 역할 |
|---|---|---|
| `data/corpus.jsonl` · `questions.json` | §4 벤치마크 (+Appendix H 케이스) | 입력 코퍼스 · QA(질문+정답) |
| `indexing/compile.py` | **Algorithm 1** (SELECTPAGES + COMPILEWIKIPAGES), §3.1 | 문서 → 위키 컴파일 |
| `indexing/error_book.py` · `validators.py` | **Error Book §3.3** (Layer 1 구조검증·코드 자동수정) | 컴파일 자기교정 |
| `wiki/` (index.md · 카테고리 _index.md · 페이지 · sources/) | **Appendix E** 위키 구조 | 컴파일된 지식 트리 |
| `retrieval/agent.py` (`wiki_search`/`wiki_read`) | **§3.2** Compositional Retrieval (ReAct) | 질의 시 링크 따라 추론 |
| `baseline/bm25_rag.py` | §4 Vanilla RAG baseline | BM25 top-k 대조군 |
| `harness/evaluate.py` | §4 평가 | F1/EM + cover |
| `llm.py` · `wiki.py` · `run_all.py` · `make_report.py` · `config.yaml` | (논문 밖) 실행 인프라 | 오케스트레이션·하이퍼파라미터 |

> 위 절반(`data`~`harness`)이 **논문 재현부**, 아래 인프라 파일은 **돌리기 위한 배관**입니다.

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
- 컴파일은 LLM 비결정성으로 실행마다 페이지 수가 약간 변동(run1 21 / run2 19).
- 다음: 위 F1 교정본 재실행 + 2Wiki 스케일업 + 고도화 기여 축 확정. → [REPORT.md](REPORT.md) 참조.
