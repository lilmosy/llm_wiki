# LLM-Wiki 재현 MVP — 4개 방법 비교 리포트

> 논문 *Retrieval as Reasoning: Self-Evolving Agent-Native Retrieval via LLM-Wiki* (arXiv 2605.25480) §4.2의
> **이번에 선택한 4개 방법**을 동일 조건에서 **간이 재현(simplified reimplementation)**.
> 코퍼스 3문항 / 60 passage (MuSiQue-Ans dev slice). 답변 LLM = `claude-sonnet-5` **전 arm 동일**.
> 임베딩 = `Qwen/Qwen3-Embedding-0.6B` (논문은 Qwen3-Embedding-8B — 대체).

> ⚠️ **논문 수치 재현이 아니다.** 논문은 3개 데이터셋 각 500문항, GLM-5.1 백본. 목표는
> 절대 수치 복제가 아니라 **각 방식이 멀티홉에서 어디서 무너지는지의 경향 확인**이다.

## 0. 한눈에 보기 (cover 내림차순)

| arm | paradigm | **cover** | F1 | EM | 질의 LLM 호출/문항 | 질의 시간/문항 |
|---|---|---|---|---|---|---|
| GraphRAG | graph + community reports | 0.667 (2/3) | 0.667 | 0.667 | 33.0 | 54.2s |
| **LLM-Wiki (ours)** | agent-native wiki | **0.667 (2/3)** | 0.619 | 0.333 | 6.3 | 13.3s |
| Vanilla RAG (Dense) | flat dense | 0.333 (1/3) | 0.333 | 0.333 | 1.0 | 5.5s |
| Vanilla RAG (BM25) | flat sparse | 0.000 (0/3) | 0.190 | 0.000 | 1.0 | 3.7s |

- **cover** = 정답이 예측에 포함되는가 (장황한 답변에 강건한 정확도 지표). 주 판정 기준.
- F1/EM = 토큰 overlap / 완전일치. 답변을 최단 스팬으로 강제했으나 여전히 표기 변형에 민감.

## 0b. 근거 회수율 — 정답 근거 문단을 실제로 회수했나

데이터셋의 `supporting_facts`(gold_pids)를 정답 라벨로 쓴 자동 채점.
**정답률과 어긋나는 arm이 진짜 성질을 드러낸다.**

| arm | 평균 근거 회수율 | m3(최고 hop) | 정답률 |
|---|---|---|---|
| GraphRAG | 0.333 | 0.00 | 2/3 |
| LLM-Wiki (ours) | 0.917 | 0.75 | 2/3 |
| Vanilla RAG (Dense) | 0.556 | 0.50 | 1/3 |
| Vanilla RAG (BM25) | 0.278 | 0.00 | 0/3 |

> `–`는 문단 id가 아니라 요약/커뮤니티를 회수하는 arm(GraphRAG·RAPTOR 일부·LLM-Wiki)이라 문단 단위 측정이 정의되지 않는 경우다.

## 1. 오프라인 인덱스 빌드 비용 (논문의 '컴파일 비용' 논점)

정확도만 비교하면 인덱스를 미리 만든 arm의 선불 비용이 안 보인다. 빌드/질의 비용을 분리해 기록한다.

| arm | 오프라인 빌드 | 빌드 LLM 호출 | 빌드 산출물 | 질의 호출/문항 |
|---|---|---|---|---|
| LLM-Wiki (ours) | ✅ 필수 | 235 | `wiki/` (md 트리) pages=261, sources=60, structural_errors=6, passage_coverage=54/60, gold_coverage=9/9 | 6.3 |
| Vanilla RAG (BM25) | ❌ 없음 | 0 | –  | 1.0 |
| Vanilla RAG (Dense) | ⚠️ 임베딩만 | 0 | `indexes/dense/` nodes=60 | 1.0 |
| GraphRAG | ✅ 필수 | 92 | `indexes/graphrag/COMMUNITIES.md` entities=266, relations=252, communities=32, extraction_failures=22 | 33.0 |

- 총 LLM 호출: **451회** (입력 768,648 tok / 출력 286,896 tok)
- 빌드는 **1회**, 질의는 **문항마다**. 코퍼스가 커지면 빌드 비용은 상각되고 질의 비용이 지배한다.

## 2. 오염 점검 — Closed-book이 몇 개 맞히나

(closed_book arm 미실행)

## 3. hop 깊이별 정확도 (논문 핵심 주장: hop↑ → 구조화 이득↑)

| hop | LLM-Wiki (ours) | Vanilla RAG (BM25) | Vanilla RAG (Dense) | GraphRAG |
|---|---|---|---|---|
| 2-hop (1문항) | 1.00 | 0.00 | 0.00 | 1.00 |
| 3-hop (1문항) | 0.00 | 0.00 | 0.00 | 0.00 |
| 4-hop (1문항) | 1.00 | 0.00 | 1.00 | 1.00 |

## 4. 문항별 정답 매트릭스 (O = cover 성공)

| id | hop | 질문 | 정답 | LLM-Wiki (ours) | Vanilla RAG (BM25) | Vanilla RAG (Dense) | GraphRAG |
|---|---|---|---|---|---|---|---|
| m1 | 2 | Who founded the company that distributed the film UH | Mike Medavoy | **O** | X | X | **O** |
| m2 | 3 | What is the birthplace of the man who does the voice | Denver | X | X | X | X |
| m3 | 4 | What treaty ceded territory to the US extending west | Treaty of Paris | **O** | X | **O** | **O** |

## 4b. 채점 아티팩트 점검 (오답으로 찍혔지만 실제로는 맞은 것)

`cover`는 정답 문자열이 예측에 **포함**되는지를 본다. 그래서 예측이 정답보다 **짧으면**
(예: 정답 `Karl I, Prince of Anhalt-Zerbst` vs 예측 `Karl I`) 내용이 맞아도 오답으로 찍힌다.
최단 스팬 강제 지시가 과하게 먹은 경우다. 자동 검출:

→ 절단형 근접 오답 없음.

## 5. 최고-hop 순차 chain 사례 상세

**What treaty ceded territory to the US extending west to the body of water by the city where the designer of Southeast Library died?**  (정답: `Treaty of Paris`)

| arm | 예측 | cover | 회수한 근거 |
|---|---|---|---|
| LLM-Wiki (ours) | `Treaty of Paris (1783)` | ✅ | ['wiki_search(Southeast Library)', "wiki_read(['Ralph-Rapson'])", 'wiki_search(Ralph Rapson death)', 'wiki_sea |
| Vanilla RAG (BM25) | `Treaty of Guadalupe Hidalgo` | ❌ | ['m047', 'm060', 'm044', 'm041', 'm057'] |
| Vanilla RAG (Dense) | `Treaty of Paris` | ✅ | ['m045', 'm058', 'm049', 'm047', 'm060'] |
| GraphRAG | `Treaty of Paris` | ✅ | ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C10', 'C11', 'C12', 'C13', 'C14', 'C15', 'C16',  |

공식 decomposition의 중간 답 경로는 **Ralph Rapson → Minneapolis → Mississippi River → Treaty of Paris**다. gold paragraph는 `m042, m045, m052, m058`이며, 각 단계의 중간 답이 다음 검색·추론의 입력으로 이어져야 한다.

## 6. 컴파일(LLM-Wiki 인덱스) 상세

- passage별 SELECTPAGES + COMPILEWIKIPAGES 트레이스 60건
- 최종 미해결 오류: 6건 / 코드 자동수정 0건
- 컴파일 중 감지된 오류: **22건**

## 7. MuSiQue slice 해석 한계


> ⚠️ **해석 한계**
> - 문항 3개는 통계적 유의성을 논할 규모가 아니다. arm 간 1문항 차이 = cover 0.333.
> - 코퍼스 60 passage에서 top-5는 전체의 8.3%다. 이 slice는 메커니즘 해부용이지 벤치마크 성능 추정용이 아니다.
> - 이 slice의 질문은 전역 요약이 아니라 국소·정확 멀티홉 질문이라, GraphRAG의 전역 질의 설계 의도가 직접 검증되지는 않는다.
> - 임베딩 모델을 소형으로 대체했다. 전 arm 동일 적용이라 arm 간 비교는 유효하나 절대 성능은 논문보다 낮다.
> - 다음 단계는 유형·hop별 문항을 늘려 같은 분석을 반복하는 것이다.

## 8. 산출물 지도

```
indexes_musique/llm_wiki/wiki/              LLM-Wiki 컴파일 결과 (md 트리)
indexes_musique/dense/INDEX.md              임베딩 행렬 메타
indexes_musique/graphrag/COMMUNITIES.md     커뮤니티 리포트
runs/results_musique.json                   이번 실행 결과
runs/compile_info_musique.json              LLM-Wiki 컴파일 감사 정보
```