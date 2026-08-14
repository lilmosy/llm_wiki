# LLM-Wiki 재현 MVP — 4개 방법 비교 리포트

> 논문 *Retrieval as Reasoning: Self-Evolving Agent-Native Retrieval via LLM-Wiki* (arXiv 2605.25480) §4.2의
> **이번에 선택한 4개 방법**을 동일 조건에서 **간이 재현(simplified reimplementation)**.
> 코퍼스 5문항 / 156 passage (2WikiMultihopQA slice). 답변 LLM = `claude-sonnet-5` **전 arm 동일**.
> 임베딩 = `Qwen/Qwen3-Embedding-0.6B` (논문은 Qwen3-Embedding-8B — 대체).

> ⚠️ **논문 수치 재현이 아니다.** 논문은 3개 데이터셋 각 500문항, GLM-5.1 백본. 목표는
> 절대 수치 복제가 아니라 **각 방식이 멀티홉에서 어디서 무너지는지의 경향 확인**이다.

## 0. 한눈에 보기 (cover 내림차순)

| arm | paradigm | **cover** | F1 | EM | 질의 LLM 호출/문항 | 질의 시간/문항 |
|---|---|---|---|---|---|---|
| **LLM-Wiki (ours)** | agent-native wiki | **1.000 (5/5)** | 1.000 | 1.000 | 3.6 | 9.6s |
| Vanilla RAG (Dense) | flat dense | 1.000 (5/5) | 1.000 | 1.000 | 1.0 | 2.8s |
| Vanilla RAG (BM25) | flat sparse | 0.800 (4/5) | 0.800 | 0.800 | 1.0 | 2.5s |
| GraphRAG | graph + community reports | 0.600 (3/5) | 0.600 | 0.600 | 100.0 | 201.6s |

- **cover** = 정답이 예측에 포함되는가 (장황한 답변에 강건한 정확도 지표). 주 판정 기준.
- F1/EM = 토큰 overlap / 완전일치. 답변을 최단 스팬으로 강제했으나 여전히 표기 변형에 민감.

## 0b. 근거 회수율 — 정답 근거 문단을 실제로 회수했나

데이터셋의 `supporting_facts`(gold_pids)를 정답 라벨로 쓴 자동 채점.
**정답률과 어긋나는 arm이 진짜 성질을 드러낸다.**

| arm | 평균 근거 회수율 | q1(최고 hop) | 정답률 |
|---|---|---|---|
| LLM-Wiki (ours) | 0.600 | 1.00 | 5/5 |
| Vanilla RAG (Dense) | 0.850 | 0.75 | 5/5 |
| Vanilla RAG (BM25) | 0.800 | 0.50 | 4/5 |
| GraphRAG | 0.200 | 0.00 | 3/5 |

> `–`는 문단 id가 아니라 요약/커뮤니티를 회수하는 arm(GraphRAG·RAPTOR 일부·LLM-Wiki)이라 문단 단위 측정이 정의되지 않는 경우다.

## 1. 오프라인 인덱스 빌드 비용 (논문의 '컴파일 비용' 논점)

정확도만 비교하면 인덱스를 미리 만든 arm의 선불 비용이 안 보인다. 빌드/질의 비용을 분리해 기록한다.

| arm | 오프라인 빌드 | 빌드 LLM 호출 | 빌드 산출물 | 질의 호출/문항 |
|---|---|---|---|---|
| LLM-Wiki (ours) | ✅ 필수 | 524 | `wiki/` (md 트리) pages=485, sources=156, structural_errors=10, passage_coverage=146/156, gold_coverage=14/14 | 3.6 |
| Vanilla RAG (BM25) | ❌ 없음 | 0 | –  | 1.0 |
| Vanilla RAG (Dense) | ⚠️ 임베딩만 | 0 | `indexes/dense/` nodes=156 | 1.0 |
| GraphRAG | ✅ 필수 | 255 | `indexes/graphrag/COMMUNITIES.md` entities=514, relations=471, communities=99, extraction_failures=18 | 100.0 |

- 총 LLM 호출: **1307회** (입력 4,210,368 tok / 출력 554,288 tok)
- 응답 캐시 히트: **152회** (API로 나가지 않음). 위 호출 수는 실제 API 호출만 센 것이므로 콜드 런 비용과 비교 가능하다.
- 빌드는 **1회**, 질의는 **문항마다**. 코퍼스가 커지면 빌드 비용은 상각되고 질의 비용이 지배한다.

## 2. 오염 점검 — Closed-book이 몇 개 맞히나

(closed_book arm 미실행)

## 3. hop 깊이별 정확도 (논문 핵심 주장: hop↑ → 구조화 이득↑)

| hop | LLM-Wiki (ours) | Vanilla RAG (BM25) | Vanilla RAG (Dense) | GraphRAG |
|---|---|---|---|---|
| 2-hop (3문항) | 1.00 | 1.00 | 1.00 | 0.67 |
| 4-hop (2문항) | 1.00 | 0.50 | 1.00 | 0.50 |

## 4. 문항별 정답 매트릭스 (O = cover 성공)

| id | hop | 질문 | 정답 | LLM-Wiki (ours) | Vanilla RAG (BM25) | Vanilla RAG (Dense) | GraphRAG |
|---|---|---|---|---|---|---|---|
| q1 | 4 | Which film has the director who is older, The Gameco | Monster A Go-Go | **O** | X | **O** | X |
| q2 | 2 | Who is the mother of the director of film Polish-Rus | Małgorzata Braunek | **O** | **O** | **O** | X |
| q3 | 2 | Which film came out first, Blind Shaft or The Mask O | The Mask Of Fu Manchu | **O** | **O** | **O** | **O** |
| q7 | 2 | Who is Charles Bretagne Marie De La Trémoille's pate | Charles Armand René de La Trémoille | **O** | **O** | **O** | **O** |
| q12 | 4 | Which film has the director born first, Once A Gentl | Once A Gentleman | **O** | **O** | **O** | **O** |

## 4b. 채점 아티팩트 점검 (오답으로 찍혔지만 실제로는 맞은 것)

`cover`는 정답 문자열이 예측에 **포함**되는지를 본다. 그래서 예측이 정답보다 **짧으면**
(예: 정답 `Karl I, Prince of Anhalt-Zerbst` vs 예측 `Karl I`) 내용이 맞아도 오답으로 찍힌다.
최단 스팬 강제 지시가 과하게 먹은 경우다. 자동 검출:

→ 절단형 근접 오답 없음.

## 5. 4-hop 케이스 상세 (논문 Appendix H Case 1)

**Which film has the director who is older, The Gamecock (Film) or Monster A Go-Go?**  (정답: `Monster A Go-Go`)

| arm | 예측 | cover | 회수한 근거 |
|---|---|---|---|
| LLM-Wiki (ours) | `Monster A Go-Go` | ✅ | ['wiki_search(The Gamecock (film))', 'wiki_search(Monster A Go-Go)', "wiki_read(['Pasquale-Festa-Campanile', ' |
| Vanilla RAG (BM25) | `The Gamecock (film)` | ❌ | ['p002', 'p003', 'p004', 'p001', 'p007'] |
| Vanilla RAG (Dense) | `Monster a Go-Go` | ✅ | ['p002', 'p007', 'p003', 'p004', 'p005'] |
| GraphRAG | `The Gamecock` | ❌ | ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C10', 'C11', 'C12', 'C13', 'C14', 'C15', 'C16',  |

이 문항은 **영화 → 감독 → 생년 → 비교**의 4단계다. 감독의 생년은 영화 문단에 없고 별도 gold 문단(p007, p002, p006, p005)에 있으므로, 한 번의 lexical 검색으로는 재료가 빠질 수 있다.

## 6. 컴파일(LLM-Wiki 인덱스) 상세

- passage별 SELECTPAGES + COMPILEWIKIPAGES 트레이스 156건
- 최종 미해결 오류: 10건 / 코드 자동수정 0건
- 컴파일 중 감지된 오류: **37건**

## 7. MuSiQue slice 해석 한계


> ⚠️ **해석 한계**
> - 문항 5개는 통계적 유의성을 논할 규모가 아니다. arm 간 1문항 차이 = cover 0.200.
> - 코퍼스 156 passage에서 top-5는 전체의 3.2%다. 이 slice는 메커니즘 해부용이지 벤치마크 성능 추정용이 아니다.
> - 이 slice의 질문은 전역 요약이 아니라 국소·정확 멀티홉 질문이라, GraphRAG의 전역 질의 설계 의도가 직접 검증되지는 않는다.
> - 임베딩 모델을 소형으로 대체했다. 전 arm 동일 적용이라 arm 간 비교는 유효하나 절대 성능은 논문보다 낮다.
> - 다음 단계는 유형·hop별 문항을 늘려 같은 분석을 반복하는 것이다.

## 8. 산출물 지도

```
indexes_2wiki/llm_wiki/wiki/              LLM-Wiki 컴파일 결과 (md 트리)
indexes_2wiki/dense/INDEX.md              임베딩 행렬 메타
indexes_2wiki/graphrag/COMMUNITIES.md     커뮤니티 리포트
runs/results_2wiki.json                   이번 2Wiki 실행 원본
runs/compile_info_2wiki.json              LLM-Wiki 컴파일 감사 정보
```
