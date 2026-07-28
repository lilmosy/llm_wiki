# CLAUDE.md — llm_wiki

논문 *"Retrieval as Reasoning: Self-Evolving Agent-Native Retrieval via LLM-Wiki"* (arXiv 2605.25480) **재구현 MVP**. 목표는 절대 F1 복제가 아니라 (a) end-to-end 동작 (b) 논문 Appendix H 트레이스 재현 (c) 패턴 재현: **위키 > flat-RAG, hop 깊을수록 격차↑**. 서사·결정·고도화 방향은 메모리 `project_llm_wiki.md`와 이 폴더의 `발표정리.md`(⑥ 확장방향) 참조.

## 실행
- `python3 run_all.py` — Phase1 컴파일 → Phase2 질의(LLM-Wiki+BM25) → Phase3 평가. `wiki/`, `error_book.yaml`, `runs/results.json` 생성.
- `python3 make_report.py` — **오프라인**(API 無)으로 `runs/results.json`에서 `REPORT.md` 재생성. cover 지표 포함.
- **`python3`를 써라 (venv 깨짐):** `.venv/bin/python`은 bad interpreter. 패키지(`anthropic` `rank_bm25` `pyyaml`)는 `pip install --user`로 시스템 파이썬에 설치돼 있음.
- API 키: 레포 루트 `../../.env`의 `ANTHROPIC_API_KEY` (llm.py가 로드).

## 하드 제약 (매번 헷갈리는 것)
- ⚠️ **`runs/results.json`은 run1(교정 전 verbose 예측). 덮어쓰지 말 것** — 논문 Appendix H 트레이스 재현본이자 REPORT.md의 근거 원본. 재실행은 LLM 비결정성으로 위키·수치가 바뀌니 반드시 백업 후.
- 오프라인 재생성: `make_report.py`는 API 없이 `results.json`에서 `REPORT.md`만 다시 그림.
- **모델은 전 비교군 `claude-opus-4-8` 동일** (논문 §4.4 통제). 바꾸지 말 것.
- **임베딩 없음 — BM25** (`rank_bm25`). §3.2 "검색 품질은 병목 아님" 근거. baseline·search-fallback 모두 BM25.
- **Error Book MVP 스코프 = 구조 오류 코드-자동수정만** (Layer 1). content 검증(Layer 2)은 log-only.
- 하이퍼파라미터(`config.yaml`): Tmax=15, patience=3, SELECTPAGES k=5.

## 구조
- `llm.py` (opus-4-8 래퍼, .env 로드, USAGE 집계) · `wiki.py` (Wiki 저장/렌더 + `wiki_search`/`wiki_read` 툴)
- `indexing/` = `compile.py`(Algorithm 1 루프) + `validators.py`(구조검증+코드수정) + `error_book.py`
- `retrieval/agent.py` (ReAct 루프) · `baseline/bm25_rag.py` · `harness/evaluate.py` (F1/EM/cover)
- `data/` = `corpus.jsonl`(16 passage, 2Wiki 스타일 큐레이션, **Appendix H 정답 케이스 2건 포함**) + `questions.json`(8문항, hop/type 라벨)
- `wiki/` = 컴파일 산출물(md 트리 + `_manifest.json` 내부 인덱스) · `runs/` 로그 · `REPORT.md`
- 스케일업(실제 2Wiki dev 앞 50개): `data/` 교체만으로 가능(로더 동일).

## 현재 결과 (run1)
cover **8/8** vs baseline 7/8 (baseline은 q1 4-hop 오답 = 논문 Case 1 재현). F1 0.554 < 0.750 = 에이전트 verbose 아티팩트(cover로 정확도 판정). 컴파일 구조오류 0건.

## 작업 관례
- 발표 자료: `발표정리.md`(원본) → `발표정리.html`(그 md를 그대로 렌더하는 스크롤 문서). **md만 고치고 html은 재생성**(내용 항상 일치).
- 파일 새로 만들 때 한글 파일명은 사라진 전례 있음 — 생성 후 `python3 -c "import os;print(os.listdir('.'))"`로 존재 확인.
