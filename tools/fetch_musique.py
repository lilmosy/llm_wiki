"""Create the 3-question MuSiQue teaching slice and append its reading guide.

Input is the official MuSiQue-Ans dev JSONL downloaded from the authors' release.
The selected rows are deliberately one sequential chain at each depth (2, 3,
and 4 hop), not a performance sample. The runner indexes their pooled official
paragraphs exactly as it does `data/2wiki/`.

    python3 tools/fetch_musique.py \
      --source /path/to/musique_ans_v1.0_dev.jsonl
"""
import argparse
import html
import json
import os
from collections import OrderedDict


HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_SOURCE = "/private/tmp/musique_raw/data/musique_ans_v1.0_dev.jsonl"
OUT = os.path.join(HERE, "data", "musique")
MISSION = os.path.join(HERE, "1st_mission.md")
START = "<!-- MUSIQUE-SECTION:START -->"
END = "<!-- MUSIQUE-SECTION:END -->"

# One clear serial chain at each depth. IDs are official MuSiQue-Ans dev IDs.
SELECTED = OrderedDict([
    ("m1", "2hop__252311_366220"),
    ("m2", "3hop1__454441_55349_651302"),
    ("m3", "4hop1__94201_642284_131926_13165"),
])

OBSERVATION = {
    "m1": "`UHF`라는 표면 단어가 위성·방송 등 distractor에도 나타난다. 첫 gold에서 `Orion Pictures`를 얻은 뒤, 그 정확한 이름으로 두 번째 gold를 찾는지 본다.",
    "m2": "`The Hobbit`, `Stan`, `South Park`처럼 널리 쓰이는 표면어가 섞인다. 다만 공식 step 3의 입력 `#2`는 Trey Parker인데 gold paragraph는 Dian Bachar의 Denver 출생을 말한다. 따라서 이 문항은 chain 추적과 함께 **gold/decomposition grounding 점검**도 필요하다.",
    "m3": "질문이 길고 treaty·territory·river 관련 distractor가 많다. 다만 공식 step 2는 ‘Ralph Rapson의 사망지’인데 표시된 gold paragraph는 Rapson이 설계한 건물이 Minneapolis에 있다는 사실을 말한다. 이 문항도 깨끗한 chain 사례라기보다 **grounding audit** 사례로 함께 읽는다.",
}


def words(text):
    return len(text.split())


def load_rows(path):
    wanted = set(SELECTED.values())
    found = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            if row["id"] in wanted:
                found[row["id"]] = row
    missing = wanted - set(found)
    if missing:
        raise SystemExit(f"selected official MuSiQue rows missing: {sorted(missing)}")
    return [(local_id, found[src_id]) for local_id, src_id in SELECTED.items()]


def build_slice(rows):
    # MuSiQue can contain multiple paragraphs with the same Wikipedia title.
    # Deduplicate by (title, paragraph text), never by title alone.
    key_to_pid, corpus = {}, []
    for _, row in rows:
        for para in row["paragraphs"]:
            key = (para["title"], para["paragraph_text"])
            if key in key_to_pid:
                continue
            pid = f"m{len(corpus) + 1:03d}"
            key_to_pid[key] = pid
            corpus.append({
                "pid": pid,
                "title": para["title"],
                "text": para["paragraph_text"],
                # Official MuSiQue supervision is paragraph-level, unlike
                # 2Wiki's sentence-level supporting_facts.
                "sentences": [para["paragraph_text"]],
            })

    questions = []
    details = []
    for local_id, row in rows:
        pids = [key_to_pid[(p["title"], p["paragraph_text"])] for p in row["paragraphs"]]
        gold = [pid for pid, p in zip(pids, row["paragraphs"]) if p["is_supporting"]]
        shape = row["id"].split("__", 1)[0]
        question = {
            "id": local_id,
            "question": row["question"],
            "answer": row["answer"],
            "answer_aliases": row.get("answer_aliases", []),
            "hop": len(row["question_decomposition"]),
            "type": "sequential_chain",
            "musique_shape": shape,
            "context_pids": pids,
            "gold_pids": gold,
            "distractor_pids": [pid for pid in pids if pid not in set(gold)],
            "gold_supporting_facts": [
                {"pid": pids[d["paragraph_support_idx"]],
                 "title": row["paragraphs"][d["paragraph_support_idx"]]["title"],
                 "paragraph_idx": d["paragraph_support_idx"]}
                for d in row["question_decomposition"]
            ],
            "question_decomposition": row["question_decomposition"],
            "src_id": row["id"],
            "note": "Official MuSiQue-Ans dev; 2/3/4-hop sequential-chain teaching slice",
        }
        questions.append(question)
        details.append((question, row, pids))
    return corpus, questions, details


def details_block(question, row, pids):
    intermediate = " → ".join(step["answer"] for step in question["question_decomposition"][:-1])
    lines = [f'<a id="{question["id"]}"></a>', "",
             f'#### ✅ {question["id"]} — {question["question"]}', "",
             f'- **정답:** {question["answer"]}',
             f'- **유형 / hop:** MuSiQue sequential chain / {question["hop"]}',
             f'- **중간 entity:** {intermediate}',
             f'- **이번 문항의 관찰점:** {OBSERVATION[question["id"]]}',
             '- **Gold reasoning chain (공식 question decomposition):**']
    for i, step in enumerate(question["question_decomposition"], 1):
        prior = f' → 중간 답 **{step["answer"]}**' if i < question["hop"] else f' → **최종 답 {step["answer"]}**'
        lines.append(f'  {i}. `{step["question"]}`{prior}')
        fact = question["gold_supporting_facts"][i - 1]
        lines.append(f'     - 공식 gold candidate: ★ `{fact["pid"]}` — **{fact["title"]}**')
    lines += ["", "**공식 candidate 20개 — 클릭해서 원문 보기**", ""]
    gold = set(question["gold_pids"])
    for i, (pid, para) in enumerate(zip(pids, row["paragraphs"]), 1):
        is_gold = pid in gold
        tag = "★ gold" if is_gold else "distractor"
        text = html.escape(para["paragraph_text"])
        if is_gold:
            text = f"<mark>{text}</mark>"
        lines += [f'<details>',
                  f'<summary>{i}. {tag} — <code>{pid}</code> — {html.escape(para["title"])} ({words(para["paragraph_text"])} words)</summary>',
                  "",
                  f'### {pid} — {html.escape(para["title"])}', "", text, "", "</details>", ""]
    return "\n".join(lines)


def mission_section(details):
    rows = []
    for q, _, _ in details:
        chain = " → ".join(d["answer"] for d in q["question_decomposition"])
        question = q["question"].replace("|", "\\|")
        rows.append(f'| [{q["id"]}](#{q["id"]}) | {question} | Sequential chain / {q["hop"]}-hop | {chain} |')
    body = [START, "", "# 7. MuSiQue란 무엇인가", "",
            "> 목표: 2Wiki의 **유형별 메커니즘 비교**와 달리, MuSiQue에서는 순차 chain에서 **어느 단계가 빠졌는지**를 읽는다.", "",
            "MuSiQue-Ans는 answerable 질문만 담은 공식 split이다. 한 질문은 여러 single-hop 질문을 조합해 만들며, `question_decomposition`은 각 sub-question·중간 답·근거 paragraph 위치를 제공한다. 따라서 ‘답을 맞혔는가’뿐 아니라, **어느 단계에서 chain이 끊겼는지**를 확인할 수 있다.", "",
            "- 공식 저장소: <https://github.com/StonyBrookNLP/musique>",
            "- 원 논문: <https://direct.mit.edu/tacl/article/doi/10.1162/tacl_a_00475/110996/MuSiQue-Multihop-Questions-via-Single-hop-Question>", "",
            "이 3문항은 공식 MuSiQue-Ans dev에서 고른 **교육용 slice**다. `2hop`·`3hop1`·`4hop1`에서 하나씩 골라, 2·3·4-hop 순차 chain을 차례로 읽도록 했다. 전체 데이터셋 성능을 대표하려는 표본은 아니다. 즉, 지금은 ‘더 긴 chain에서 어느 단계가 실패하는가’를 보기에는 충분하지만, MuSiQue의 모든 생성 형태를 대표하지는 않는다.", "",
            "MuSiQue dev에는 `2hop`, `3hop1`, `3hop2`, `4hop1`, `4hop2`, `4hop3`가 있다. 앞 숫자는 hop 수, 뒤 숫자는 질문을 시작하는 **entity/branch 수**다. 즉 `3hop1`은 한 entity에서 출발해 세 사실을 잇는 bridge이고, `3hop2`는 두 entity/branch의 답을 조합해야 하는 composition이다. 이것은 2Wiki의 네 논리 유형과 일대일 대응하지 않지만, 실제 추론 그래프의 모양을 알려 준다.", "",
            "# 8. MuSiQue 현재 데이터는 어떻게 생겼나", "",
            "```text", "공식 MuSiQue 한 문항", "question + answer + question_decomposition + paragraphs 20개 + is_supporting", "", "현재 프로젝트", "data/musique/corpus.jsonl       3문항의 공식 candidate 60개 paragraph를 공통 corpus로 저장", "data/musique/questions.json     공식 후보 순서·gold/distractor·decomposition을 pid로 기록", "```", "",
            "이 slice에서는 문항끼리 겹치는 paragraph가 없어, 공통 corpus도 정확히 `3문항 × 20 candidate = 60 paragraph`다. 문항을 읽을 때는 각 질문의 공식 candidate 20개를 보고, RAG를 실행할 때는 60개 전체를 공통 인덱스로 사용한다.", "",
            "원문 내용·candidate 순서·gold/distractor·decomposition·정답은 바꾸지 않았다. 여러 RAG가 같은 자료를 인덱싱하도록, 공식 문항 안에 중첩되어 있던 paragraph를 `corpus.jsonl`로 평탄화하고 질문 파일에서 `pid`로 다시 연결한 것뿐이다.", "",
            "| MuSiQue 공식 필드 | 현재 필드 | 뜻 |", "|---|---|---|", "| `paragraphs` | `corpus.jsonl` + `context_pids` | 공식 후보 20 paragraph와 그 원래 순서 |", "| `is_supporting` | `gold_pids`, `distractor_pids` | paragraph-level gold 근거 라벨 |", "| `question_decomposition` | 동일 | sub-question, 중간 답, gold paragraph 위치 |", "| `answer_aliases` | 동일 | 답 표기 변형 평가용 별칭 |", "",
            "> ★ gold는 공식 `is_supporting=true` paragraph다. MuSiQue 공개 형식은 gold **문장** 위치를 주지 않으므로, 아래 원문에서는 gold paragraph 전체를 표시한다.", "",
            "# 9. MuSiQue 공식 질문 구조: hop 기반 순차 chain", "",
            "| 공식 ID 앞부분 | 추론 그래프 | 현재 사용 여부 |", "|---|---|---|", "| `2hop` (= 2hop1) | 2-hop / 1 entity: 한 갈래 bridge `A → B → 답` | m1 |", "| `3hop1` | 3-hop / 1 entity: 한 갈래 bridge `A → B → C → 답` | m2 |", "| `3hop2` | 3-hop / 2 entities: 두 갈래 정보를 마지막 질문에서 조합 | 미포함 |", "| `4hop1` | 4-hop / 1 entity: 더 긴 한 갈래 bridge | m3 |", "| `4hop2` | 4-hop / 2 entities: bridge와 composition이 결합 | 미포함 |", "| `4hop3` | 4-hop / 3 entities: composition 뒤에 bridge가 이어지는 결합 구조 | 미포함 |", "",
            "따라서 이번 세 문항은 **한 갈래 bridge 길이만 2→3→4로 늘리는 실습**이다. 여러 branch를 조합하는 `3hop2`·`4hop2`·`4hop3`는 아직 다루지 않았다. 2Wiki bridge-comparison의 병렬 비교와 MuSiQue composition을 나란히 볼 때 추가하면 좋다.", "",
            "### 현재 3문항을 고른 이유", "",
            "| 문항 | 선택 이유 | 이 문항을 통해 보려는 것 |", "|---|---|---|", "| m1 (`2hop`) | 가장 짧고 원문 grounding이 명확한 기본 chain | 첫 gold에서 얻은 `Orion Pictures`를 다음 검색·추론의 입력으로 쓰는가 |", "| m2 (`3hop1`) | 3-hop 길이를 추가한 순차 사례 | chain은 `The Hobbit → South Park → Trey Parker → Denver`이지만, 공식 gold/decomposition의 step 3 grounding이 어긋나는지까지 점검 |", "| m3 (`4hop1`) | 가장 긴 4-hop 순차 사례 | 긴 chain의 누적 실패와 함께, step 2의 official grounding이 원문에 직접 있는지 점검 |", "",
            "> 따라서 m1은 **깨끗한 기본 chain**, m2·m3는 **순차 chain + 데이터 grounding audit** 사례다. `grounding audit`은 공식 `question_decomposition`의 각 step·`paragraph_support_idx`·실제 gold paragraph가 같은 사실을 가리키는지 대조하는 작업이다. 현재 m2·m3에서 보인 불일치는 이 두 공식 라벨과 원문 사이의 **문항 단위 정합성 문제**이며, MuSiQue 전체가 틀렸다는 뜻은 아니다. 순수한 hop 성능 비교를 하려면 원문과 decomposition이 모두 명확히 맞는 문항을 별도로 골라야 한다.", "",
            "| 문항 | decomposition이 요구하는 사실 | 실제 연결된 gold candidate가 말하는 사실 | 결론 |", "|---|---|---|---|", "| m2 step 3 | `Trey Parker의 출생지 → Denver` | `m030`은 **Dian Bachar**가 Denver에서 태어났다고 말하고 Trey Parker는 친구로만 언급 | Dian Bachar가 Stan의 성우라는 뜻이 아니다. 이전 단계의 Trey Parker가 Dian Bachar로 바뀌어 chain이 끊긴다. |", "| m3 step 2 | `Ralph Rapson의 사망지 → Minneapolis` | `m042`는 Rapson이 설계한 건물이 Minneapolis에 있다고 말함 | ‘사망지’ 관계를 직접 뒷받침하지 않는다. Minneapolis라는 값만 우연히 연결된다. |", "",
            "### `>>`와 `#1` 표기 읽는 법", "",
            "MuSiQue decomposition은 triple 표기가 아니라 **자연어 sub-question template**이다. `X >> 관계`는 ‘X에 대해 그 관계를 묻는다’는 뜻이고, 화살표 오른쪽 답이 다음 단계 입력이 된다. `#1`, `#2`는 각각 바로 앞 첫째·둘째 sub-question의 답을 대입하라는 참조다.", "",
            "```text", "The Hobbit >> part of the series       = The Hobbit은 어느 series의 일부인가? → South Park", "who does the voice of Stan on #1       = #1(South Park)에서 Stan의 목소리는 누구인가? → Trey Parker", "#2 >> place of birth                   = #2(Trey Parker)의 출생지는 어디인가? → Denver", "```", "",
            "즉 2Wiki의 `주체 — 관계 → 객체`와 정보 흐름은 같다. 다만 MuSiQue는 관계를 정규화된 한 단어가 아니라 자연어 질문으로 적고, `#n`으로 이전 객체를 다음 단계 주체 자리에 넣는다.", "",
            "# 10. MuSiQue 3문항 한눈에 보기", "",
            "| 문항 | 질문 | 유형 / hop | gold reasoning chain의 중간 답 |", "|---|---|---|---|", *rows, "",
            "현재 세 문항은 모두 순차 chain이므로 **중간 entity**를 본다. 비교형 질문이 아니라서 2Wiki처럼 ‘비교값’은 없다. 각 문항의 관찰점은 첫 검색어의 모호성, 다음 단계로 넘길 entity, 관련 있어 보이는 distractor를 미리 표시한 것이다.", "",
            "# 11. MuSiQue 실제 문항 상세: 공식 20개 candidate와 원문", "",
            "### 권장 읽기 순서", "",
            "| 순서 | 읽을 문항 | 먼저 확인할 것 |", "|---:|---|---|", "| 1 | [m1](#m1) | `UHF → Orion Pictures → Mike Medavoy`: 중간 답을 다음 검색 키로 쓰는 가장 짧은 clean chain |", "| 2 | [m2](#m2) | `The Hobbit → South Park → Trey Parker → Denver`: retrieval·중간 추론·최종 답변 실패를 구분하고, official grounding도 원문과 대조 |", "| 3 | [m3](#m3) | `Southeast Library → Ralph Rapson → Minneapolis → Mississippi River → Treaty of Paris`: 긴 chain의 누적 실패와 official grounding을 함께 점검 |", "",
            "아래에서는 공식 candidate 20개를 순서 그대로 둔다. ★ gold는 공식 `is_supporting=true` paragraph이며, 다른 16~18개는 공식 distractor다.", ""]
    for q, row, pids in details:
        body.append(details_block(q, row, pids))
    body += [END, ""]
    return "\n".join(body)


def write_mission(details):
    text = open(MISSION, encoding="utf-8").read()
    block = mission_section(details)
    if START in text and END in text:
        before, rest = text.split(START, 1)
        _, after = rest.split(END, 1)
        text = before.rstrip() + "\n\n" + block + after
    else:
        text = text.rstrip() + "\n\n" + block
        toc = "- [6. 빌드 전 완료 기준](#6-빌드-전-완료-기준)"
        replacement = toc + ("\n- [7. MuSiQue란 무엇인가](#7-musique란-무엇인가)"
                             "\n- [8. MuSiQue 현재 데이터는 어떻게 생겼나](#8-musique-현재-데이터는-어떻게-생겼나)"
                             "\n- [9. MuSiQue 공식 질문 구조](#9-musique-공식-질문-구조-hop-기반-순차-chain)"
                             "\n- [10. MuSiQue 3문항 한눈에 보기](#10-musique-3문항-한눈에-보기)"
                             "\n- [11. MuSiQue 실제 문항 상세](#11-musique-실제-문항-상세-공식-20개-candidate와-원문)")
        text = text.replace(toc, replacement, 1)
    open(MISSION, "w", encoding="utf-8").write(text)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default=DEFAULT_SOURCE)
    args = parser.parse_args()
    rows = load_rows(args.source)
    corpus, questions, details = build_slice(rows)
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "corpus.jsonl"), "w", encoding="utf-8") as f:
        for item in corpus:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    with open(os.path.join(OUT, "questions.json"), "w", encoding="utf-8") as f:
        json.dump(questions, f, ensure_ascii=False, indent=1)
    write_mission(details)
    print(f"wrote {OUT}: {len(corpus)} passages, {len(questions)} questions")
    for q in questions:
        print(f"  {q['id']} hop{q['hop']} gold={len(q['gold_pids'])} src={q['src_id']}")


if __name__ == "__main__":
    main()
