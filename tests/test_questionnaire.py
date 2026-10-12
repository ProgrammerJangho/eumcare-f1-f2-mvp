"""F2(문진 완결성 점검) 테스트.

- 상단 테스트: Gemini API를 호출하지 않는 순수 단위 테스트.
- @pytest.mark.api 테스트: 실제 API를 호출한다.
  GEMINI_API_KEY가 없으면 자동으로 skip된다.

API 테스트만 실행:   pytest -m api
API 테스트 제외:     pytest -m "not api"
"""

from __future__ import annotations

import json
import os
from typing import Any

import pytest

from app.completeness import calculate_completeness
from app.config import (
    DEFAULT_DATASET_DIR,
    F2_FORBIDDEN_PROMPT_KEYS,
    MODEL_INPUT_FIELDS,
)
from app.dataset_loader import load_datasets
from app.observation_group import (
    apply_observation_groups,
    get_observation_group,
    resolve_group_status,
)
from app.errors import QuotaExceededError
from app.questionnaire import (
    QuestionnaireError,
    QuestionnaireInputError,
    QuestionnaireQuotaError,
    QuestionnaireValidationError,
    run_f2_only,
    build_f2_fallback_schema,
    build_f2_prompt,
    build_f2_schema,
    check_questionnaire,
    load_questionnaire,
    project_item,
    run_f2,
    validate_f2_result,
)


@pytest.fixture(scope="session")
def datasets() -> dict[str, Any]:
    return load_datasets(DEFAULT_DATASET_DIR)


class _FakeResponse:
    def __init__(self, text: str | None) -> None:
        self.text = text
        self.prompt_feedback = None


class _FakeClient:
    """google-genai의 client.models.generate_content 만 흉내낸다."""

    def __init__(self, payload: str | None) -> None:
        self.payload = payload
        self.calls: list[dict[str, Any]] = []
        outer = self

        class _Models:
            def generate_content(self, **kwargs: Any) -> _FakeResponse:
                outer.calls.append(kwargs)
                return _FakeResponse(outer.payload)

        self.models = _Models()

    @property
    def last_kwargs(self) -> dict[str, Any]:
        return self.calls[-1]


def _payload(statuses: dict[str, str]) -> str:
    return json.dumps(
        {"items": [{"id": i, "status": s} for i, s in statuses.items()]}
    )


def _all(items: list[dict[str, Any]], status: str) -> dict[str, str]:
    return {item["id"]: status for item in items}


# ── 문진 항목 로드 ────────────────────────────────────────────────────


def test_load_questionnaire_uses_only_selected_dataset(datasets: dict[str, Any]) -> None:
    dataset = datasets["CC03_headache"]
    items = load_questionnaire(dataset)

    expected = len(dataset["history_items"]) + len(dataset["check_items"])
    assert len(items) == expected
    # 데이터셋 순서(history 다음 check)를 유지한다.
    assert [i["id"] for i in items] == [
        i["id"] for i in dataset["history_items"] + dataset["check_items"]
    ]
    # 다른 주호소의 항목이 섞이지 않는다.
    other_ids = {
        i["id"]
        for cc_id, d in datasets.items()
        if cc_id != "CC03_headache"
        for i in d["history_items"] + d["check_items"]
    }
    assert not ({i["id"] for i in items} & other_ids)


def test_load_questionnaire_handles_empty_history(datasets: dict[str, Any]) -> None:
    """CC05는 history_items가 비어 있다."""
    items = load_questionnaire(datasets["CC05_joint_pain"])
    assert len(items) == len(datasets["CC05_joint_pain"]["check_items"])


# ── 모델 입력 필드 제한 ───────────────────────────────────────────────


def test_project_item_has_exactly_five_fields(datasets: dict[str, Any]) -> None:
    for dataset in datasets.values():
        assert tuple(dataset["usage_policy"]["model_input_fields"]) == MODEL_INPUT_FIELDS
        for item in load_questionnaire(dataset):
            assert set(project_item(item)) == set(MODEL_INPUT_FIELDS)


def test_f2_prompt_excludes_clinical_metadata(datasets: dict[str, Any]) -> None:
    """source, metadata, observation_group 등이 F2 프롬프트에 유출되지 않는다.

    substring 검사만으로는 부족하다. 그룹 키 "구토"는 HA_R002의 item 값과
    글자가 같고, 일부 항목은 source_text가 description에 그대로 포함되어 있어
    정상적인 전달 내용과 구분되지 않는다. 따라서 구조적으로 검사한다:
    5개 필드만 남긴 항목으로 만든 프롬프트와 원본 항목으로 만든 프롬프트가
    완전히 같아야 한다. 같다면 5개 필드 밖의 어떤 값도 프롬프트에 영향을
    줄 수 없다.
    """
    for cc_id, dataset in datasets.items():
        items = load_questionnaire(dataset)
        full_prompt, _ = build_f2_prompt("배가 아파요", items)
        stripped = [project_item(i) for i in items]
        stripped_prompt, _ = build_f2_prompt("배가 아파요", stripped)

        assert full_prompt == stripped_prompt, f"{cc_id}: 5개 필드 밖의 값이 반영됨"

        # 필드 이름 자체는 어떤 형태로도 등장하지 않아야 한다.
        for key in F2_FORBIDDEN_PROMPT_KEYS:
            assert key not in full_prompt, f"{cc_id}: 금지 키 유출 {key}"


def test_grouped_items_look_independent_in_prompt(datasets: dict[str, Any]) -> None:
    """그룹으로 묶인 항목이 프롬프트에서 서로 연결되어 보이지 않는다.

    LLM은 HA_009와 HA_R002가 같은 관찰이라는 사실을 알 수 없어야 한다.
    두 항목은 다른 항목과 똑같은 형식으로 각각 독립 제시된다.
    """
    items = load_questionnaire(datasets["CC03_headache"])
    grouped = [i for i in items if get_observation_group(i) == "구토"]
    assert len(grouped) == 2

    prompt, _ = build_f2_prompt("토했어요", items)
    for item in grouped:
        block = (
            f"[{item['id']}] ({item['category']} / {item['check_method']})\n"
            f"    항목: {item['item']}\n"
            f"    설명: {item['description']}"
        )
        assert block in prompt
    # 두 id가 한 줄에 함께 등장하지 않는다.
    for line in prompt.splitlines():
        assert not (grouped[0]["id"] in line and grouped[1]["id"] in line)


def test_f2_request_payload_excludes_clinical_metadata(datasets: dict[str, Any]) -> None:
    """실제로 전송되는 요청 payload 전체에 금지 키가 없어야 한다."""
    dataset = datasets["CC03_headache"]
    items = load_questionnaire(dataset)
    client = _FakeClient(_payload(_all(items, "unconfirmed")))

    check_questionnaire("머리가 아파요", items, client=client)

    sent = json.dumps(client.last_kwargs, ensure_ascii=False, default=str)
    for key in F2_FORBIDDEN_PROMPT_KEYS:
        assert key not in sent, f"요청 payload에 금지 키 유출: {key}"


def test_prompt_keeps_consultation_text_verbatim(datasets: dict[str, Any]) -> None:
    text = "어제부터 머리가 너무 아프고 지끈거려요. 토할 것 같기도 해요."
    _, user_prompt = build_f2_prompt(
        text, load_questionnaire(datasets["CC03_headache"])
    )
    assert text in user_prompt


# ── schema ────────────────────────────────────────────────────────────


def test_f2_schema_restricts_ids(datasets: dict[str, Any]) -> None:
    items = load_questionnaire(datasets["CC03_headache"])
    schema = build_f2_schema(items)
    entry = schema["properties"]["items"]

    assert entry["items"]["properties"]["id"]["enum"] == [i["id"] for i in items]
    assert entry["items"]["properties"]["status"]["enum"] == [
        "confirmed",
        "unconfirmed",
        "uncertain",
    ]
    assert entry["minItems"] == entry["maxItems"] == len(items)


def test_f2_fallback_schema_has_no_id_enum(datasets: dict[str, Any]) -> None:
    items = load_questionnaire(datasets["CC03_headache"])
    schema = build_f2_fallback_schema(items)
    assert "enum" not in schema["properties"]["items"]["items"]["properties"]["id"]


# ── 결과 검증 ─────────────────────────────────────────────────────────


def test_validate_reorders_to_dataset_order(datasets: dict[str, Any]) -> None:
    """LLM이 준 순서를 믿지 않고 데이터셋 순서로 재구성한다."""
    items = load_questionnaire(datasets["CC05_joint_pain"])
    shuffled = list(reversed([i["id"] for i in items]))
    result = {"items": [{"id": i, "status": "unconfirmed"} for i in shuffled]}

    resolved = validate_f2_result(result, items)

    assert list(resolved) == [i["id"] for i in items]


def test_validate_rejects_missing_item(datasets: dict[str, Any]) -> None:
    items = load_questionnaire(datasets["CC05_joint_pain"])
    statuses = _all(items, "unconfirmed")
    statuses.pop(items[0]["id"])

    with pytest.raises(QuestionnaireError, match="누락된"):
        validate_f2_result(json.loads(_payload(statuses)), items)


def test_validate_rejects_unknown_item(datasets: dict[str, Any]) -> None:
    items = load_questionnaire(datasets["CC05_joint_pain"])
    result = json.loads(_payload(_all(items, "unconfirmed")))
    result["items"].append({"id": "MSK_R999", "status": "confirmed"})

    with pytest.raises(QuestionnaireError, match="데이터셋에 없는"):
        validate_f2_result(result, items)


def test_validate_rejects_duplicate_item(datasets: dict[str, Any]) -> None:
    items = load_questionnaire(datasets["CC05_joint_pain"])
    result = json.loads(_payload(_all(items, "unconfirmed")))
    result["items"].append({"id": items[0]["id"], "status": "confirmed"})

    with pytest.raises(QuestionnaireError, match="중복"):
        validate_f2_result(result, items)


def test_validate_rejects_bad_status(datasets: dict[str, Any]) -> None:
    items = load_questionnaire(datasets["CC05_joint_pain"])
    result = json.loads(_payload(_all(items, "unconfirmed")))
    result["items"][0]["status"] = "maybe"

    with pytest.raises(QuestionnaireError, match="status"):
        validate_f2_result(result, items)


# ── observation_group 후처리 ──────────────────────────────────────────


def test_group_merges_only_in_display_items(datasets: dict[str, Any]) -> None:
    """Test 7 - CC03의 구토는 display에서 하나로 합쳐지고 raw는 보존된다."""
    dataset = datasets["CC03_headache"]
    items = load_questionnaire(dataset)
    grouped = [i["id"] for i in items if get_observation_group(i) == "구토"]
    assert len(grouped) == 2, "CC03에 구토 그룹 항목이 2개여야 한다"

    statuses = _all(items, "unconfirmed")
    for item_id in grouped:
        statuses[item_id] = "confirmed"
    client = _FakeClient(_payload(statuses))

    result = run_f2("토했어요", dataset, client=client)

    raw = {r["id"]: r["status"] for r in result["raw_items"]}
    assert len(result["raw_items"]) == len(items)
    for item_id in grouped:
        assert raw[item_id] == "confirmed"

    merged = [d for d in result["display_items"] if d["display_key"] == "구토"]
    assert len(merged) == 1
    assert merged[0]["status"] == "confirmed"
    assert sorted(merged[0]["source_item_ids"]) == sorted(grouped)
    assert len(result["display_items"]) == len(items) - 1


def test_similar_items_without_group_stay_separate(datasets: dict[str, Any]) -> None:
    """Test 8 - CC02의 38도 이상 발열과 3일 이상 지속은 그룹이 아니다."""
    dataset = datasets["CC02_cough"]
    items = load_questionnaire(dataset)
    assert all(get_observation_group(i) is None for i in items)

    client = _FakeClient(_payload(_all(items, "confirmed")))
    result = run_f2("사흘째 38도가 넘어요", dataset, client=client)

    keys = {d["display_key"] for d in result["display_items"]}
    assert {"COUGH_R006", "COUGH_R007"} <= keys
    assert len(result["display_items"]) == len(items)


def test_group_status_policies() -> None:
    assert resolve_group_status(["confirmed", "confirmed"]) == "confirmed"
    assert resolve_group_status(["unconfirmed", "unconfirmed"]) == "unconfirmed"
    # 기본(conservative): 섞이면 uncertain
    assert resolve_group_status(["confirmed", "unconfirmed"]) == "uncertain"
    assert resolve_group_status(["confirmed", "uncertain"]) == "uncertain"
    # any_confirmed 정책
    assert (
        resolve_group_status(["confirmed", "unconfirmed"], policy="any_confirmed")
        == "confirmed"
    )


def test_display_items_follow_dataset_order(datasets: dict[str, Any]) -> None:
    dataset = datasets["CC04_diarrhea"]
    items = load_questionnaire(dataset)
    display = apply_observation_groups(_all(items, "unconfirmed"), items)

    first_ids = [d["source_item_ids"][0] for d in display]
    assert first_ids == [i["id"] for i in items if i["id"] in set(first_ids)]


# ── 완결성 계산 ───────────────────────────────────────────────────────


def test_completeness_counts_display_items() -> None:
    display = [
        {"display_key": "a", "item": "a", "status": "confirmed", "source_item_ids": ["a"]},
        {"display_key": "b", "item": "b", "status": "confirmed", "source_item_ids": ["b"]},
        {"display_key": "c", "item": "c", "status": "unconfirmed", "source_item_ids": ["c"]},
        {"display_key": "d", "item": "d", "status": "uncertain", "source_item_ids": ["d"]},
    ]
    summary = calculate_completeness(display)
    assert summary == {
        "total_items": 4,
        "confirmed_items": 2,
        "unconfirmed_items": 1,
        "uncertain_items": 1,
        "completeness_percent": 50.0,
    }


def test_completeness_uses_group_merged_count(datasets: dict[str, Any]) -> None:
    """그룹으로 합쳐진 항목은 1건으로 센다."""
    dataset = datasets["CC03_headache"]
    items = load_questionnaire(dataset)
    client = _FakeClient(_payload(_all(items, "confirmed")))

    result = run_f2("전부 확인했어요", dataset, client=client)

    assert result["total_items"] == len(items) - 1
    assert result["completeness_percent"] == 100.0


def test_completeness_policy_is_swappable() -> None:
    display = [
        {"display_key": "a", "item": "a", "status": "confirmed", "source_item_ids": ["a"]},
        {"display_key": "b", "item": "b", "status": "uncertain", "source_item_ids": ["b"]},
    ]
    assert calculate_completeness(display)["completeness_percent"] == 50.0
    lenient = calculate_completeness(
        display, confirmed_statuses=("confirmed", "uncertain")
    )
    assert lenient["completeness_percent"] == 100.0


# ── run_f2 통합 (fake client) ─────────────────────────────────────────


def test_run_f2_output_shape(datasets: dict[str, Any]) -> None:
    dataset = datasets["CC05_joint_pain"]
    items = load_questionnaire(dataset)
    statuses = _all(items, "unconfirmed")
    statuses[items[0]["id"]] = "confirmed"

    result = run_f2("무릎이 아파요", dataset, client=_FakeClient(_payload(statuses)))

    assert set(result) == {
        "raw_items",
        "display_items",
        "total_items",
        "confirmed_items",
        "unconfirmed_items",
        "uncertain_items",
        "completeness_percent",
    }
    for entry in result["display_items"]:
        assert set(entry) == {
            "display_key",
            "item",
            "description",
            "status",
            "source_item_ids",
        }


def test_run_f2_rejects_empty_text(datasets: dict[str, Any]) -> None:
    with pytest.raises(QuestionnaireError, match="비어 있습니다"):
        run_f2("   ", datasets["CC05_joint_pain"], client=_FakeClient("{}"))


def test_invalid_json_from_api_raises(datasets: dict[str, Any]) -> None:
    items = load_questionnaire(datasets["CC05_joint_pain"])
    with pytest.raises(QuestionnaireError, match="JSON 파싱 실패"):
        check_questionnaire("무릎이 아파요", items, client=_FakeClient("설명문 {a:1}"))


# ── F1 + F2 연결 ──────────────────────────────────────────────────────


def test_f1_uncertain_skips_f2(monkeypatch: pytest.MonkeyPatch, datasets: dict[str, Any]) -> None:
    """F1이 uncertain이면 F2를 호출하지 않는다."""
    import main as main_module

    monkeypatch.setattr(
        main_module,
        "classify_chief_complaint",
        lambda text, ds: {
            "status": "uncertain",
            "chief_complaint_id": None,
            "chief_complaint_name": None,
        },
    )

    def _boom(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("F1이 uncertain인데 F2가 호출되었다")

    monkeypatch.setattr(main_module, "run_f2", _boom)

    result = main_module.analyze("몸이 좀 이상해요", datasets)
    assert result["status"] == "uncertain"
    assert result["f2"] is None


def test_f1_classified_runs_f2(monkeypatch: pytest.MonkeyPatch, datasets: dict[str, Any]) -> None:
    import main as main_module

    monkeypatch.setattr(
        main_module,
        "classify_chief_complaint",
        lambda text, ds: {
            "status": "classified",
            "chief_complaint_id": "CC05_joint_pain",
            "chief_complaint_name": "관절통",
        },
    )
    called: list[str] = []

    def _fake_run_f2(text: str, dataset: dict[str, Any]) -> dict[str, Any]:
        called.append(dataset["chief_complaint"]["id"])
        return {"total_items": 4, "completeness_percent": 0.0, "display_items": []}

    monkeypatch.setattr(main_module, "run_f2", _fake_run_f2)

    result = main_module.analyze("무릎이 아파요", datasets)
    assert called == ["CC05_joint_pain"]
    assert result["f2"] is not None


# ── API 호출 테스트 ───────────────────────────────────────────────────

requires_api_key = pytest.mark.skipif(
    not os.getenv("GEMINI_API_KEY"), reason="GEMINI_API_KEY 없음"
)


def _status_of(result: dict[str, Any], item_id: str) -> str:
    return {r["id"]: r["status"] for r in result["raw_items"]}[item_id]


@pytest.mark.api
@requires_api_key
def test_headache_live(datasets: dict[str, Any]) -> None:
    """Test 1 - 두통. 불안·긴장 시 악화와 휴식 시 호전이 확인된다."""
    result = run_f2(
        "머리가 욱신거리는데 스트레스 받으면 더 심해지고 쉬면 좀 괜찮아져요.",
        datasets["CC03_headache"],
    )
    assert _status_of(result, "HA_003") == "confirmed"
    assert _status_of(result, "HA_006") == "confirmed"


@pytest.mark.api
@requires_api_key
def test_joint_pain_live(datasets: dict[str, Any]) -> None:
    """Test 4 - 관절통. 부종과 열감이 확인된다."""
    result = run_f2("무릎이 붓고 뜨거운 것 같아요.", datasets["CC05_joint_pain"])
    assert _status_of(result, "MSK_R003") == "confirmed"
    assert _status_of(result, "MSK_R004") == "confirmed"


@pytest.mark.api
@requires_api_key
def test_negation_counts_as_confirmed_live(datasets: dict[str, Any]) -> None:
    """Test 5 - 부정 표현도 '확인됨'이다."""
    result = run_f2(
        "일주일 전부터 기침이 계속돼요. 열은 없었어요.", datasets["CC02_cough"]
    )
    assert _status_of(result, "COUGH_R006") == "confirmed"


@pytest.mark.api
@requires_api_key
def test_no_information_live(datasets: dict[str, Any]) -> None:
    """Test 6 - 정보가 없으면 전부 unconfirmed."""
    result = run_f2("오늘 병원 왔어요.", datasets["CC05_joint_pain"])
    assert all(r["status"] == "unconfirmed" for r in result["raw_items"])


# ══════════════════════════════════════════════════════════════════════
# 과잉 판정 방지 (§2~§11)
# ══════════════════════════════════════════════════════════════════════

# 항목 이름 -> 실제 데이터셋 위치. 데이터셋을 수정하지 않고 실제 항목만 쓴다.
STRICTNESS_CASES = [
    # (테스트 이름, chief_complaint_id, item_id, 상담 메모, 기대 status)
    ("통증 부위 / 상위 언급", "CC01_abdominal_pain", "ABD_008", "배가 아파요.", "unconfirmed"),
    ("통증 부위 / 세부 언급", "CC01_abdominal_pain", "ABD_008", "오른쪽 아랫배가 아파요.", "confirmed"),
    ("발열 여부 / 명시적 부정", "CC04_diarrhea", "DIA_R002", "열은 없어요.", "confirmed"),
    ("기침 지속기간 / 상위 언급", "CC02_cough", "COUGH_001", "기침해요.", "unconfirmed"),
    ("기침 지속기간 / 기간 언급", "CC02_cough", "COUGH_001", "일주일째 기침해요.", "confirmed"),
    ("관절 열감 / 상위 언급", "CC05_joint_pain", "MSK_R004", "무릎이 아파요.", "unconfirmed"),
    ("관절 열감 / 열감 언급", "CC05_joint_pain", "MSK_R004", "무릎이 뜨거워요.", "confirmed"),
]


def test_strictness_cases_use_real_dataset_items(datasets: dict[str, Any]) -> None:
    """§18 케이스가 실제 데이터셋 항목을 가리키는지 확인한다."""
    for _, cc_id, item_id, _, _ in STRICTNESS_CASES:
        ids = {i["id"] for i in load_questionnaire(datasets[cc_id])}
        assert item_id in ids, f"{cc_id}에 {item_id}가 없다"


def test_prompt_contains_strictness_examples(datasets: dict[str, Any]) -> None:
    """§11 - 7개 판정 예시가 system instruction에 실제로 들어간다."""
    items = load_questionnaire(datasets["CC01_abdominal_pain"])
    prompt, _ = build_f2_prompt("배가 아파요.", items)

    for fragment in (
        '항목 "통증 부위" / 상담 "배가 아파요." -> unconfirmed',
        '항목 "통증 부위" / 상담 "오른쪽 아랫배가 아파요." -> confirmed',
        '항목 "발열 여부" / 상담 "열은 없어요." -> confirmed',
        '항목 "기침 지속기간" / 상담 "기침해요." -> unconfirmed',
        '항목 "기침 지속기간" / 상담 "일주일째 기침해요." -> confirmed',
        '항목 "관절의 열감" / 상담 "무릎이 아파요." -> unconfirmed',
        '항목 "관절의 열감" / 상담 "무릎이 뜨거워요." -> confirmed',
    ):
        assert fragment in prompt, f"예시 누락: {fragment}"


def test_prompt_states_overreach_rule(datasets: dict[str, Any]) -> None:
    """상위 개념으로 세부 항목을 채우지 말라는 규칙이 프롬프트에 있다."""
    items = load_questionnaire(datasets["CC01_abdominal_pain"])
    prompt, _ = build_f2_prompt("배가 아파요.", items)
    assert "상위 개념으로 세부 항목을 채우지 않는다" in prompt
    assert "다른 문진 항목의 정보에서 유추해야 하는 경우" in prompt


# ══════════════════════════════════════════════════════════════════════
# 실제 dataset 기반 회귀 (§19) - fake client
# ══════════════════════════════════════════════════════════════════════


def _run_with_expected(
    dataset: dict[str, Any], text: str, expected: dict[str, str]
) -> dict[str, Any]:
    """expected에 없는 항목은 unconfirmed로 채운 fake 응답으로 run_f2를 돌린다."""
    items = load_questionnaire(dataset)
    statuses = _all(items, "unconfirmed")
    statuses.update(expected)
    return run_f2(text, dataset, client=_FakeClient(_payload(statuses)))


def test_cc01_regression(datasets: dict[str, Any]) -> None:
    """§19 CC01 - 통증 부위가 과잉 confirmed되지 않는다."""
    result = _run_with_expected(
        datasets["CC01_abdominal_pain"],
        "어제부터 배가 아팠고 오늘은 더 심해졌어요. 토하기도 했어요.",
        {"ABD_003": "confirmed", "ABD_006": "confirmed", "ABD_R007": "confirmed"},
    )
    raw = {r["id"]: r["status"] for r in result["raw_items"]}
    assert raw["ABD_003"] == "confirmed"      # 발생 시점
    assert raw["ABD_006"] == "confirmed"      # 구토 동반 여부
    assert raw["ABD_R007"] == "confirmed"     # 복통 후 구토가 이어짐
    assert raw["ABD_008"] == "unconfirmed"    # 통증 부위 - 과잉 판정 금지


def test_cc02_regression(datasets: dict[str, Any]) -> None:
    result = _run_with_expected(
        datasets["CC02_cough"],
        "일주일 전부터 기침이 계속되고 열은 없었어요.",
        {"COUGH_001": "confirmed", "COUGH_R006": "confirmed"},
    )
    raw = {r["id"]: r["status"] for r in result["raw_items"]}
    assert raw["COUGH_001"] == "confirmed"    # 기침 지속기간
    assert raw["COUGH_R006"] == "confirmed"   # 38도 이상의 발열
    assert raw["COUGH_R003"] == "unconfirmed"  # 상담에 없는 항목


def test_cc03_regression(datasets: dict[str, Any]) -> None:
    result = _run_with_expected(
        datasets["CC03_headache"],
        "머리가 욱신거리는데 스트레스 받으면 더 심해지고 쉬면 좀 괜찮아져요.",
        {"HA_003": "confirmed", "HA_006": "confirmed"},
    )
    raw = {r["id"]: r["status"] for r in result["raw_items"]}
    assert raw["HA_003"] == "confirmed"       # 불안·긴장 시 악화
    assert raw["HA_006"] == "confirmed"       # 휴식 시 호전
    assert raw["HA_004"] == "unconfirmed"     # 목 움직임 시 호전
    assert raw["HA_005"] == "unconfirmed"     # 더운 찜질 시 호전


def test_cc05_regression(datasets: dict[str, Any]) -> None:
    result = _run_with_expected(
        datasets["CC05_joint_pain"],
        "오른쪽 무릎이 붓고 뜨거운 느낌이 있어요.",
        {"MSK_R003": "confirmed", "MSK_R004": "confirmed"},
    )
    raw = {r["id"]: r["status"] for r in result["raw_items"]}
    assert raw["MSK_R003"] == "confirmed"     # 관절이 부어 있음
    assert raw["MSK_R004"] == "confirmed"     # 관절의 열감


# ══════════════════════════════════════════════════════════════════════
# F2-only 모드 (§16, §17)
# ══════════════════════════════════════════════════════════════════════


def test_run_f2_only_skips_f1(datasets: dict[str, Any]) -> None:
    items = load_questionnaire(datasets["CC05_joint_pain"])
    client = _FakeClient(_payload(_all(items, "unconfirmed")))

    result = run_f2_only("무릎이 아파요", "CC05_joint_pain", datasets, client=client)

    assert result["total_items"] == len(items)
    # F2 호출 1회뿐. F1은 호출되지 않는다.
    assert len(client.calls) == 1


def test_run_f2_only_rejects_unknown_cc(datasets: dict[str, Any]) -> None:
    with pytest.raises(QuestionnaireInputError, match="데이터셋에 없는"):
        run_f2_only("무릎이 아파요", "CC99_nope", datasets, client=_FakeClient("{}"))


def test_analyze_f2_only_shape(datasets: dict[str, Any]) -> None:
    """main.analyze_f2_only()가 analyze()와 같은 결과 구조를 만든다."""
    items = load_questionnaire(datasets["CC05_joint_pain"])
    client = _FakeClient(_payload(_all(items, "unconfirmed")))
    direct = run_f2_only("무릎이 아파요", "CC05_joint_pain", datasets, client=client)

    assert set(direct) >= {"raw_items", "display_items", "completeness_percent"}


# ══════════════════════════════════════════════════════════════════════
# 429 / 실패 유형 구분 (§22, §23)
# ══════════════════════════════════════════════════════════════════════


class _FailingClient:
    """지정한 예외를 발생시키는 가짜 클라이언트."""

    def __init__(self, exc: Exception) -> None:
        self.exc = exc
        self.calls = 0
        outer = self

        class _Models:
            def generate_content(self, **kwargs: Any) -> Any:
                outer.calls += 1
                raise outer.exc

        self.models = _Models()


def test_quota_error_is_distinguished(datasets: dict[str, Any]) -> None:
    """429는 일반 API 오류와 구분되고, 재시도하지 않는다."""
    items = load_questionnaire(datasets["CC05_joint_pain"])
    client = _FailingClient(RuntimeError("429 RESOURCE_EXHAUSTED: quota exceeded"))

    with pytest.raises(QuestionnaireQuotaError) as excinfo:
        check_questionnaire("무릎이 아파요", items, client=client)

    assert isinstance(excinfo.value, QuotaExceededError)
    assert isinstance(excinfo.value, QuestionnaireError)
    assert "할당량" in str(excinfo.value)
    # 429에서는 fallback 스키마로 두 번째 호출을 하지 않는다.
    assert client.calls == 1


def test_network_error_does_not_retry(datasets: dict[str, Any]) -> None:
    """스키마와 무관한 오류에서도 중복 호출하지 않는다."""
    items = load_questionnaire(datasets["CC05_joint_pain"])
    client = _FailingClient(ConnectionError("connection reset"))

    with pytest.raises(QuestionnaireError):
        check_questionnaire("무릎이 아파요", items, client=client)

    assert client.calls == 1


def test_api_attempts_are_capped(datasets: dict[str, Any]) -> None:
    """스키마 미지원 오류여도 호출은 MAX_API_ATTEMPTS를 넘지 않는다."""
    from app.config import MAX_API_ATTEMPTS

    items = load_questionnaire(datasets["CC05_joint_pain"])
    client = _FailingClient(RuntimeError("400 INVALID_ARGUMENT: schema unsupported"))

    with pytest.raises(QuestionnaireError):
        check_questionnaire("무릎이 아파요", items, client=client)

    assert client.calls <= MAX_API_ATTEMPTS


def test_error_types_are_distinguishable(datasets: dict[str, Any]) -> None:
    """§23 - 입력 오류와 검증 오류가 서로 다른 타입이다."""
    items = load_questionnaire(datasets["CC05_joint_pain"])

    with pytest.raises(QuestionnaireInputError):
        check_questionnaire("   ", items, client=_FakeClient("{}"))

    with pytest.raises(QuestionnaireValidationError):
        validate_f2_result({"items": []}, items)


# ══════════════════════════════════════════════════════════════════════
# API 테스트 - F2-only 로 호출 (§17: F1 호출 생략)
# ══════════════════════════════════════════════════════════════════════


@pytest.mark.api
@requires_api_key
@pytest.mark.parametrize(
    ("label", "cc_id", "item_id", "text", "expected"),
    STRICTNESS_CASES,
    ids=[c[0] for c in STRICTNESS_CASES],
)
def test_strictness_live(
    datasets: dict[str, Any],
    label: str,
    cc_id: str,
    item_id: str,
    text: str,
    expected: str,
) -> None:
    """§18 Test 1~7 - 실제 판정. F1을 건너뛰어 호출 1회만 쓴다."""
    result = run_f2_only(text, cc_id, datasets)
    actual = {r["id"]: r["status"] for r in result["raw_items"]}[item_id]
    assert actual == expected, f"{label}: {item_id} = {actual} (기대 {expected})"


@pytest.mark.api
@requires_api_key
def test_cc01_no_overreach_live(datasets: dict[str, Any]) -> None:
    """§26-8 - CC01에서 통증 부위가 과잉 confirmed되지 않는다."""
    result = run_f2_only(
        "어제부터 배가 아팠고 오늘은 더 심해졌어요. 토하기도 했어요.",
        "CC01_abdominal_pain",
        datasets,
    )
    raw = {r["id"]: r["status"] for r in result["raw_items"]}
    assert raw["ABD_008"] == "unconfirmed"
    assert raw["ABD_006"] == "confirmed"
