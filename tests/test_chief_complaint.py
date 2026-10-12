"""chief_complaint 테스트 (Gemini provider).

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

from app.chief_complaint import (
    RESPONSE_MIME_TYPE,
    ClassificationError,
    build_candidate_block,
    build_fallback_schema,
    build_response_schema,
    build_system_instruction,
    classify_chief_complaint,
    validate_result,
)
from app.config import DEFAULT_DATASET_DIR, FORBIDDEN_PROMPT_KEYS
from app.dataset_loader import load_datasets


@pytest.fixture(scope="session")
def datasets() -> dict[str, Any]:
    return load_datasets(DEFAULT_DATASET_DIR)


class _FakeResponse:
    def __init__(self, text: str | None) -> None:
        self.text = text
        self.prompt_feedback = None


class _FakeClient:
    """google-genai의 client.models.generate_content 만 흉내내는 가짜 클라이언트."""

    def __init__(self, payload: str | None, fail_json_schema: bool = False) -> None:
        self.payload = payload
        self.fail_json_schema = fail_json_schema
        self.calls: list[dict[str, Any]] = []
        outer = self

        class _Models:
            def generate_content(self, **kwargs: Any) -> _FakeResponse:
                outer.calls.append(kwargs)
                if outer.fail_json_schema and "response_json_schema" in kwargs["config"]:
                    raise RuntimeError("response_json_schema unsupported")
                return _FakeResponse(outer.payload)

        self.models = _Models()

    @property
    def last_kwargs(self) -> dict[str, Any]:
        return self.calls[-1]


def _classified_payload(datasets: dict[str, Any], cc_id: str) -> str:
    return json.dumps(
        {
            "status": "classified",
            "chief_complaint_id": cc_id,
            "chief_complaint_name": datasets[cc_id]["chief_complaint"]["name"],
        }
    )


# ── 1. 프롬프트 구성 ──────────────────────────────────────────────────


def test_prompt_contains_all_loaded_ids(datasets: dict[str, Any]) -> None:
    block = build_candidate_block(datasets)
    for cc_id, data in datasets.items():
        assert cc_id in block
        assert data["chief_complaint"]["name"] in block
        assert data["chief_complaint"]["canonical_term"] in block


# ── 2. schema enum ───────────────────────────────────────────────────


def test_schema_restricts_ids_to_loaded_datasets(datasets: dict[str, Any]) -> None:
    schema = build_response_schema(datasets)
    assert set(schema["properties"]["chief_complaint_id"]["enum"]) == set(datasets) | {None}
    assert schema["properties"]["status"]["enum"] == ["classified", "uncertain"]
    assert schema["required"] == [
        "status",
        "chief_complaint_id",
        "chief_complaint_name",
    ]


def test_fallback_schema_is_openapi_compatible(datasets: dict[str, Any]) -> None:
    """OpenAPI 서브셋은 type union을 지원하지 않으므로 nullable을 쓴다."""
    schema = build_fallback_schema(datasets)
    for field in ("chief_complaint_id", "chief_complaint_name"):
        assert schema["properties"][field]["type"] == "string"
        assert schema["properties"][field]["nullable"] is True


# ── 3. 빈 입력 ────────────────────────────────────────────────────────


def test_empty_text_returns_uncertain_without_api(datasets: dict[str, Any]) -> None:
    for text in ("", "   ", "\n\t "):
        assert classify_chief_complaint(text, datasets, client=None) == {
            "status": "uncertain",
            "chief_complaint_id": None,
            "chief_complaint_name": None,
        }


# ── 4. 결과 검증 ──────────────────────────────────────────────────────


def test_validate_result_overrides_name_from_dataset(datasets: dict[str, Any]) -> None:
    """이름은 LLM을 믿지 않고 데이터셋에서 가져온다."""
    cc_id = next(iter(datasets))
    result = validate_result(
        {
            "status": "classified",
            "chief_complaint_id": cc_id,
            "chief_complaint_name": "엉뚱한값",
        },
        datasets,
    )
    assert result["chief_complaint_name"] == datasets[cc_id]["chief_complaint"]["name"]


def test_validate_result_rejects_bad_status(datasets: dict[str, Any]) -> None:
    with pytest.raises(ClassificationError, match="status"):
        validate_result(
            {"status": "maybe", "chief_complaint_id": None, "chief_complaint_name": None},
            datasets,
        )


def test_validate_result_forces_nulls_on_uncertain(datasets: dict[str, Any]) -> None:
    result = validate_result(
        {
            "status": "uncertain",
            "chief_complaint_id": next(iter(datasets)),
            "chief_complaint_name": "복통",
        },
        datasets,
    )
    assert result["chief_complaint_id"] is None
    assert result["chief_complaint_name"] is None


# ── 5. fake client 왕복 ───────────────────────────────────────────────


def test_fake_client_flow(datasets: dict[str, Any]) -> None:
    cc_id = next(iter(datasets))
    client = _FakeClient(_classified_payload(datasets, cc_id))

    result = classify_chief_complaint("배가 아파요", datasets, client=client)

    assert result == {
        "status": "classified",
        "chief_complaint_id": cc_id,
        "chief_complaint_name": datasets[cc_id]["chief_complaint"]["name"],
    }
    kwargs = client.last_kwargs
    config = kwargs["config"]
    assert config["response_mime_type"] == RESPONSE_MIME_TYPE
    assert "response_json_schema" in config
    assert "system_instruction" in config
    # TEMPERATURE가 None이면 파라미터를 보내지 않는다.
    assert "temperature" not in config
    # 상담 메모를 변형하지 않고 그대로 전달한다.
    assert "배가 아파요" in kwargs["contents"]


def test_falls_back_to_response_schema(datasets: dict[str, Any]) -> None:
    """response_json_schema를 거부하면 response_schema로 재시도한다."""
    cc_id = next(iter(datasets))
    client = _FakeClient(_classified_payload(datasets, cc_id), fail_json_schema=True)

    result = classify_chief_complaint("배가 아파요", datasets, client=client)

    assert result["chief_complaint_id"] == cc_id
    assert len(client.calls) == 2
    assert "response_schema" in client.calls[1]["config"]


# ── 6. 잘못된 JSON ────────────────────────────────────────────────────


def test_invalid_json_from_api_raises(datasets: dict[str, Any]) -> None:
    client = _FakeClient("설명문이 섞인 응답 {status: classified}")
    with pytest.raises(ClassificationError, match="JSON 파싱 실패"):
        classify_chief_complaint("배가 아파요", datasets, client=client)


def test_empty_response_raises(datasets: dict[str, Any]) -> None:
    client = _FakeClient(None)
    with pytest.raises(ClassificationError, match="빈 응답"):
        classify_chief_complaint("배가 아파요", datasets, client=client)


# ── 7. 없는 ID ────────────────────────────────────────────────────────


def test_hallucinated_id_from_api_raises(datasets: dict[str, Any]) -> None:
    client = _FakeClient(
        json.dumps(
            {
                "status": "classified",
                "chief_complaint_id": "CC99_fever",
                "chief_complaint_name": "발열",
            }
        )
    )
    with pytest.raises(ClassificationError, match="데이터셋에 없는"):
        classify_chief_complaint("열이 나요", datasets, client=client)


# ── 9. 임상 정보 유출 방지 ────────────────────────────────────────────


def test_prompt_excludes_clinical_content(datasets: dict[str, Any]) -> None:
    """red_flag, source, metadata, validation_notes 등이 프롬프트에 유출되지 않는다."""
    prompt = build_system_instruction(datasets)
    for key in FORBIDDEN_PROMPT_KEYS:
        assert key not in prompt, f"금지 키가 프롬프트에 유출됨: {key}"

    for data in datasets.values():
        for item in data.get("check_items", []) + data.get("history_items", []):
            assert item["id"] not in prompt
            assert item["item"] not in prompt
            group = item.get("metadata", {}).get("observation_group")
            outcome = item.get("metadata", {}).get("source_flow_outcome")
            if outcome:
                assert outcome not in prompt
            if group:
                assert f'"{group}"' not in prompt


def test_clinical_content_absent_from_actual_request(datasets: dict[str, Any]) -> None:
    """실제로 전송되는 payload 전체에도 임상 정보가 없어야 한다."""
    cc_id = next(iter(datasets))
    client = _FakeClient(_classified_payload(datasets, cc_id))
    classify_chief_complaint("배가 아파요", datasets, client=client)

    sent = json.dumps(client.last_kwargs, ensure_ascii=False, default=str)
    for key in FORBIDDEN_PROMPT_KEYS:
        assert key not in sent, f"요청 payload에 금지 키 유출: {key}"
    for data in datasets.values():
        for item in data.get("check_items", []) + data.get("history_items", []):
            assert item["id"] not in sent


# ── 11. optional field 처리 ───────────────────────────────────────────


def test_dataset_without_expression_mapping_still_appears(datasets: dict[str, Any]) -> None:
    """CC01처럼 expression_mapping이 없어도 canonical_term으로 후보가 만들어진다."""
    without = [
        cc_id for cc_id, d in datasets.items() if not d.get("expression_mapping")
    ]
    block = build_candidate_block(datasets)
    for cc_id in without:
        cc = datasets[cc_id]["chief_complaint"]
        assert f"- {cc_id}: {cc['name']} (canonical_term: {cc['canonical_term']})" in block


def test_build_candidate_block_handles_minimal_dataset() -> None:
    """aliases와 expression_mapping이 아예 없는 데이터셋도 처리한다."""
    minimal = {
        "CCXX_a": {
            "chief_complaint": {"id": "CCXX_a", "name": "가", "canonical_term": "가"}
        }
    }
    assert build_candidate_block(minimal) == "- CCXX_a: 가 (canonical_term: 가)"


# ── API 호출 테스트 ───────────────────────────────────────────────────

requires_api_key = pytest.mark.skipif(
    not os.getenv("GEMINI_API_KEY"), reason="GEMINI_API_KEY 없음"
)


@pytest.mark.api
@requires_api_key
@pytest.mark.parametrize(
    ("text", "expected_id"),
    [
        ("머리가 지끈거리고 계속 아파요.", "CC03_headache"),
        ("계속 설사를 해요.", "CC04_diarrhea"),
        ("무릎이 붓고 아파요.", "CC05_joint_pain"),
        ("배가 아프고 구토도 했어요.", "CC01_abdominal_pain"),
        ("계속 기침하고 열도 나요.", "CC02_cough"),
    ],
)
def test_classification_live(datasets: dict[str, Any], text: str, expected_id: str) -> None:
    result = classify_chief_complaint(text, datasets)
    assert result["status"] == "classified"
    assert result["chief_complaint_id"] == expected_id


@pytest.mark.api
@requires_api_key
def test_uncertain_live(datasets: dict[str, Any]) -> None:
    result = classify_chief_complaint("몸이 좀 이상한 것 같아요.", datasets)
    assert result["status"] == "uncertain"
    assert result["chief_complaint_id"] is None
