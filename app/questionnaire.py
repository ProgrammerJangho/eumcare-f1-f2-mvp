"""F2 문진 완결성 점검.

선택된 chief complaint의 데이터셋에 있는 문진 항목만 사용해, 상담 메모에서
각 항목이 확인되었는지 판정한다.

status의 의미 (매우 중요):
    confirmed   = 해당 문진 항목을 "확인했다".
                  증상이 존재한다는 뜻이 아니다.
                  "열은 없었어요"처럼 명시적 부정도 확인한 것이므로 confirmed다.
    unconfirmed = 상담 메모에 해당 항목에 대한 정보가 없다.
    uncertain   = 내용이 애매해 확인 여부를 단정할 수 없다.

이 모듈은 진단, 응급 여부, 의뢰 여부, 위험도, 치료를 판단하지 않으며
그런 정보를 LLM에 전달하지도 않는다.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from app.chief_complaint import call_gemini_json, create_client
from app.errors import (
    EumcareError,
    GeminiAPIError,
    InvalidInputError,
    QuotaExceededError,
    ResultValidationError,
)
from app.completeness import calculate_completeness
from app.config import ITEM_STATUSES, MODEL_INPUT_FIELDS, MODEL_NAME
from app.observation_group import apply_observation_groups

logger = logging.getLogger(__name__)

SYSTEM_INSTRUCTION = """\
너는 문진 완결성 점검 도구다.

상담 메모와 아래 문진 항목을 비교하여, 각 항목이 요구하는 정보가 상담에서
확인되었는지만 판단한다.

## 판정 값의 의미

- confirmed: 그 문진 항목이 요구하는 정보가 상담 메모에서 확인되었다.
  "그 증상이 있다"는 뜻이 아니라 "그 항목을 확인했다"는 뜻이다.
- unconfirmed: 상담 메모에 그 항목이 요구하는 정보가 없다.
- uncertain: 상담 표현이 애매해서 확인 여부를 단정하기 어렵다.

## 가장 중요한 규칙: 상위 개념으로 세부 항목을 채우지 않는다

문진 항목이 요구하는 정보 단위가 상담 메모에 실제로 나타나야 confirmed다.
상위 증상만 언급되었다면, 그 아래 세부 항목은 아직 확인되지 않은 것이다.

- "배가 아파요"는 복통이 있다는 정보다. 통증이 어느 부위인지는 확인되지
  않았다.
- "무릎이 아파요"는 관절통이 있다는 정보다. 열감이 있는지는 확인되지
  않았다.
- "머리가 아파요"는 두통이 있다는 정보다. 얼마나 지속되었는지는 확인되지
  않았다.

## confirmed로 만들지 않는 경우

- 상위 개념이나 관련 증상만 언급된 경우
- 일반 상식이나 임상적 개연성으로 추론해야 하는 경우
- 다른 문진 항목의 정보에서 유추해야 하는 경우

## 부정 표현

명시적인 부정도 그 항목을 확인한 것이므로 confirmed다.
- 항목 "발열", 상담 "열은 없어요" -> confirmed (발열 여부를 확인했다)
- 항목 "구토 동반 여부", 상담 "토하지 않았어요" -> confirmed
반면 항목과 무관한 일반적 진술은 confirmed가 아니다.
- 항목 "발열", 상담 "요즘 몸이 괜찮아요" -> unconfirmed

## 수치와 조건이 붙은 항목

항목이 특정 기간이나 수치를 요구하면, 상담에 그 기준을 판단할 수 있는
정보가 있어야 confirmed다. 항목에 없는 기준을 새로 만들지 않는다.
- 항목 "14일 이상 지속되는 설사", 상담 "설사해요" -> unconfirmed
- 항목 "14일 이상 지속되는 설사", 상담 "3일째 설사해요" -> unconfirmed
- 항목 "14일 이상 지속되는 설사", 상담 "2주 넘게 설사했어요" -> confirmed
- 항목 "38도 이상의 발열", 상담 "열이 나요" -> unconfirmed
- 항목 "38도 이상의 발열", 상담 "38.5도까지 올랐어요" -> confirmed

## uncertain 사용 기준

정보가 명백히 없으면 unconfirmed, 명백히 있으면 confirmed다. 애매하다는
이유로 습관적으로 uncertain을 고르지 않는다. 여러 해석이 가능하거나 상위
언급인지 세부 확인인지 구분되지 않을 때만 uncertain을 쓴다.

## 판정 예시

1. 항목 "통증 부위" / 상담 "배가 아파요." -> unconfirmed
2. 항목 "통증 부위" / 상담 "오른쪽 아랫배가 아파요." -> confirmed
3. 항목 "발열 여부" / 상담 "열은 없어요." -> confirmed
4. 항목 "기침 지속기간" / 상담 "기침해요." -> unconfirmed
5. 항목 "기침 지속기간" / 상담 "일주일째 기침해요." -> confirmed
6. 항목 "관절의 열감" / 상담 "무릎이 아파요." -> unconfirmed
7. 항목 "관절의 열감" / 상담 "무릎이 뜨거워요." -> confirmed

## 그 밖에 반드시 지킬 규칙

1. 질병을 진단하지 않는다.
2. 응급 여부를 판단하지 않는다.
3. 의뢰 여부를 판단하지 않는다.
4. 치료나 약물을 추천하지 않는다.
5. 항목의 중요도나 위험도를 판단하지 않는다.
6. 아래 제공된 항목 외의 항목을 만들지 않는다. 새로운 문진 질문을 만들지
   않는다.
7. 각 항목을 독립적으로 판단한다. 한 항목의 판정을 다른 항목의 근거로
   쓰지 않는다.
8. 아래 제공된 모든 항목에 대해 정확히 하나씩 결과를 반환한다.
9. 상담 메모를 요약하거나 바꾸어 해석하지 않는다.
10. 각 항목의 category 값(history, red_flag)은 데이터셋의 분류 라벨일 뿐이며
    항목의 중요도나 위험도를 뜻하지 않는다. category를 근거로 판정을 바꾸지
    않는다.
11. 하나의 상담 문장이 여러 항목의 정보를 실제로 담고 있다면 그 항목들은
    모두 confirmed로 판정해도 된다. 다만 그 문장이 실제로 각 항목이 요구하는
    정보를 담고 있을 때만 그렇다.

문진 항목:
{items}
"""


USER_PROMPT_TEMPLATE = """\
상담 메모:
\"\"\"
{consultation_text}
\"\"\"

위 상담 메모를 기준으로 각 문진 항목의 확인 여부를 판정하라."""


class QuestionnaireError(EumcareError):
    """F2 문진 점검 실패."""


class QuestionnaireInputError(QuestionnaireError, InvalidInputError):
    """F2 입력이 잘못된 경우 (빈 상담 메모, 문진 항목 없음 등)."""


class QuestionnaireValidationError(QuestionnaireError, ResultValidationError):
    """LLM 응답이 데이터셋 제약을 어긴 경우."""


class QuestionnaireQuotaError(QuestionnaireError, QuotaExceededError):
    """F2 중 Gemini 할당량/rate limit 초과."""


# ── 문진 항목 구성 (순수 함수, API 불필요) ────────────────────────────


def load_questionnaire(dataset: dict[str, Any]) -> list[dict[str, Any]]:
    """데이터셋에서 문진 항목을 데이터셋 순서 그대로 꺼낸다.

    history_items 다음에 check_items가 온다. 항목 dict은 원본 그대로이며
    metadata를 포함한다. metadata는 후처리에서만 쓰이고 프롬프트에는 절대
    들어가지 않는다(build_f2_prompt 참조).
    """
    items = list(dataset.get("history_items", [])) + list(dataset.get("check_items", []))
    if not items:
        raise QuestionnaireInputError(
            f"문진 항목이 없습니다: {dataset['chief_complaint']['id']}"
        )
    logger.info(
        "문진 항목 수: %d (%s)", len(items), dataset["chief_complaint"]["id"]
    )
    return items


def project_item(item: dict[str, Any]) -> dict[str, Any]:
    """LLM에 보낼 5개 필드만 남긴다.

    화이트리스트 방식이므로 metadata, source, source_role, needs_review,
    observation_group 등은 구조적으로 빠진다.
    """
    return {field: item.get(field, "") for field in MODEL_INPUT_FIELDS}


def build_f2_prompt(
    consultation_text: str, questionnaire_items: list[dict[str, Any]]
) -> tuple[str, str]:
    """(system_instruction, user_prompt)를 만든다.

    상담 메모는 요약하거나 수정하지 않고 원문 그대로 넣는다.
    """
    lines: list[str] = []
    for item in questionnaire_items:
        projected = project_item(item)
        lines.append(
            f"[{projected['id']}] ({projected['category']} / {projected['check_method']})"
        )
        lines.append(f"    항목: {projected['item']}")
        lines.append(f"    설명: {projected['description']}")

    system_instruction = SYSTEM_INSTRUCTION.format(items="\n".join(lines))
    user_prompt = USER_PROMPT_TEMPLATE.format(consultation_text=consultation_text)
    return system_instruction, user_prompt


def build_f2_schema(questionnaire_items: list[dict[str, Any]]) -> dict[str, Any]:
    """response_json_schema용 스키마. 실제 item ID만 enum으로 허용한다."""
    item_ids = [item["id"] for item in questionnaire_items]
    return {
        "type": "object",
        "properties": {
            "items": {
                "type": "array",
                "minItems": len(item_ids),
                "maxItems": len(item_ids),
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "string", "enum": item_ids},
                        "status": {"type": "string", "enum": list(ITEM_STATUSES)},
                    },
                    "required": ["id", "status"],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["items"],
        "additionalProperties": False,
    }


def build_f2_fallback_schema(
    questionnaire_items: list[dict[str, Any]],
) -> dict[str, Any]:
    """response_schema(OpenAPI 서브셋)용 스키마.

    id enum이 빠지므로 이 경로에서는 validate_f2_result()가 유일한 방어선이다.
    """
    return {
        "type": "object",
        "properties": {
            "items": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "string"},
                        "status": {"type": "string", "enum": list(ITEM_STATUSES)},
                    },
                    "required": ["id", "status"],
                    "propertyOrdering": ["id", "status"],
                },
            }
        },
        "required": ["items"],
    }


# ── 결과 검증 (순수 함수, API 불필요) ─────────────────────────────────


def validate_f2_result(
    result: Any, questionnaire_items: list[dict[str, Any]]
) -> dict[str, str]:
    """LLM 응답을 검증하고 {item_id: status} 로 정규화한다.

    검증 항목:
      1. 반환된 id가 실제 데이터셋 item ID인가
      2. 중복 ID가 있는가
      3. 데이터셋의 모든 item이 반환되었는가
      4. status가 허용된 3개 중 하나인가
      5. 없는 item을 만들어내지 않았는가

    Raises:
        QuestionnaireError: 위 조건 중 하나라도 어긋난 경우.
    """
    if not isinstance(result, dict):
        raise QuestionnaireValidationError(f"응답이 객체가 아닙니다: {type(result).__name__}")

    entries = result.get("items")
    if not isinstance(entries, list):
        raise QuestionnaireValidationError("응답에 items 배열이 없습니다.")

    expected_ids = [item["id"] for item in questionnaire_items]
    resolved: dict[str, str] = {}
    seen: list[str] = []

    for entry in entries:
        if not isinstance(entry, dict):
            raise QuestionnaireValidationError(f"items 원소가 객체가 아닙니다: {entry!r}")
        item_id = entry.get("id")
        status = entry.get("status")
        if status not in ITEM_STATUSES:
            raise QuestionnaireValidationError(
                f"허용되지 않은 status: {status!r} (item {item_id!r})"
            )
        seen.append(item_id)
        resolved[item_id] = status

    duplicates = sorted({i for i in seen if seen.count(i) > 1})
    if duplicates:
        raise QuestionnaireValidationError(f"중복된 item ID: {', '.join(map(str, duplicates))}")

    expected_set, got_set = set(expected_ids), set(resolved)
    unknown = sorted(got_set - expected_set)
    if unknown:
        raise QuestionnaireValidationError(
            f"데이터셋에 없는 item ID: {', '.join(map(str, unknown))}"
        )
    missing = sorted(expected_set - got_set)
    if missing:
        raise QuestionnaireValidationError(f"판정이 누락된 item ID: {', '.join(missing)}")

    # LLM이 준 순서는 믿지 않고 데이터셋 순서로 재구성한다.
    return {item_id: resolved[item_id] for item_id in expected_ids}


# ── API 호출 ──────────────────────────────────────────────────────────


def _call_llm(
    client: Any,
    system_instruction: str,
    user_prompt: str,
    questionnaire_items: list[dict[str, Any]],
) -> str:
    """F2용 Gemini 호출. 실패는 QuestionnaireError 계열로 바꿔 던진다.

    호출 횟수 제한과 429 처리는 call_gemini_json()이 담당한다.
    """
    try:
        return call_gemini_json(
            client,
            system_instruction,
            user_prompt,
            build_f2_schema(questionnaire_items),
            build_f2_fallback_schema(questionnaire_items),
        )
    except QuotaExceededError as exc:
        raise QuestionnaireQuotaError(str(exc)) from exc
    except GeminiAPIError as exc:
        raise QuestionnaireError(str(exc)) from exc


def check_questionnaire(
    consultation_text: str,
    questionnaire_items: list[dict[str, Any]],
    client: Any | None = None,
) -> dict[str, str]:
    """상담 메모와 문진 항목을 비교해 원시 판정을 반환한다.

    Returns:
        {item_id: "confirmed"|"unconfirmed"|"uncertain"} (데이터셋 순서)
    """
    if not consultation_text or not consultation_text.strip():
        raise QuestionnaireInputError("상담 메모가 비어 있습니다.")
    if not questionnaire_items:
        raise QuestionnaireInputError("문진 항목이 비어 있습니다.")

    if client is None:
        client = create_client()

    system_instruction, user_prompt = build_f2_prompt(
        consultation_text, questionnaire_items
    )
    logger.debug("모델: %s / 문진 항목 %d개", MODEL_NAME, len(questionnaire_items))

    content = _call_llm(client, system_instruction, user_prompt, questionnaire_items)

    try:
        parsed = json.loads(content)
    except json.JSONDecodeError as exc:
        raise QuestionnaireValidationError(
            f"응답 JSON 파싱 실패: {exc.msg} / 원문: {content[:200]}"
        ) from exc

    raw_results = validate_f2_result(parsed, questionnaire_items)
    counts = {status: 0 for status in ITEM_STATUSES}
    for status in raw_results.values():
        counts[status] += 1
    logger.info("F2 raw result 완료: %d개 항목", len(raw_results))
    logger.info("F2 confirmed items: %d", counts["confirmed"])
    logger.info("F2 unconfirmed items: %d", counts["unconfirmed"])
    logger.info("F2 uncertain items: %d", counts["uncertain"])
    logger.debug(
        "confirmed 항목: %s",
        ", ".join(i for i, s in raw_results.items() if s == "confirmed") or "-",
    )
    return raw_results


def run_f2(
    consultation_text: str,
    dataset: dict[str, Any],
    client: Any | None = None,
) -> dict[str, Any]:
    """F2 전체 파이프라인.

    문진 항목 로드 -> Gemini 판정 -> observation_group 후처리 ->
    완결성 계산 순으로 진행한다. 원시 판정(raw_items)은 그대로 보존한다.

    Returns:
        {raw_items, display_items, total_items, confirmed_items,
         unconfirmed_items, uncertain_items, completeness_percent}
    """
    questionnaire_items = load_questionnaire(dataset)
    raw_results = check_questionnaire(
        consultation_text, questionnaire_items, client=client
    )

    display_items = apply_observation_groups(raw_results, questionnaire_items)
    summary = calculate_completeness(display_items)

    return {
        "raw_items": [
            {"id": item_id, "status": status} for item_id, status in raw_results.items()
        ],
        "display_items": display_items,
        **summary,
    }


def run_f2_only(
    consultation_text: str,
    chief_complaint_id: str,
    datasets: dict[str, dict[str, Any]],
    client: Any | None = None,
) -> dict[str, Any]:
    """F1을 건너뛰고 지정한 주호소로 바로 F2를 실행한다.

    F1 호출을 생략하므로 F2만 반복 검증할 때 API 호출이 절반으로 줄어든다.
    테스트와 디버깅 용도이며, 실제 상담 흐름에서는 run_f2()를 F1 결과와 함께
    사용한다.

    Args:
        consultation_text: 상담 메모 원문.
        chief_complaint_id: 사용할 주호소 ID. 반드시 datasets에 있어야 한다.
        datasets: load_datasets()가 반환한 dict.
        client: 테스트용 주입 지점.

    Raises:
        QuestionnaireInputError: 없는 주호소 ID인 경우.
    """
    if chief_complaint_id not in datasets:
        raise QuestionnaireInputError(
            f"데이터셋에 없는 chief_complaint_id: {chief_complaint_id!r} "
            f"(가능한 값: {', '.join(datasets)})"
        )
    logger.info("F2-only 실행: %s (F1 호출 생략)", chief_complaint_id)
    return run_f2(consultation_text, datasets[chief_complaint_id], client=client)
