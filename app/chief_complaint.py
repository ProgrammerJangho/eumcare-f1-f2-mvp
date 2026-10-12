"""상담 메모에서 주호소(chief complaint)를 선택하는 모듈.

LLM provider는 Google Gemini(google-genai SDK) 하나만 사용한다.

이 모듈은 주호소 분류만 수행한다. 진단, 응급 여부, 의뢰 여부, 치료 권고는
다루지 않으며 그런 정보를 LLM에 전달하지도 않는다.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any

from app.errors import (
    EumcareError,
    GeminiAPIError,
    QuotaExceededError,
)
from app.config import (
    API_KEY_ENV,
    MAX_API_ATTEMPTS,
    MAX_ALIASES_PER_CC,
    MAX_EXPRESSIONS_PER_CC,
    MODEL_NAME,
    TEMPERATURE,
)

logger = logging.getLogger(__name__)

RESPONSE_MIME_TYPE = "application/json"

# response_json_schema 지원 여부 캐시. 한 번 미지원으로 확인되면 이후
# 호출에서 첫 번째 시도를 건너뛰어 불필요한 API 호출을 줄인다.
_JSON_SCHEMA_SUPPORTED: bool | None = None

SYSTEM_INSTRUCTION = """\
너는 보건진료소 상담 메모에서 주호소(chief complaint)를 하나 선택하는 분류기다.

역할:
- 상담 메모를 읽고, 아래 후보 목록 중 가장 적절한 주호소 하나를 고른다.
- 그 외의 판단은 하지 않는다.

반드시 지킬 규칙:
1. 질병을 진단하지 않는다. 질환명을 추론해 주호소를 바꾸지 않는다.
2. 환자가 처음 호소한 주된 증상을 기준으로 선택한다.
3. 주호소와 동반 증상을 구분한다. 동반 증상만으로 주호소를 바꾸지 않는다.
4. 증상의 위험도나 심각성을 기준으로 주호소를 임의로 바꾸지 않는다.
5. 의뢰 여부나 응급 여부를 판단하지 않는다.
6. 치료법이나 약물을 제시하지 않는다.
7. 아래 후보 목록에 있는 id 중 하나만 사용한다. 새 id를 만들지 않는다.
8. 어느 주호소인지 명확히 판단할 수 없으면 status를 "uncertain"으로 하고
   chief_complaint_id와 chief_complaint_name을 모두 null로 둔다.
9. 상담 메모를 요약하거나 바꾸어 해석하지 않는다.

후보 판단 보조 자료:
- 각 주호소의 canonical_term은 진료지침 원문 표현이다.
- expressions는 데이터셋 팀이 만든 구어 표현 후보이며 원문 근거가 아니다.
  참고만 하고 절대적인 규칙으로 쓰지 않는다. 목록에 없는 표현이라도
  의미가 맞으면 해당 주호소를 선택할 수 있다.

현재 선택 가능한 주호소:
{candidates}
"""

USER_PROMPT_TEMPLATE = """\
상담 메모:
\"\"\"
{consultation_text}
\"\"\"

위 상담 메모의 주호소를 선택하라."""

UNCERTAIN_RESULT: dict[str, Any] = {
    "status": "uncertain",
    "chief_complaint_id": None,
    "chief_complaint_name": None,
}


class ClassificationError(EumcareError):
    """주호소 분류 실패."""


class ClassificationQuotaError(ClassificationError, QuotaExceededError):
    """F1 중 Gemini 할당량/rate limit 초과."""


# ── 프롬프트 구성 (순수 함수, API 불필요) ─────────────────────────────


def build_candidate_block(datasets: dict[str, dict[str, Any]]) -> str:
    """후보 주호소 목록 문자열을 만든다.

    id, name, canonical_term, aliases, expression_mapping의 표현만 사용한다.
    history_items, check_items, metadata, source, source_role,
    validation_notes, scope_reference 등 임상 정보는 포함하지 않는다.
    expression_mapping이나 aliases가 없는 데이터셋(예: CC01)도 정상 처리한다.
    """
    lines: list[str] = []
    for cc_id, data in datasets.items():
        cc = data["chief_complaint"]
        lines.append(f"- {cc_id}: {cc['name']} (canonical_term: {cc['canonical_term']})")

        aliases = [
            a.get("term")
            for a in cc.get("aliases", [])
            if isinstance(a, dict) and a.get("term")
        ][:MAX_ALIASES_PER_CC]
        if aliases:
            lines.append(f"    aliases: {', '.join(aliases)}")

        expressions = [
            e.get("term")
            for e in data.get("expression_mapping", {}).get("expressions", [])
            if isinstance(e, dict) and e.get("term")
        ][:MAX_EXPRESSIONS_PER_CC]
        if expressions:
            lines.append(f"    expressions: {', '.join(expressions)}")

    return "\n".join(lines)


def build_system_instruction(datasets: dict[str, dict[str, Any]]) -> str:
    """Gemini system_instruction 전체를 만든다."""
    return SYSTEM_INSTRUCTION.format(candidates=build_candidate_block(datasets))


def build_response_schema(datasets: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """response_json_schema용 JSON Schema를 만든다.

    chief_complaint_id는 실제 로드된 id와 null만 허용하도록 enum을 건다.
    그래도 Python 쪽 검증(validate_result)은 생략하지 않는다.
    """
    allowed_ids: list[Any] = list(datasets.keys())
    allowed_names: list[Any] = [d["chief_complaint"]["name"] for d in datasets.values()]
    return {
        "type": "object",
        "properties": {
            "status": {"type": "string", "enum": ["classified", "uncertain"]},
            "chief_complaint_id": {
                "type": ["string", "null"],
                "enum": allowed_ids + [None],
            },
            "chief_complaint_name": {
                "type": ["string", "null"],
                "enum": allowed_names + [None],
            },
        },
        "required": ["status", "chief_complaint_id", "chief_complaint_name"],
        "additionalProperties": False,
    }


def build_fallback_schema(datasets: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """response_schema(OpenAPI 서브셋)용 스키마.

    response_json_schema를 거부하는 모델/백엔드를 위한 대비책이다.
    OpenAPI 서브셋은 type union을 지원하지 않으므로 nullable을 사용하고,
    id enum은 걸지 않는다. 대신 validate_result()가 반드시 검증한다.
    """
    return {
        "type": "object",
        "properties": {
            "status": {"type": "string", "enum": ["classified", "uncertain"]},
            "chief_complaint_id": {"type": "string", "nullable": True},
            "chief_complaint_name": {"type": "string", "nullable": True},
        },
        "required": ["status", "chief_complaint_id", "chief_complaint_name"],
        "propertyOrdering": ["status", "chief_complaint_id", "chief_complaint_name"],
    }


# ── 결과 검증 (순수 함수, API 불필요) ─────────────────────────────────


def validate_result(
    result: Any, datasets: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    """LLM 응답을 검증하고 정규화한다.

    - status는 classified 또는 uncertain만 허용한다.
    - classified면 chief_complaint_id가 실제 로드된 id여야 한다.
    - name은 LLM 값을 믿지 않고 데이터셋 값으로 덮어쓴다.
    - uncertain이면 id와 name을 null로 강제한다.

    Raises:
        ClassificationError: 구조가 어긋나거나 없는 id가 온 경우.
    """
    if not isinstance(result, dict):
        raise ClassificationError(f"응답이 객체가 아닙니다: {type(result).__name__}")

    status = result.get("status")
    if status not in ("classified", "uncertain"):
        raise ClassificationError(f"허용되지 않은 status: {status!r}")

    if status == "uncertain":
        return dict(UNCERTAIN_RESULT)

    cc_id = result.get("chief_complaint_id")
    if cc_id not in datasets:
        raise ClassificationError(
            f"데이터셋에 없는 chief_complaint_id: {cc_id!r} "
            f"(가능한 값: {', '.join(datasets)})"
        )

    # 이름은 Gemini 값을 신뢰하지 않고 데이터셋에서 가져온다.
    canonical_name = datasets[cc_id]["chief_complaint"]["name"]
    if result.get("chief_complaint_name") != canonical_name:
        logger.debug(
            "이름 불일치, 데이터셋 값으로 교정: %r -> %r",
            result.get("chief_complaint_name"),
            canonical_name,
        )

    return {
        "status": "classified",
        "chief_complaint_id": cc_id,
        "chief_complaint_name": canonical_name,
    }


# ── API 호출 ──────────────────────────────────────────────────────────


def create_client() -> Any:
    """Gemini 클라이언트를 만든다. API Key는 환경변수에서만 읽는다."""
    api_key = os.getenv(API_KEY_ENV)
    if not api_key:
        raise ClassificationError(
            f"환경변수 {API_KEY_ENV}가 설정되어 있지 않습니다. "
            f".env.example을 참고해 .env를 만들거나 export 하세요."
        )
    try:
        from google import genai
    except ImportError as exc:
        raise ClassificationError(
            "google-genai 패키지가 설치되어 있지 않습니다. "
            "pip install -r requirements.txt"
        ) from exc

    return genai.Client(api_key=api_key)


def build_gemini_config(
    system_instruction: str, schema: dict[str, Any], *, json_schema_mode: bool
) -> dict[str, Any]:
    """generate_content에 넘길 config dict를 만든다.

    TEMPERATURE가 None이면 temperature 키를 아예 넣지 않는다.
    """
    config: dict[str, Any] = {
        "system_instruction": system_instruction,
        "response_mime_type": RESPONSE_MIME_TYPE,
    }
    if json_schema_mode:
        config["response_json_schema"] = schema
    else:
        config["response_schema"] = schema

    if TEMPERATURE is not None:
        config["temperature"] = TEMPERATURE
    return config


def extract_response_text(response: Any) -> str:
    """Gemini 응답에서 본문 문자열을 꺼낸다."""
    text = getattr(response, "text", None)
    if not text:
        # 안전 필터 등으로 본문이 비는 경우 원인을 남긴다.
        feedback = getattr(response, "prompt_feedback", None)
        raise ClassificationError(
            f"Gemini가 빈 응답을 반환했습니다. prompt_feedback={feedback!r}"
        )
    return text


def is_quota_error(exc: Exception) -> bool:
    """429 / RESOURCE_EXHAUSTED 계열인지 판단한다."""
    message = str(exc)
    code = getattr(exc, "code", None) or getattr(exc, "status_code", None)
    return (
        code == 429
        or "RESOURCE_EXHAUSTED" in message
        or "429" in message
        or "quota" in message.lower()
        or "rate limit" in message.lower()
    )


def is_auth_error(exc: Exception) -> bool:
    """인증/권한 오류인지 판단한다."""
    message = str(exc)
    code = getattr(exc, "code", None) or getattr(exc, "status_code", None)
    return code in (401, 403) or "PERMISSION_DENIED" in message or "API key" in message


def should_retry_with_fallback_schema(exc: Exception) -> bool:
    """response_json_schema 실패를 fallback 스키마로 재시도할지 판단한다.

    스키마를 지원하지 않아서 난 오류일 때만 재시도한다. 할당량 초과나 인증
    오류, 네트워크 오류에서 두 번째 호출을 하는 것은 낭비이고, 429에서는
    상황을 악화시킨다.
    """
    if is_quota_error(exc) or is_auth_error(exc):
        return False
    message = str(exc).lower()
    code = getattr(exc, "code", None) or getattr(exc, "status_code", None)
    return (
        code == 400
        or "invalid_argument" in message
        or "schema" in message
        or "unsupported" in message
        or "not supported" in message
    )


def describe_api_error(exc: Exception) -> str:
    """API 예외를 짧고 읽을 수 있는 문자열로 만든다. API Key는 절대 넣지 않는다."""
    name = type(exc).__name__
    message = str(exc)
    if is_quota_error(exc):
        return (
            f"Gemini API 할당량/rate limit 초과({name}). 잠시 후 다시 시도하세요: "
            f"{message[:300]}"
        )
    if is_auth_error(exc):
        return f"Gemini 인증 실패({name}). {API_KEY_ENV} 값을 확인하세요: {message[:200]}"
    return f"Gemini API 호출 실패({name}): {message[:300]}"


def call_gemini_json(
    client: Any,
    system_instruction: str,
    contents: str,
    primary_schema: dict[str, Any],
    fallback_schema: dict[str, Any],
) -> str:
    """Gemini를 structured output으로 호출하고 본문 문자열을 반환한다.

    F1과 F2가 공유한다.

    1순위: response_json_schema (enum까지 제약)
    2순위: response_schema (OpenAPI 서브셋) - 스키마 미지원 오류일 때만

    호출 횟수는 MAX_API_ATTEMPTS를 넘지 않는다. 429나 인증 오류에서는
    재시도하지 않고 즉시 중단한다.

    Raises:
        QuotaExceededError: 할당량/rate limit 초과.
        GeminiAPIError: 그 밖의 API 호출 실패.
    """
    global _JSON_SCHEMA_SUPPORTED

    attempts: list[tuple[dict[str, Any], bool]] = []
    if _JSON_SCHEMA_SUPPORTED is not False:
        attempts.append((primary_schema, True))
    attempts.append((fallback_schema, False))
    attempts = attempts[:MAX_API_ATTEMPTS]

    last_error: Exception | None = None
    for index, (schema, json_schema_mode) in enumerate(attempts):
        config = build_gemini_config(
            system_instruction, schema, json_schema_mode=json_schema_mode
        )
        try:
            response = client.models.generate_content(
                model=MODEL_NAME, contents=contents, config=config
            )
            if json_schema_mode:
                _JSON_SCHEMA_SUPPORTED = True
            return extract_response_text(response)
        except EumcareError:
            raise
        except Exception as exc:
            last_error = exc
            is_last = index == len(attempts) - 1
            if is_last or not should_retry_with_fallback_schema(exc):
                break
            _JSON_SCHEMA_SUPPORTED = False
            logger.warning(
                "response_json_schema 미지원으로 판단, response_schema로 1회 재시도: %s",
                type(exc).__name__,
            )

    assert last_error is not None
    if is_quota_error(last_error):
        raise QuotaExceededError(describe_api_error(last_error)) from last_error
    raise GeminiAPIError(describe_api_error(last_error)) from last_error


def _call_llm(
    client: Any, system_instruction: str, user_prompt: str, datasets: dict[str, Any]
) -> str:
    """F1용 Gemini 호출. 실패는 ClassificationError 계열로 바꿔 던진다."""
    try:
        return call_gemini_json(
            client,
            system_instruction,
            user_prompt,
            build_response_schema(datasets),
            build_fallback_schema(datasets),
        )
    except QuotaExceededError as exc:
        raise ClassificationQuotaError(str(exc)) from exc
    except GeminiAPIError as exc:
        raise ClassificationError(str(exc)) from exc


def classify_chief_complaint(
    consultation_text: str,
    datasets: dict[str, dict[str, Any]],
    client: Any | None = None,
) -> dict[str, Any]:
    """상담 메모에서 주호소를 분류한다.

    함수 인터페이스는 provider와 무관하다. 호출하는 쪽은 Gemini를 쓰는지
    알 필요가 없다.

    Args:
        consultation_text: 상담 메모 원문. 내용을 수정하지 않고 그대로 전달한다.
        datasets: load_datasets()가 반환한 dict.
        client: 테스트용 주입 지점. None이면 새로 만든다.

    Returns:
        {"status": "classified"|"uncertain",
         "chief_complaint_id": str|None,
         "chief_complaint_name": str|None}

    Raises:
        ClassificationError: API 실패, 파싱 실패, 검증 실패.
    """
    if not consultation_text or not consultation_text.strip():
        logger.info("빈 상담 메모 - API를 호출하지 않고 uncertain 반환")
        return dict(UNCERTAIN_RESULT)

    if not datasets:
        raise ClassificationError("로드된 데이터셋이 없습니다.")

    if client is None:
        client = create_client()

    system_instruction = build_system_instruction(datasets)
    user_prompt = USER_PROMPT_TEMPLATE.format(consultation_text=consultation_text)

    logger.debug("모델: %s / 후보 %d개", MODEL_NAME, len(datasets))
    logger.info("입력: %s", consultation_text.replace("\n", " ")[:120])

    content = _call_llm(client, system_instruction, user_prompt, datasets)

    try:
        parsed = json.loads(content)
    except json.JSONDecodeError as exc:
        raise ClassificationError(
            f"응답 JSON 파싱 실패: {exc.msg} / 원문: {content[:200]}"
        ) from exc

    logger.debug("Gemini 반환 JSON: %s", parsed)
    result = validate_result(parsed, datasets)
    logger.info("최종 결과: %s", result)
    return result
