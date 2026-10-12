"""실패 유형 구분용 예외 계층.

호출하는 쪽이 "코드 버그인지, 입력 문제인지, API 문제인지, 할당량 문제인지"를
구분할 수 있어야 한다. 특히 429(RESOURCE_EXHAUSTED)는 일반 API 오류와 분리한다.

기존 ClassificationError / QuestionnaireError 이름은 그대로 두고, 아래
카테고리를 다중 상속으로 함께 부여한다. 따라서 기존 코드의
``except ClassificationError`` 는 계속 동작하며, 필요하면
``isinstance(exc, QuotaExceededError)`` 로 원인을 더 좁힐 수 있다.
"""

from __future__ import annotations


class EumcareError(Exception):
    """이 프로젝트가 의도적으로 발생시키는 모든 오류의 최상위."""


class InvalidInputError(EumcareError):
    """사용자 입력이 잘못된 경우 (빈 상담 메모, 없는 주호소 ID 등)."""


class ResultValidationError(EumcareError):
    """LLM 응답이 스키마/데이터셋 제약을 어긴 경우."""


class GeminiAPIError(EumcareError):
    """Gemini API 호출 실패 (네트워크, 인증, 서버 오류 등)."""


class QuotaExceededError(GeminiAPIError):
    """Gemini API 할당량/rate limit 초과 (HTTP 429, RESOURCE_EXHAUSTED).

    재시도해도 즉시 해결되지 않으므로 같은 요청을 무제한 반복하지 않는다.
    """
