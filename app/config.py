"""이음케어 F2 MVP 설정 상수.

API Key는 이 파일에 두지 않는다. 환경변수 GEMINI_API_KEY에서만 읽는다.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

# ── 모델 ──────────────────────────────────────────────────────────────
# 사용할 수 없는 모델명이면 이 상수만 바꾸면 된다.
MODEL_NAME: str = os.getenv("EUMCARE_MODEL", "gemini-3.6-flash")

# Gemini generation config에 넣을 temperature.
# None이면 파라미터 자체를 보내지 않는다.
TEMPERATURE: float | None = None

# ── 경로 ──────────────────────────────────────────────────────────────
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
DEFAULT_DATASET_DIR: Path = PROJECT_ROOT / "datasets"

# ── 프롬프트에 넣을 정보량 (비용 절감) ─────────────────────────────────
MAX_ALIASES_PER_CC: int = 8
MAX_EXPRESSIONS_PER_CC: int = 8

# ── 환경변수 이름 ─────────────────────────────────────────────────────
API_KEY_ENV: str = "GEMINI_API_KEY"

# ── F2 문진 완결성 ────────────────────────────────────────────────────
# LLM에 전달할 문진 항목 필드. 데이터셋의 usage_policy.model_input_fields와
# 정확히 같아야 한다.
MODEL_INPUT_FIELDS: tuple[str, ...] = (
    "id",
    "category",
    "check_method",
    "item",
    "description",
)

# 항목별 판정 상태.
ITEM_STATUSES: tuple[str, ...] = ("confirmed", "unconfirmed", "uncertain")

# observation_group으로 묶인 항목들의 상태가 서로 다를 때의 처리 정책.
#   "conservative" : 전부 같은 상태일 때만 그 상태, 섞이면 uncertain (기본값)
#   "any_confirmed": 하나라도 confirmed면 confirmed
# 그룹은 양방향 동일 관찰만 묶으므로 상태가 갈리면 모델이 스스로 모순된
# 것이다. 기본값은 이를 uncertain으로 드러내는 보수적 정책을 쓴다.
GROUP_STATUS_POLICY: str = "conservative"

# 완결성 계산에서 confirmed로 셀 상태. unconfirmed와 uncertain은
# "확인되지 않음"으로 처리한다.
COMPLETENESS_CONFIRMED_STATUSES: tuple[str, ...] = ("confirmed",)

# ── API 호출 제어 ─────────────────────────────────────────────────────
# 한 번의 판정 요청에 허용하는 최대 API 호출 횟수.
# 1회는 response_json_schema, 필요하면 1회 더 response_schema로 재시도한다.
# 429(할당량 초과)나 인증 오류에서는 재시도하지 않고 즉시 중단한다.
MAX_API_ATTEMPTS: int = 2

# 주호소 분류에 필요하지 않아 LLM에 절대 전달하지 않는 키.
# 프롬프트 구성은 화이트리스트 방식이지만, 테스트에서 유출 여부를 검사하기
# 위해 명시적으로 남겨 둔다.
FORBIDDEN_PROMPT_KEYS: tuple[str, ...] = (
    "history_items",
    "check_items",
    "validation_notes",
    "scope_reference",
    "usage_policy",
    "source_document",
    "metadata",
    "source",
    "source_role",
    "source_flow_outcome",
    "observation_group",
    "red_flag",
)

# F2 프롬프트용 금지 키.
# category는 데이터셋의 model_input_fields에 포함된 필수 전달 필드이고,
# 그 값이 "history" 또는 "red_flag"이므로 red_flag 문자열은 F2 프롬프트에
# 정상적으로 등장한다. 이는 분류 라벨일 뿐이며 위험도나 임상적 의미를
# 전달하는 것이 아니다(F2 system instruction에 그 점을 명시한다).
F2_FORBIDDEN_PROMPT_KEYS: tuple[str, ...] = tuple(
    k for k in FORBIDDEN_PROMPT_KEYS if k != "red_flag"
)


def configure_logging(level: str | None = None) -> None:
    """루트 로거를 설정한다. LOG_LEVEL 환경변수로 조절할 수 있다."""
    resolved = (level or os.getenv("LOG_LEVEL") or "INFO").upper()
    logging.basicConfig(
        level=getattr(logging, resolved, logging.INFO),
        format="%(levelname)s %(name)s: %(message)s",
    )
