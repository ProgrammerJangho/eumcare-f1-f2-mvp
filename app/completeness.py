"""문진 완결성 계산.

계산은 raw_items가 아니라 display_items 기준이다. observation_group으로
합쳐진 항목은 하나의 관찰로 센다.

기본 모드에서 unconfirmed와 uncertain은 모두 "확인되지 않음"으로 처리한다.
정책을 나중에 바꿀 수 있도록 셈하는 규칙을 인자로 분리해 두었다.
"""

from __future__ import annotations

import logging
from typing import Any

from app.config import COMPLETENESS_CONFIRMED_STATUSES

logger = logging.getLogger(__name__)


def calculate_completeness(
    display_items: list[dict[str, Any]],
    confirmed_statuses: tuple[str, ...] = COMPLETENESS_CONFIRMED_STATUSES,
) -> dict[str, Any]:
    """display_items 기준으로 문진 완결성을 계산한다.

    Args:
        display_items: apply_observation_groups()의 반환값.
        confirmed_statuses: 완결성에 포함할 상태. 기본은 confirmed만.

    Returns:
        {total_items, confirmed_items, unconfirmed_items, uncertain_items,
         completeness_percent}
    """
    total = len(display_items)
    counts = {"confirmed": 0, "unconfirmed": 0, "uncertain": 0}
    for entry in display_items:
        counts[entry["status"]] += 1

    counted = sum(counts[s] for s in confirmed_statuses if s in counts)
    percent = round(counted / total * 100, 1) if total else 0.0

    logger.info("문진 완결성: %s%% (%d/%d)", percent, counted, total)
    return {
        "total_items": total,
        "confirmed_items": counts["confirmed"],
        "unconfirmed_items": counts["unconfirmed"],
        "uncertain_items": counts["uncertain"],
        "completeness_percent": percent,
    }
