"""observation_group 후처리.

observation_group은 데이터셋 팀이 정의한 필드로, 서로 다른 확인 항목이
동일한 관찰 사실을 의미적으로 완전히 공유할 때만 부여된다(양방향 판정).

이 모듈은 LLM 판정이 끝난 뒤에만 동작한다. LLM은 observation_group의
존재 자체를 알지 못하며, 원시 판정(raw_items)은 절대 합쳐지지 않는다.
합치기는 화면 표시(display_items)와 완결성 통계에만 적용된다.
"""

from __future__ import annotations

import logging
from typing import Any

from app.config import GROUP_STATUS_POLICY

logger = logging.getLogger(__name__)


def get_observation_group(item: dict[str, Any]) -> str | None:
    """문진 항목의 observation_group 값을 꺼낸다. 없으면 None."""
    group = item.get("metadata", {}).get("observation_group")
    return group if isinstance(group, str) and group.strip() else None


def resolve_group_status(
    statuses: list[str], policy: str = GROUP_STATUS_POLICY
) -> str:
    """한 그룹에 속한 항목들의 상태를 하나로 합친다.

    두 정책 모두 아래 두 경우는 동일하게 동작한다.
      - 전부 confirmed   -> confirmed
      - 전부 unconfirmed -> unconfirmed

    섞였을 때만 갈린다.
      - "conservative"  : uncertain (기본값)
      - "any_confirmed" : 하나라도 confirmed면 confirmed

    그룹은 양방향 동일 관찰만 묶으므로 상태가 갈린다는 것은 모델이 같은
    관찰을 두고 서로 다른 판정을 냈다는 뜻이다. 기본값은 이를 감추지 않고
    uncertain으로 드러낸다.
    """
    unique = set(statuses)
    if len(unique) == 1:
        return statuses[0]

    if policy == "any_confirmed" and "confirmed" in unique:
        return "confirmed"
    return "uncertain"


def apply_observation_groups(
    raw_results: dict[str, str],
    questionnaire_items: list[dict[str, Any]],
    policy: str = GROUP_STATUS_POLICY,
) -> list[dict[str, Any]]:
    """원시 판정을 화면 표시용 항목 목록으로 변환한다.

    Args:
        raw_results: {item_id: status}. 원시 판정이며 이 함수는 수정하지 않는다.
        questionnaire_items: 데이터셋 순서를 그대로 유지한 문진 항목 목록.
        policy: 그룹 상태 합치기 정책.

    Returns:
        display_items 목록. 데이터셋 순서를 따르며, 같은 그룹의 항목은
        처음 등장한 자리에 하나로 합쳐진다.
        각 원소: {display_key, item, description, status, source_item_ids}
    """
    display: list[dict[str, Any]] = []
    group_index: dict[str, int] = {}

    for item in questionnaire_items:
        item_id = item["id"]
        status = raw_results[item_id]
        group = get_observation_group(item)

        if group is None:
            display.append(
                {
                    "display_key": item_id,
                    "item": item["item"],
                    "description": item.get("description", ""),
                    "status": status,
                    "source_item_ids": [item_id],
                }
            )
            continue

        if group not in group_index:
            group_index[group] = len(display)
            display.append(
                {
                    "display_key": group,
                    "item": group,
                    "description": item.get("description", ""),
                    "status": status,
                    "source_item_ids": [item_id],
                }
            )
            continue

        entry = display[group_index[group]]
        entry["source_item_ids"].append(item_id)
        entry["status"] = resolve_group_status(
            [raw_results[i] for i in entry["source_item_ids"]], policy=policy
        )

    merged = sum(1 for d in display if len(d["source_item_ids"]) > 1)
    logger.info(
        "그룹 후처리 완료: 원시 %d개 -> 표시 %d개 (그룹 %d개)",
        len(questionnaire_items),
        len(display),
        merged,
    )
    return display
