"""datasets/ 폴더의 주호소 JSON을 읽어 검증하는 모듈.

주호소 목록은 전적으로 JSON 파일이 source of truth이며,
이 모듈은 어떤 chief complaint도 코드에 하드코딩하지 않는다.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from app.config import DEFAULT_DATASET_DIR
from app.errors import EumcareError

logger = logging.getLogger(__name__)

# chief_complaint 안에 반드시 있어야 하는 필드.
REQUIRED_CC_FIELDS: tuple[str, ...] = ("id", "name", "canonical_term")


class DatasetError(EumcareError):
    """데이터셋 로딩/검증 실패."""


def _read_json_file(path: Path) -> dict[str, Any]:
    """JSON 파일 하나를 읽어 dict로 반환한다."""
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise DatasetError(f"파일을 읽을 수 없습니다: {path} ({exc})") from exc

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise DatasetError(
            f"JSON 파싱 실패: {path} (line {exc.lineno}, col {exc.colno}: {exc.msg})"
        ) from exc

    if not isinstance(data, dict):
        raise DatasetError(f"최상위가 객체가 아닙니다: {path}")
    return data


def _validate_dataset(data: dict[str, Any], path: Path) -> str:
    """필수 필드를 검사하고 chief_complaint.id를 반환한다."""
    cc = data.get("chief_complaint")
    if not isinstance(cc, dict):
        raise DatasetError(f"chief_complaint 객체가 없습니다: {path}")

    missing = [f for f in REQUIRED_CC_FIELDS if not cc.get(f)]
    if missing:
        raise DatasetError(
            f"chief_complaint 필수 필드 누락: {path} -> {', '.join(missing)}"
        )

    cc_id = cc["id"]
    if not isinstance(cc_id, str):
        raise DatasetError(f"chief_complaint.id가 문자열이 아닙니다: {path}")
    return cc_id


def load_datasets(dataset_dir: str | Path = DEFAULT_DATASET_DIR) -> dict[str, dict[str, Any]]:
    """datasets 디렉터리의 .json 파일을 모두 읽어 id -> dataset dict로 반환한다.

    Args:
        dataset_dir: 데이터셋 폴더 경로.

    Returns:
        {chief_complaint.id: 데이터셋 전체 dict}

    Raises:
        DatasetError: 폴더가 없거나, JSON이 깨졌거나, 필수 필드가 없거나,
            id가 중복되거나, 읽을 파일이 하나도 없는 경우.
    """
    directory = Path(dataset_dir)
    if not directory.is_dir():
        raise DatasetError(f"데이터셋 폴더가 없습니다: {directory}")

    datasets: dict[str, dict[str, Any]] = {}
    seen_paths: dict[str, Path] = {}

    for path in sorted(directory.glob("*.json")):
        data = _read_json_file(path)
        cc_id = _validate_dataset(data, path)

        if cc_id in datasets:
            raise DatasetError(
                f"chief_complaint.id 중복: '{cc_id}' "
                f"({seen_paths[cc_id].name}, {path.name})"
            )

        datasets[cc_id] = data
        seen_paths[cc_id] = path

    if not datasets:
        raise DatasetError(f"읽을 수 있는 데이터셋이 없습니다: {directory}")

    logger.info("데이터셋 %d개 로드: %s", len(datasets), ", ".join(datasets))
    return datasets


def get_display_name(dataset: dict[str, Any]) -> str:
    """데이터셋에서 주호소 표시 이름을 꺼낸다."""
    return dataset["chief_complaint"]["name"]
