"""dataset_loader 단위 테스트. Gemini API를 호출하지 않는다."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.config import DEFAULT_DATASET_DIR
from app.dataset_loader import DatasetError, load_datasets


def _write(directory: Path, filename: str, payload: dict | str) -> Path:
    path = directory / filename
    text = payload if isinstance(payload, str) else json.dumps(payload, ensure_ascii=False)
    path.write_text(text, encoding="utf-8")
    return path


def _dataset(cc_id: str, name: str) -> dict:
    return {
        "dataset_version": "0.1",
        "chief_complaint": {"id": cc_id, "name": name, "canonical_term": name, "aliases": []},
    }


def test_loads_real_datasets() -> None:
    """실제 datasets/ 폴더가 정상적으로 로드되고 필수 필드를 갖는다."""
    datasets = load_datasets(DEFAULT_DATASET_DIR)
    assert len(datasets) >= 1
    for cc_id, data in datasets.items():
        cc = data["chief_complaint"]
        assert cc["id"] == cc_id
        assert cc["name"] and cc["canonical_term"]


def test_returns_dict_keyed_by_id(tmp_path: Path) -> None:
    _write(tmp_path, "a.json", _dataset("CCXX_a", "가"))
    _write(tmp_path, "b.json", _dataset("CCYY_b", "나"))

    datasets = load_datasets(tmp_path)

    assert set(datasets) == {"CCXX_a", "CCYY_b"}
    assert datasets["CCXX_a"]["chief_complaint"]["name"] == "가"


def test_ignores_non_json_files(tmp_path: Path) -> None:
    _write(tmp_path, "a.json", _dataset("CCXX_a", "가"))
    (tmp_path / "readme.md").write_text("무시되어야 함", encoding="utf-8")
    (tmp_path / "notes.txt").write_text("무시되어야 함", encoding="utf-8")

    assert set(load_datasets(tmp_path)) == {"CCXX_a"}


def test_optional_fields_may_be_missing(tmp_path: Path) -> None:
    """expression_mapping과 aliases가 없어도 로드된다(CC01이 실제로 그렇다)."""
    _write(
        tmp_path,
        "a.json",
        {"chief_complaint": {"id": "CCXX_a", "name": "가", "canonical_term": "가"}},
    )
    assert "CCXX_a" in load_datasets(tmp_path)


def test_duplicate_id_raises(tmp_path: Path) -> None:
    _write(tmp_path, "a.json", _dataset("CCXX_dup", "가"))
    _write(tmp_path, "b.json", _dataset("CCXX_dup", "나"))

    with pytest.raises(DatasetError, match="중복"):
        load_datasets(tmp_path)


def test_missing_required_field_raises(tmp_path: Path) -> None:
    _write(tmp_path, "a.json", {"chief_complaint": {"id": "CCXX_a", "name": "가"}})

    with pytest.raises(DatasetError, match="canonical_term"):
        load_datasets(tmp_path)


def test_missing_chief_complaint_raises(tmp_path: Path) -> None:
    _write(tmp_path, "a.json", {"dataset_version": "0.1"})

    with pytest.raises(DatasetError, match="chief_complaint"):
        load_datasets(tmp_path)


def test_broken_json_raises(tmp_path: Path) -> None:
    _write(tmp_path, "a.json", '{"chief_complaint": {')

    with pytest.raises(DatasetError, match="JSON 파싱 실패"):
        load_datasets(tmp_path)


def test_missing_directory_raises(tmp_path: Path) -> None:
    with pytest.raises(DatasetError, match="폴더가 없습니다"):
        load_datasets(tmp_path / "nope")


def test_empty_directory_raises(tmp_path: Path) -> None:
    with pytest.raises(DatasetError, match="읽을 수 있는 데이터셋이 없습니다"):
        load_datasets(tmp_path)
