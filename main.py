"""이음케어 F2 MVP - 주호소 추출(F1) + 문진 완결성 점검(F2) CLI.

사용법:
    python main.py "어제부터 머리가 너무 아프고 지끈거려요."
    python main.py                      # 인자를 주지 않으면 대화형
    python main.py --json "..."         # 결과 JSON만 출력
    python main.py --no-f2 "..."        # F1만 수행
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

from app.chief_complaint import ClassificationError, classify_chief_complaint
from app.config import DEFAULT_DATASET_DIR, MODEL_NAME, configure_logging
from app.dataset_loader import DatasetError, load_datasets
from app.errors import (
    EumcareError,
    InvalidInputError,
    QuotaExceededError,
    ResultValidationError,
)
from app.questionnaire import QuestionnaireError, run_f2, run_f2_only

STATUS_MARK = {"confirmed": "✓", "unconfirmed": "□", "uncertain": "?"}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="상담 메모의 주호소를 분류하고 문진 완결성을 점검한다."
    )
    parser.add_argument(
        "text_positional",
        nargs="?",
        help="상담 메모. 생략하면 대화형 입력.",
    )
    parser.add_argument(
        "--datasets", default=str(DEFAULT_DATASET_DIR), help="데이터셋 폴더 경로"
    )
    parser.add_argument("--no-f2", action="store_true", help="F1만 수행한다")
    parser.add_argument(
        "--f2-only",
        action="store_true",
        help="F1(주호소 추출) API 호출을 생략하고 --cc 로 지정한 주호소로 F2만 실행한다",
    )
    parser.add_argument("--cc", help="--f2-only 에서 사용할 chief_complaint_id")
    parser.add_argument("--text", help="상담 메모 (위치 인자 대신 사용 가능)")
    parser.add_argument("--json", action="store_true", help="결과 JSON만 출력한다")
    parser.add_argument("--log-level", default=None, help="DEBUG / INFO / WARNING")
    return parser.parse_args()


def analyze(
    consultation_text: str, datasets: dict[str, Any], *, run_f2_step: bool = True
) -> dict[str, Any]:
    """F1을 수행하고, 주호소가 분류된 경우에만 F2를 이어서 수행한다."""
    result = classify_chief_complaint(consultation_text, datasets)
    result["f2"] = None

    if not run_f2_step or result["status"] != "classified":
        return result

    dataset = datasets[result["chief_complaint_id"]]
    result["f2"] = run_f2(consultation_text, dataset)
    return result


def analyze_f2_only(
    consultation_text: str, chief_complaint_id: str, datasets: dict[str, Any]
) -> dict[str, Any]:
    """F1 API 호출 없이 지정한 주호소로 F2만 실행한다.

    결과 구조는 analyze()와 같아서 화면 출력 코드를 그대로 쓸 수 있다.
    """
    return {
        "status": "classified",
        "chief_complaint_id": chief_complaint_id,
        "chief_complaint_name": datasets[chief_complaint_id]["chief_complaint"]["name"],
        "f2": run_f2_only(consultation_text, chief_complaint_id, datasets),
    }


def _print_human(result: dict[str, Any]) -> None:
    """사용자용 요약 출력. 내부 설계 정보는 표시하지 않는다."""
    if result["status"] != "classified":
        print("주호소: 판단할 수 없음 (uncertain)")
        return

    print(f"주호소: {result['chief_complaint_name']} ({result['chief_complaint_id']})")

    f2 = result.get("f2")
    if not f2:
        return

    print(
        f"문진 완결성: {f2['completeness_percent']}% "
        f"({f2['confirmed_items']}/{f2['total_items']})"
    )

    buckets: dict[str, list[str]] = {"confirmed": [], "unconfirmed": [], "uncertain": []}
    for entry in f2["display_items"]:
        buckets[entry["status"]].append(entry["item"])

    for status, title in (
        ("confirmed", "확인됨"),
        ("unconfirmed", "미확인"),
        ("uncertain", "판단 보류"),
    ):
        if not buckets[status]:
            continue
        print(f"\n{title}")
        for label in buckets[status]:
            print(f"  {STATUS_MARK[status]} {label}")


def _error_label(exc: EumcareError) -> str:
    """실패 유형을 구분해 라벨을 만든다. 429는 일반 API 오류와 분리한다."""
    if isinstance(exc, QuotaExceededError):
        return "할당량 초과"
    if isinstance(exc, InvalidInputError):
        return "입력 오류"
    if isinstance(exc, ResultValidationError):
        return "응답 검증 실패"
    if isinstance(exc, DatasetError):
        return "데이터셋 오류"
    return "API 오류"


def _run_once(text: str, datasets: dict[str, Any], args: argparse.Namespace) -> int:
    try:
        if args.f2_only:
            result = analyze_f2_only(text, args.cc, datasets)
        else:
            result = analyze(text, datasets, run_f2_step=not args.no_f2)
    except ClassificationError as exc:
        print(f"[F1 {_error_label(exc)}] {exc}", file=sys.stderr)
        return 3 if isinstance(exc, QuotaExceededError) else 1
    except QuestionnaireError as exc:
        print(f"[F2 {_error_label(exc)}] {exc}", file=sys.stderr)
        return 3 if isinstance(exc, QuotaExceededError) else 1

    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        _print_human(result)
    return 0


def main() -> int:
    args = _parse_args()
    configure_logging(args.log_level)

    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass  # python-dotenv 없이 환경변수만 써도 동작한다.

    try:
        datasets = load_datasets(args.datasets)
    except DatasetError as exc:
        print(f"[데이터셋 오류] {exc}", file=sys.stderr)
        return 2

    if args.f2_only and not args.cc:
        print("[입력 오류] --f2-only 에는 --cc 가 필요합니다.", file=sys.stderr)
        return 2
    if args.f2_only and args.cc not in datasets:
        print(
            f"[입력 오류] 데이터셋에 없는 주호소: {args.cc!r} "
            f"(가능한 값: {', '.join(datasets)})",
            file=sys.stderr,
        )
        return 2

    if not args.json:
        mode = f"F2-only ({args.cc})" if args.f2_only else "F1 + F2"
        print(
            f"모델: {MODEL_NAME} / 모드: {mode} / "
            f"주호소 {len(datasets)}개: {', '.join(datasets)}\n"
        )

    text = args.text if args.text is not None else args.text_positional
    if text is not None:
        return _run_once(text, datasets, args)

    print("상담 메모를 입력하세요. 빈 줄이나 Ctrl-D 로 종료합니다.")
    while True:
        try:
            text = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if not text:
            return 0
        _run_once(text, datasets, args)


if __name__ == "__main__":
    raise SystemExit(main())
