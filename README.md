# 이음케어 F2 MVP — 주호소 자동 추출 모듈

상담 메모를 입력받아 `datasets/`에 있는 주호소 중 하나를 선택한다.
진단, 응급 여부, 의뢰 여부, 치료 권고는 다루지 않는다.

LLM provider: **Google Gemini (`google-genai`)**

## 프로젝트 구조

```
project/
├── datasets/                  # source of truth. 주호소 목록은 여기서만 온다
│   ├── CC01_abdominal_pain.json   # 복통
│   ├── CC02_cough.json            # 기침
│   ├── CC03_headache.json         # 두통
│   ├── CC04_diarrhea.json         # 설사
│   └── CC05_joint_pain.json       # 관절통
├── app/
│   ├── __init__.py
│   ├── config.py              # 모델명, 경로, 상수, 로깅
│   ├── dataset_loader.py      # JSON 로딩 + 검증 (provider 무관, 변경 없음)
│   └── chief_complaint.py     # 프롬프트 구성 + Gemini 호출 + 결과 검증
├── tests/
│   ├── test_dataset_loader.py
│   └── test_chief_complaint.py
├── main.py                    # CLI (provider 무관, 변경 없음)
├── requirements.txt
├── pytest.ini
└── .env.example
```

## 1. 가상환경 생성

```bash
cd project
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
```

## 2. 패키지 설치

```bash
pip install -r requirements.txt
```

## 3. GEMINI_API_KEY 설정

```bash
cp .env.example .env
# .env 를 열어 GEMINI_API_KEY 값을 채운다
```

또는 셸에서 직접:

```bash
export GEMINI_API_KEY="..."      # Windows: set GEMINI_API_KEY=...
```

API Key는 코드나 로그에 절대 남기지 않는다.

## 4. 로컬 테스트 (API 호출 없음)

```bash
pytest -m "not api"
```

## 5. 실제 API 테스트

```bash
pytest -m api        # GEMINI_API_KEY 없으면 자동 skip
pytest               # 전체
```

## 실행

```bash
python main.py "어제부터 머리가 너무 아프고 지끈거려요. 토할 것 같기도 해요."
python main.py                      # 대화형
python main.py --log-level DEBUG "계속 기침하고 열도 나요."
```

출력:

```json
{
  "status": "classified",
  "chief_complaint_id": "CC03_headache",
  "chief_complaint_name": "두통"
}
```

## 모델 변경

`app/config.py`의 `MODEL_NAME` 상수 하나만 바꾸면 된다. 환경변수로도 덮어쓸 수 있다.

```bash
EUMCARE_MODEL=gemini-2.5-pro python main.py "배가 아파요"
```

`config.TEMPERATURE`는 기본값이 `None`이며, 이 경우 generation config에
`temperature`를 아예 넣지 않는다. 숫자로 바꾸면 그때만 전달된다.

## LLM에 전달하는 정보

주호소 분류에 필요한 최소 정보만 화이트리스트 방식으로 보낸다.

| 전달함 | 전달하지 않음 |
|---|---|
| `chief_complaint.id` | `history_items`, `check_items` |
| `chief_complaint.name` | `metadata` (`source_flow_outcome`, `observation_group`) |
| `chief_complaint.canonical_term` | `source`, `source_role`, `validation_notes` |
| `aliases[].term` | `scope_reference`, `usage_policy` |
| `expression_mapping.expressions[].term` | 치료·의뢰·red flag 정보 |

`test_prompt_excludes_clinical_content`와
`test_clinical_content_absent_from_actual_request`가 이 경계를 검사한다.
후자는 system instruction뿐 아니라 실제로 전송되는 요청 payload 전체를 본다.

## 잘못된 ID 방어

1. `response_json_schema`의 `enum`에 실제 로드된 id만 넣어 디코딩 단계에서 제약한다.
2. 모델/백엔드가 `response_json_schema`를 거부하면 `response_schema`(OpenAPI 서브셋)로
   자동 재시도한다. 이 경로에서는 id enum이 빠지므로 3번이 유일한 방어선이 된다.
3. `validate_result()`에서 반드시 다시 검증한다. 없는 id면 `ClassificationError`.
4. `chief_complaint_name`은 Gemini 값을 믿지 않고 데이터셋 값으로 덮어쓴다.

---

# F2 — 문진 완결성 점검

F1이 주호소를 분류하면, 그 주호소 데이터셋의 문진 항목만 사용해 상담 메모에서
각 항목이 확인되었는지 판정한다. F1이 `uncertain`이면 F2는 실행되지 않는다.

```
상담 메모 → F1 주호소 추출 → dataset 선택 → 문진 항목 로드
→ 항목별 confirmed/unconfirmed/uncertain 판정
→ observation_group 후처리 → 완결성 계산
```

## status의 의미

`confirmed`는 **"해당 문진 항목을 확인했다"**는 뜻이지 "그 증상이 있다"는 뜻이
아니다.

| 문진 항목 | 상담 메모 | status |
|---|---|---|
| 발열 | "열은 없었어요" | `confirmed` (확인함) |
| 발열 | "요즘 좀 피곤해요" | `unconfirmed` |
| 관절의 열감 | "무릎이 이상하게 뜨거운 느낌이 있어요" | `uncertain` 가능 |

## 모듈

| 모듈 | 책임 |
|---|---|
| `app/questionnaire.py` | 문진 항목 로드, 프롬프트 구성, Gemini 호출, 결과 검증, `run_f2` |
| `app/observation_group.py` | 그룹 후처리 (raw → display) |
| `app/completeness.py` | 완결성 계산 |

## raw_items와 display_items

원시 판정은 절대 합쳐지지 않는다. 합치기는 화면 표시와 통계에만 적용된다.

```json
{
  "raw_items": [
    {"id": "HA_009", "status": "confirmed"},
    {"id": "HA_R002", "status": "confirmed"}
  ],
  "display_items": [
    {"display_key": "구토", "item": "구토", "status": "confirmed",
     "source_item_ids": ["HA_009", "HA_R002"]}
  ]
}
```

완결성은 `display_items` 기준이다. CC03은 raw 33개 → display 32개로 계산된다.

## 그룹 상태 정책

`config.GROUP_STATUS_POLICY`로 바꾼다.

| 정책 | 전부 confirmed | 전부 unconfirmed | 섞임 |
|---|---|---|---|
| `conservative` (기본) | confirmed | unconfirmed | **uncertain** |
| `any_confirmed` | confirmed | unconfirmed | confirmed |

그룹은 양방향 동일 관찰만 묶으므로, 상태가 갈린다는 것은 모델이 같은 관찰을
두고 서로 다른 판정을 냈다는 뜻이다. 기본값은 이를 감추지 않는다.

## 완결성 계산 정책

`config.COMPLETENESS_CONFIRMED_STATUSES`가 기본 `("confirmed",)`이며,
`unconfirmed`와 `uncertain`은 모두 "확인되지 않음"으로 처리한다.
`calculate_completeness(display_items, confirmed_statuses=...)`로 호출 시점에도
바꿀 수 있다.

## F2 LLM에 전달하는 정보

데이터셋의 `usage_policy.model_input_fields`와 정확히 같은 5개 필드만 보낸다.

```
id, category, check_method, item, description
```

`metadata`(`observation_group`, `source_flow_outcome`), `source`, `source_role`,
`needs_review`, `validation_notes`, `scope_reference`는 전달하지 않는다.

`test_f2_prompt_excludes_clinical_metadata`가 구조적으로 검증한다. 5개 필드만
남긴 항목으로 만든 프롬프트와 원본 항목으로 만든 프롬프트가 문자열까지 같아야
통과한다. 단순 substring 검사로는 부족한데, 그룹 키 `"구토"`가 `HA_R002`의
`item` 값과 글자가 같고 일부 항목은 `source_text`가 `description`에 그대로
포함되어 있어 정상 전달 내용과 구분되지 않기 때문이다.

## 실행

```bash
python main.py "머리가 욱신거리는데 스트레스 받으면 심해지고 쉬면 괜찮아요."
python main.py --json "..."      # 결과 JSON 전체
python main.py --no-f2 "..."     # F1만
```

출력:

```
주호소: 두통 (CC03_headache)
문진 완결성: 9.4% (3/32)

확인됨
  ✓ 정신적·심리적 불안과 긴장 시 악화
  ✓ 휴식 시 호전
  ✓ 구토

미확인
  □ 목 움직임 시 호전
  ...

판단 보류
  ? 눈충혈
```

## 테스트

```bash
pytest -m "not api"                        # 전체 단위 테스트
pytest -m "not api" tests/test_questionnaire.py   # F2만
pytest -m api                              # 실제 Gemini 호출
```


---

# F2 판정 엄격화 (과잉 판정 방지)

## 문제

상담 메모에 상위 증상만 있는데 세부 문진 항목을 `confirmed`로 판정하는 일이
있었다.

```
상담: "어제부터 배가 아팠고 오늘은 더 심해졌어요. 토하기도 했어요."
→ ABD_008 "통증 부위"가 confirmed로 과잉 판정
```

"배가 아프다"는 복통이 있다는 정보이지, 통증이 **어느 부위**인지 확인했다는
정보가 아니다.

## 해결

`app/questionnaire.py`의 system instruction에 규칙과 판정 예시를 넣었다.
데이터셋이나 판정 로직은 건드리지 않았다.

| 문진 항목 | 상담 메모 | status |
|---|---|---|
| 통증 부위 | "배가 아파요." | `unconfirmed` |
| 통증 부위 | "오른쪽 아랫배가 아파요." | `confirmed` |
| 발열 여부 | "열은 없어요." | `confirmed` |
| 기침 지속기간 | "기침해요." | `unconfirmed` |
| 기침 지속기간 | "일주일째 기침해요." | `confirmed` |
| 관절의 열감 | "무릎이 아파요." | `unconfirmed` |
| 관절의 열감 | "무릎이 뜨거워요." | `confirmed` |

수치·기간 조건이 붙은 항목은 상담에 그 기준을 판단할 정보가 있어야 한다.
`14일 이상 지속되는 설사`는 "3일째 설사해요"로 `confirmed`가 되지 않는다.

`uncertain`은 정말 판정이 어려울 때만 쓴다. 정보가 명백히 없으면
`unconfirmed`다.

## F2-only 모드 (API 호출 절감)

F1 호출을 생략하고 지정한 주호소로 바로 F2를 실행한다. F2만 반복 검증할 때
호출이 절반으로 준다.

```bash
python main.py --f2-only --cc CC01_abdominal_pain \
  --text "어제부터 배가 아팠고 오늘은 더 심해졌어요. 토하기도 했어요."
```

```python
from app.questionnaire import run_f2_only
result = run_f2_only(text, "CC01_abdominal_pain", datasets)
```

API 테스트도 전부 `run_f2_only`를 쓴다. §18 케이스 7개 = 호출 7회.

## 429 / 실패 유형 구분

`app/errors.py`에 예외 계층을 두었다. 기존 `ClassificationError` /
`QuestionnaireError` 이름은 그대로라 기존 `except` 절이 계속 동작하고,
필요하면 카테고리로 더 좁힐 수 있다.

```
EumcareError
├── InvalidInputError        빈 상담 메모, 없는 주호소 ID
├── ResultValidationError    LLM 응답이 데이터셋 제약 위반
├── DatasetError             JSON 로딩/검증 실패
└── GeminiAPIError           API 호출 실패
    └── QuotaExceededError   429 / RESOURCE_EXHAUSTED
```

호출 횟수는 `config.MAX_API_ATTEMPTS`(기본 2)를 넘지 않는다. 폴백 재시도는
**스키마 미지원 오류일 때만** 한다. 429·인증 오류·네트워크 오류에서는 즉시
중단한다. 이전 구조는 어떤 예외에서든 두 번째 호출을 했는데, 429에서 그건
상황을 악화시킬 뿐이었다.

`response_json_schema`가 한 번 미지원으로 확인되면 모듈 레벨에 캐시되어
이후 호출은 첫 시도를 건너뛴다.

CLI 종료 코드: `0` 성공 / `1` 일반 실패 / `2` 입력·데이터셋 오류 /
`3` 할당량 초과.

## 로그

```
INFO app.questionnaire: 문진 항목 수: 23 (CC01_abdominal_pain)
INFO app.questionnaire: F2 raw result 완료: 23개 항목
INFO app.questionnaire: F2 confirmed items: 3
INFO app.questionnaire: F2 unconfirmed items: 20
INFO app.questionnaire: F2 uncertain items: 0
INFO app.observation_group: 그룹 후처리 완료: 원시 23개 -> 표시 23개 (그룹 0개)
INFO app.completeness: 문진 완결성: 13.0% (3/23)
```

confirmed 항목 ID 목록은 `--log-level DEBUG`에서만 나온다. 상담 메모 원문은
반복 로그에 남기지 않는다.

## 모델

기본값이 `gemini-3.6-flash`다. `EUMCARE_MODEL` 환경변수로 덮어쓸 수 있다.
