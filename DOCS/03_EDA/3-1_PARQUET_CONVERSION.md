# 3-1 Parquet 변환

- 상태: 완료 (구현 2026-11-05, 문서 정리 2026-11-06)
- 관련: `EDA_PLAN.md` 3.1절·8절, 스크립트 `scripts/convert_bosch_to_parquet.py`(커밋 `3963456`), 검증 원본 `DATA/raw/bosch_parquet/MANIFEST.json`(Git 무시)
- 참고: 설계와 컨펌 사항(1~4절)은 `EDA_PLAN.md`에서 승인된 내용을 이 형식으로 옮겨 적은 것이다. 이 템플릿 이전에 진행된 단계라 사후 정리이다.

## 1. 목표

CSV 7개(약 16 GB)를 값이 바뀌지 않는 Parquet으로 한 번 변환해, 이후 EDA에서 필요한 컬럼만 빠르게 읽을 수 있게 한다.

## 2. 확인하려는 것

1. 변환이 무손실인가: 행 수, 컬럼 수와 순서, 셀 값, 컬럼별 null 수와 합계가 CSV와 같은가
2. CSV에 숫자로 해석되지 않는 값이 숨어 있지 않은가(조용히 null로 바뀌지 않는가)
3. 7.6 GiB 메모리에서 변환이 가능한가, 파일별 피크 메모리는 얼마인가

## 3. 구현 방법

- polars `scan_csv`(전체 스키마 명시, `ignore_errors=False`) → `sink_parquet`(zstd, 행 그룹 50,000행). 스트리밍이라 전체를 메모리에 올리지 않는다.
- 자료형: `Id`=Int64, `Response`=Int8, numeric/date 값=Float64, categorical 값=Utf8. CSV의 빈 필드는 null이 된다.
- 검증은 CSV 원문을 `csv` 모듈로 한 번 훑어 변환 결과와 독립적으로 비교한다. 시드는 `20261105`.
  - 행 수, 컬럼 수·순서, 모든 행의 필드 수
  - 무작위 50행의 모든 셀
  - 무작위 8개 컬럼의 null 수와 합계(상대 오차 1e-9)
- 파일 1개씩 실행해 파일별 피크 메모리를 얻는다.

```bash
.venv/bin/python scripts/convert_bosch_to_parquet.py DATA/raw/bosch DATA/raw/bosch_parquet <파일이름>
```

이름을 생략하면 7개 전부 처리한다.

## 4. 컨펌 사항

| # | 항목 | 옵션 (장단점) | 왜 컨펌이 필요했나 | 선택 |
|---|---|---|---|---|
| 1 | 데이터 로딩 도구 | **polars**: 스트리밍 변환과 lazy 집계를 한 도구로 처리, pandas와 API가 다름 / duckdb: SQL로 바로 변환, 후처리는 다른 라이브러리 필요 / pandas+pyarrow: 익숙하지만 청크 코드를 직접 작성하고 느림 | 이후 모든 단계의 코드 기반이 되므로 바꾸기 비싸다 | polars. 노트북 탐색은 pandas(`.to_pandas()`로 연결) (2026-11-01) |
| 2 | numeric/date 자료형 | **Float64**: 값이 그대로 보존, 메모리 2배 / Float32: 메모리 절반, 소수 3자리 값에서 손실 가능 | 변환본이 원본 대용이라 한 번 손실되면 되돌릴 수 없다 | Float64. 다운캐스팅은 단계 5에서 결정 (2026-11-01, 설명 후 승인) |
| 3 | 변환본 위치 | **`DATA/raw/bosch_parquet/`**: "원본의 무손실 변환본"이라는 의미가 드러남 / `DATA/interim/`: 가공물 위치지만 원본과 의미가 섞임 | Git 무시 정책과 데이터 취급 구분이 위치로 결정된다 | `raw/bosch_parquet/` (사용자 결정, 2026-11-01) |
| 4 | 데이터 저장 디스크 | **WSL 홈(`~/projects/ML/DATA`)**: 반복 읽기가 빠르고 여유 952 GB, 프로젝트 밖이라 심볼릭 링크 필요 / `/mnt/c/.../ML/DATA`: 경로가 단순, Windows 디스크를 거쳐 느리고 여유 426 GB | 16 GB를 반복해서 읽는 작업이라 I/O 속도가 전체 EDA 시간에 영향을 준다 | 명시적 선택 없음. 사용자가 응답하지 않아 추천안(WSL 홈)으로 진행하고 보고했으며, 이후 이의 없음 (2026-11-05) |





---





## 5. 결과

| 파일 | 행 수 | 컬럼 수 | CSV | Parquet | 변환 시간 | 피크 메모리 |
|---|---|---|---|---|---|---|
| sample_submission | 1,183,748 | 2 | 10.8 MiB | 1.2 MiB | 0.4초 | 111 MiB |
| train_numeric | 1,183,747 | 970 | 1.99 GiB | 0.30 GiB | 31초 | 2.7 GiB |
| test_numeric | 1,183,748 | 969 | 1.99 GiB | 0.30 GiB | 22초 | 2.7 GiB |
| train_date | 1,183,747 | 1,157 | 2.69 GiB | 0.77 GiB | 31초 | 2.8 GiB |
| test_date | 1,183,748 | 1,157 | 2.69 GiB | 0.77 GiB | 33초 | 2.8 GiB |
| train_categorical | 1,183,747 | 2,141 | 2.49 GiB | 0.04 GiB | 65초 | 6.1 GiB |
| test_categorical | 1,183,748 | 2,141 | 2.49 GiB | 0.04 GiB | 65초 | 6.1 GiB |

- CSV 약 16 GB → Parquet 약 2.3 GB. 피크 메모리는 `ru_maxrss` 기준이며 변환 단계까지의 값이다.

**검증: 7개 파일 모두 통과**

| 검사 | 결과 |
|---|---|
| 행 수 (CSV 줄 수, `csv` 모듈 행 수, Parquet 행 수) | 모두 일치 |
| 컬럼 수와 순서 | 모두 일치 |
| 필드 수가 헤더와 다른 행 | 0개 |
| 무작위 50행의 셀 불일치 | 0개 |
| 무작위 8개 컬럼의 null 수와 합계 | 모두 일치 |
| 숫자로 해석되지 않는 값 | 없음 (엄격 파싱이 실패하지 않음) |

**변환 중 부수적으로 확인된 사실** (공식 답은 3-2에서 확정)

- train은 1,183,747행, test는 1,183,748행이고 `sample_submission`은 test와 같다.
- `train_numeric`의 `Response`는 0이 1,176,868개, 1이 6,879개다(불량률 약 0.58%). 값 종류는 0과 1뿐이다.
- 컬럼명 `L{라인}_S{스테이션}_{F|D}{번호}`는 `Id`, `Response`를 제외한 모든 컬럼에 맞는다. 라인은 0~3이다.
- 스테이션 수는 numeric 50개, date 52개, categorical 34개로 파일마다 다르다.

**한계**

- 셀 값 검증은 50행 샘플이다. 행 수, 필드 수, 집계는 전체 기준이다.
- numeric/date의 NaN·inf는 검증하지 않았다. polars는 NaN을 null로 세지 않아 null 수 비교에 잡히지 않는다.
- categorical 값의 공백·대소문자 변형은 확인하지 않았다.
- Parquet 파일 자체의 해시는 기록하지 않았다.





---





## 6. 실험일지

| 날짜 | 문제 | 해결 아이디어 | 결과 |
|---|---|---|---|
| 2026-11-05 | 환경에 `DATA/`, Kaggle CLI, `uv`, `.venv`, 토큰이 없음 (데이터는 Git에 없음) | `DATA_ACQUISITION.md` 절차를 재현한다. 토큰은 사용자가 발급해 `~/.kaggle/access_token`(권한 600)에 저장하고, `uv`와 CLI를 설치한 뒤 다운로드하고 `extract_bosch.py`로 푼다 | **해결.** 외부 zip 1개, 내부 zip 7개, CSV 7개의 SHA-256이 문서 4절 표와 모두 일치. 토큰이 대화에 평문으로 남아 폐기·재발급을 권고함(처리 여부 미확인) |
| 2026-11-05 | `uv tool install kaggle`로 설치한 최신 CLI(2.2.5)가 `six`를 의존성에 선언하지 않아 `ModuleNotFoundError` | 도구 환경에 `six`를 함께 설치한다 | **해결.** `uv tool install kaggle --with six --force`로 동작. `DATA_ACQUISITION.md`의 2.2.4와 버전이 달라졌으나 그 문서는 수정하지 않음 |
| 2026-11-05 | 데이터를 WSL 홈에 두고 `DATA`를 심볼릭 링크로 연결. `.gitignore`의 `DATA/`는 디렉터리만 무시해 링크가 커밋될 수 있다고 판단 | `.gitignore`는 그대로 두고 로컬 전용 `.git/info/exclude`에 `/DATA`를 추가한다 | **해결(제한적).** `git status`에 `DATA`가 나타나지 않음. 링크 생성 직후의 선제 조치라 추가 전 상태는 확인하지 않음. 로컬 설정이라 다른 곳에서 클론하면 따라오지 않음 |
| 2026-11-05 | categorical 변환 피크 메모리 6.1 GiB, 가용 메모리 7.6 GiB에 근접 | 변환 코드는 유지하고 통과 여부를 확인한다. 이후 분석은 컬럼을 나눠 읽는다 | **통과, 여유 작음.** 두 파일 모두 변환됨. 3-2, 3-5의 categorical 분석은 컬럼 200개 안팎씩 읽는다 |





---





## 7. 다음 단계로 넘기는 것

- 3-2에서 확인: numeric/date의 NaN·inf, categorical 값의 공백·대소문자, 위 5절의 부수적 사실을 공식 답으로 확정
- 미해결: 워킹 트리 CRLF와 인덱스 LF의 불일치로 수정 표시가 뜸. 원인과 후보는 `3-0_ENV_SETUP.md` 6절에 기록했고 정리 방법은 아직 정하지 않았다.
- 미해결: `DATA_ACQUISITION.md`의 저장 경로(`/home/youngin/...`)와 Kaggle CLI 버전(2.2.4)이 현재 환경과 다름. 수정 여부는 미정이다.
