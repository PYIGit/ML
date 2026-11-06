# 3-3 결측 구조와 공정 경로(Q4)

- 상태: 설계 중 (2026-11-06). 컨펌 사항 3건 미응답. 구현 전
- 관련: `EDA_PLAN.md` Q4, 3.2절, 3.6절, 5절(3-3) / 선행: `3-2_STRUCTURE_AND_TARGET.md`, `EDA_FINDINGS.md` Q3의 미확인 항목
- 입력: `DATA/raw/bosch_parquet/`의 train 파일 3개, 3-2 산출물(`DATA/interim/eda/`)

## 1. 목표

부품이 어떤 스테이션을 거쳤는지를 결측 패턴으로 복원해(스테이션 방문 행렬) 경로의 종류와 빈도, 경로별 불량률, 라인 간 관계를 확인한다. 그리고 결측을 "스테이션을 거치지 않아 생긴 결측"과 "거친 스테이션 안에서 일부 피처만 빈 결측"으로 나눠 규모를 센다. 단계 5의 결측 처리 전략과 단계 4의 "어디까지 통과한 시점에 예측하는가" 설계의 근거가 된다.

## 2. 확인하려는 것

1. **컬럼별 결측률 분포**: 전부 비어 있는 컬럼, 거의 비어 있는 컬럼의 수(numeric, date, categorical 각각)
2. **행별 값이 있는 컬럼 수 분포**: numeric, date, categorical 각각
3. **스테이션 방문 행렬**: 부품 × 스테이션의 방문 여부를 numeric, date, categorical 세 기준으로 만들고, 스테이션별 방문율을 구한다.
4. **경로**(방문 스테이션 집합): 경로의 종류 수, 경로별 부품 수, 상위 N개 경로가 전체에서 차지하는 비율
5. **경로별 불량률**: 그룹 크기, 불량 수, 불량률, Wilson 구간, 소표본 표시. 3-2의 미확인 항목인 "방문 스테이션 수 12인 그룹(불량률 6.09%)에 공통 경로가 있는가"를 이 표로 확인한다.
6. **라인 간 관계**: 한 부품이 여러 라인을 거치는가, 라인이 배타적인가(라인 조합별 부품 수와 불량률, 라인 쌍 동시 방문)
7. **구조적 결측과 산발적 결측**의 규모: 파일 종류별, 스테이션별 비율
8. **교차 파일 일관성**: numeric, date, categorical 기준 방문이 일치하는가. 불일치의 규모와 패턴. numeric 값이 하나도 없는 582개 행(3-2에서 확인)은 date·categorical 기준으로는 어느 스테이션을 방문했는가
9. **계산 검증**: numeric 기준으로 새로 계산한 방문 수가 3-2의 `row_features.parquet`과 같은지, 경로 표와 결측 분해의 합계가 맞는지, date·categorical 기준 방문이 샘플 행의 독립 계산과 같은지

## 3. 구현 방법

**입력**

| 입력 | 읽는 부분 | 쓰이는 곳 |
|---|---|---|
| `DATA/raw/bosch_parquet/train_numeric.parquet` | `Id`, `Response`, 모든 numeric 컬럼(스테이션 단위로 나눠 스트리밍) | 방문 행렬(numeric 기준), 행별 값이 있는 컬럼 수, 경로별 불량률 |
| `DATA/raw/bosch_parquet/train_date.parquet` | `Id`, 모든 date 컬럼(스테이션 단위로 나눠 스트리밍) | 방문 행렬(date 기준), 행별 값이 있는 컬럼 수 |
| `DATA/raw/bosch_parquet/train_categorical.parquet` | `Id`, 모든 categorical 컬럼(스테이션 단위로, 컬럼 50개 이하씩 나눠 스트리밍) | 방문 행렬(categorical 기준), 행별 값이 있는 컬럼 수 |
| `DATA/interim/eda/column_meta.parquet`(3-2 산출) | 컬럼 이름, 스테이션, 파일 종류, `null_count`, `null_rate` | 결측률 분포, 스테이션별 컬럼 묶음, 구조적·산발적 결측 분해 |
| `DATA/interim/eda/row_features.parquet`(3-2 산출) | `Id`, `Response`, 값이 있는 numeric 컬럼 수, 방문 스테이션 수, 방문 라인 수 | 이번 단계의 numeric 기준 계산이 3-2 값과 같은지 검증 |
| `DATA/interim/eda/station_structure.csv`(3-2 산출) | 스테이션별 numeric·date·categorical 존재 여부 | 교차 파일 비교 대상 스테이션 결정 |

test 파일은 이 단계에서 읽지 않는다(컨펌 사항 #2에서 "train만"을 고르는 경우).

**출력** (파일 이름은 설계안이며 구현 시 바뀔 수 있다. `DATA/interim/eda/` 아래는 모두 Git 무시, `DOCS/03_EDA/figures/` 아래 그림은 추적)

| 출력 | 내용 | 예상 크기 | 쓰이는 곳 |
|---|---|---|---|
| `DATA/interim/eda/station_visits_train.parquet` | `Id`, `Response`, 스테이션별 방문 여부(numeric 50개, date 52개, categorical 34개 = 136개 bool 컬럼), 행별 값이 있는 컬럼 수 3개(numeric, date, categorical). 1,183,747행 | 수십 MB 추정 | 이 단계의 모든 표, 3-4의 시간 분석 |
| `DATA/interim/eda/missing_rate_summary.csv` | 파일 종류별 컬럼 결측률 구간표(100%, 99% 이상, 90% 이상, 50% 이상, 50% 미만) | 수 KB | Q4 보고 |
| `DATA/interim/eda/column_missing_types.parquet` | 피처 컬럼 4,264개별 결측 수, 방문하지 않은 행 수(구조적 결측), 방문한 행 안의 결측 수(산발적 결측), 산발적 결측률 | 1 MB 미만 | Q4 보고, 단계 5 입력 |
| `DATA/interim/eda/routes_numeric.parquet` | numeric 기준 경로별 비트마스크, 스테이션 목록, 스테이션 수, 부품 수, 불량 수, 불량률, Wilson 구간, 소표본 표시 | 경로 수에 따름(구현 후 확정) | Q4 보고, 3-4, 단계 4 |
| `DATA/interim/eda/routes_date.parquet` | date 기준 경로 표(같은 형식) | 경로 수에 따름 | 기준 비교 |
| `DATA/interim/eda/routes_categorical.parquet` | categorical 기준 경로 표(같은 형식) | 경로 수에 따름 | 기준 비교 |
| `DATA/interim/eda/line_combinations.csv` | 라인 조합(4개 라인의 방문 여부 조합)별 부품 수, 불량 수, 불량률, Wilson 구간, 라인 쌍 동시 방문 조건부 확률 | 수 KB | Q4 보고 |
| `DATA/interim/eda/cross_file_consistency.csv` | 두 파일 이상에 있는 스테이션별 numeric·date·categorical 방문의 일치·불일치 부품 수 | 수십 KB | Q4 보고 |
| `DATA/interim/eda/q4_summary.json` | 위 표들의 요약 수치(경로 수, 상위 N 점유율, 구조적·산발적 결측 비율, 일관성 요약), 실행 시간, 피크 메모리 | 수 KB | 이 문서 5절, `EDA_FINDINGS.md` |
| `DOCS/03_EDA/figures/3-3_column_missing_rate.png` | 파일 종류별 컬럼 결측률 분포 | 수십 KB | 보고, `EDA_FINDINGS.md` |
| `DOCS/03_EDA/figures/3-3_nonnull_per_row.png` | 행별 값이 있는 컬럼 수 분포 | 수십 KB | 보고 |
| `DOCS/03_EDA/figures/3-3_station_visit_rate.png` | 스테이션별 방문율(세 기준 비교) | 수십 KB | 보고 |
| `DOCS/03_EDA/figures/3-3_route_cumulative_share.png` | 상위 N개 경로의 누적 점유율 | 수십 KB | 보고 |
| `DOCS/03_EDA/figures/3-3_line_combinations.png` | 라인 조합별 부품 수와 불량률 | 수십 KB | 보고 |

- 코드 산출물(Git 추적): `src/ml/data/visits.py`, `src/ml/data/routes.py`, `src/ml/data/missing.py`, `scripts/eda_3_3_missing_routes.py`, `scripts/plot_3_3_figures.py`(컨펌 사항 #3에서 "스크립트"를 고를 때), `tests/test_visits.py`, `tests/test_routes.py`, `tests/test_missing.py`
- 문서 산출물: 이 문서의 5~7절, `EDA_FINDINGS.md`의 Q4 절

**방법**

코드 구성:

| 파일 | 내용 |
|---|---|
| `src/ml/data/visits.py` | `compute_station_visits`: 스테이션별로 컬럼을 묶음 단위로 나눠 읽어, 스테이션 방문 여부(컬럼 중 하나라도 값이 있으면 방문)와 행별 값이 있는 컬럼 수를 스트리밍으로 계산 |
| `src/ml/data/routes.py` | 방문 열을 비트마스크 경로 키로 변환, 경로 표 생성(부품 수, 불량 수, Wilson 구간), 상위 N 점유율, 라인 조합 표 |
| `src/ml/data/missing.py` | 결측률 구간표, 구조적·산발적 결측 분해, 교차 파일 일관성 표 |
| `scripts/eda_3_3_missing_routes.py` | `visits`, `missing`, `routes`, `consistency` 단계별 실행(단계마다 피크 메모리 기록) |
| `tests/test_visits.py`, `tests/test_routes.py`, `tests/test_missing.py` | 작은 합성 데이터로 계산 규칙 검증 |

항목별 방법:
- **방문 행렬(항목 3)**: 3-2에서 `train_numeric` 전체를 한 번에 계산했더니 피크가 6.5 GiB였다. 이번에는 스테이션 단위로 나누고, 컬럼이 많은 스테이션은 묶음(numeric·date 100개, categorical 50개 이하)으로 나눠 방문 여부는 OR로, 값이 있는 컬럼 수는 합으로 누적한다. 스테이션당 컬럼은 최대 numeric 284개, date 333개, categorical 660개다. 피크 메모리 목표는 3 GiB 이하이며 실제 값은 측정해서 기록한다.
- **결측률 분포(항목 1)**: `column_meta.parquet`의 `null_rate`를 구간으로 나눠 센다. 데이터를 읽지 않는다.
- **행별 값이 있는 컬럼 수(항목 2)**: 방문 행렬을 만들 때 같이 누적한 값으로 분포를 구한다.
- **경로(항목 4, 5)**: 방문 열을 비트마스크(numeric 50비트, date 52비트, categorical 34비트, 모두 64비트 이내)로 묶어 `group_by`로 센다. 경로는 스테이션 집합이며 방문 순서는 다루지 않는다(순서는 3-4, Q5). 경로가 많으면 문서에는 상위 경로와 "부품 수 1,000 미만 경로의 합계"만 싣고, 전체는 Parquet에 둔다. 불량률에는 그룹 크기, 불량 수, Wilson 구간을 붙이고 1,000개 미만은 소표본으로 표시한다. 눈에 띄는 경로는 가설로만 기록한다.
- **라인 간 관계(항목 6)**: 각 라인에 속한 스테이션 중 하나라도 방문하면 그 라인을 방문한 것으로 보고, 4비트 라인 조합(최대 16가지)별로 센다. 라인 쌍마다 P(라인 j 방문 | 라인 i 방문)도 계산한다.
- **구조적·산발적 결측(항목 7)**: 컬럼 c가 속한 스테이션 s에서 방문하지 않은 행 수가 구조적 결측(방문하지 않으면 그 스테이션의 모든 컬럼이 비는 것은 방문 정의에서 항상 성립하므로 정의상 그렇다)이고, 산발적 결측은 `null_count(c) - (방문하지 않은 행 수)`다. 방문 여부는 같은 파일 기준으로 쓴다. 이 값은 데이터를 다시 읽지 않고 `column_meta.parquet`와 방문 행렬만으로 계산한다.
- **교차 파일 일관성(항목 8)**: 세 파일 중 두 개 이상에 있는 스테이션마다 방문 여부의 교차표(둘 다, 한쪽만, 둘 다 아님)를 만들고, 부품 단위로는 공통 스테이션에서 세 기준의 경로가 같은 비율과 가장 흔한 불일치 패턴을 센다. 한 파일에만 있는 스테이션(`L3_S42`, `L3_S46`)은 불일치가 아니라 "비교 불가"로 분리한다.

**검증**
- numeric 기준으로 새로 계산한 값이 있는 컬럼 수, 방문 스테이션 수, 방문 라인 수가 3-2의 `row_features.parquet`과 1,183,747행 전부에서 같아야 한다. 다르면 원인을 찾을 때까지 이후 결과를 쓰지 않는다.
- 경로 표의 부품 수 합이 1,183,747, 불량 수 합이 6,879인지 확인한다. 경로의 스테이션 수가 3-2의 방문 스테이션 수 분포와 맞는지도 본다.
- 구조적 + 산발적 결측이 컬럼별 `null_count`와 같아야 한다. 산발적 결측이 음수가 나오면 방문 계산 오류이므로 원인을 찾는다.
- date와 categorical 기준 방문은 3-2와 같은 방식으로 샘플 행을 numpy로 독립 계산해 비교한다.

**보고.** 표(결측률 구간, 행별 값이 있는 컬럼 수, 스테이션별 방문율, 경로 상위 N, 경로별 불량률, 라인 조합, 구조적·산발적 결측, 교차 파일 일관성)와 그림 5개, 검증 결과, 피크 메모리. 그림은 `dataviz` 스킬의 규칙을 따라 만든다.

## 4. 컨펌 사항

| # | 항목 | 옵션 (장단점) | 왜 컨펌이 필요했나 | 선택 |
|---|---|---|---|---|
| 1 | 경로를 정의하는 기준 파일 | **A. 세 기준을 모두 계산하고 대표 표는 numeric 기준(추천)**: 3-2의 방문 수와 일관되고 기준 간 비교가 쉬움, numeric에 없는 `L3_S42`·`L3_S46`이 대표 경로에서 빠짐 / B. date 기준을 대표로: 52개 스테이션을 모두 포함하고 시각 기록이 통과의 직접 증거일 수 있음, 그 추정은 Q5 전까지 검증되지 않았고 3-2 수치와 달라짐 / C. 합집합(어느 파일에든 값이 있으면 방문): 가장 포괄적, "방문"의 의미가 흐려지고 설명이 복잡해짐 | 경로는 단계 4가 "어디까지 통과한 시점에 예측하는가"를 정할 때 쓰는 기준이고, 세 파일의 스테이션 집합이 달라(50/52/34개) 기준에 따라 경로 수와 불량률 표가 달라진다 | 미응답 (2026-11-06) |
| 2 | test 파일의 방문 행렬을 계산하는 시점 | **A. train만 계산하고 test는 3-6에서(추천)**: 이 단계 범위와 시간이 작고 Q4가 train 기준 질문임 / B. train과 test를 모두 계산: test에만 있는 경로 같은 train·test 차이를 일찍 알 수 있고 3-6이 가벼워짐, 계산과 산출물이 늘고(시간은 train과 비슷할 것으로 예상) test를 분포 비교 용도로만 쓴다는 원칙 아래서 사용 시점이 앞당겨짐 | 계획서가 train·test 비교를 Q9(3-6)에 두었고 test는 분포 비교에만 쓰도록 제한했으므로 test 사용 시점을 정해야 한다 | 미응답 (2026-11-06) |
| 3 | 그림 생성 방식 | **A. 스크립트(`scripts/plot_3_3_figures.py`)로 PNG 생성(추천)**: 실행만으로 재현되고 변경 이력이 깔끔하며 노트북 출력 정리가 필요 없음, 자유 탐색은 별도 / B. 노트북(`notebooks/03_3_missing_routes.ipynb`): 계획서 3.3절의 원안이고 탐색이 편함, 출력 제거 후 커밋해야 하고 이 프로젝트의 첫 노트북이 됨 | 계획서 3.3절은 시각화를 노트북에서 하도록 했는데 3-2까지는 노트북 없이 표만 만들었다. 방침에서 벗어날 수 있고 재현성 기준(완료 기준 4번)에도 영향이 있다 | 미응답 (2026-11-06) |





---





## 5. 결과

(구현 후 작성)





---





## 6. 실험일지

| 날짜 | 문제 | 해결 아이디어 | 결과 |
|---|---|---|---|
| | | | |





---





## 7. 다음 단계로 넘기는 것

(구현 후 작성)
