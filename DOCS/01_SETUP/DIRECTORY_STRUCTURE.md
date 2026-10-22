# 디렉터리 구조

- 작성일: 2026-10-22
- 상태: 확정 (단계 1)
- 관련 문서: `DOCS/ROADMAP.md`

## 1. 전체 구조

```
ML/
├── README                 # 프로젝트 소개 (추후 README.md로 변경 검토)
├── pyproject.toml         # 패키지 이름 ml, Python >= 3.12
├── uv.lock
├── .python-version
├── .gitignore
│
├── DOCS/                  # 모든 문서. 단계별 디렉터리
│   ├── ROADMAP.md         # 전체 로드맵 (단계 공통)
│   ├── DECISIONS.md       # 여러 단계에 걸치는 설계 결정 기록 (필요 시 생성)
│   ├── 01_SETUP/          # 환경 구축: 디렉터리 구조, 도구 선정 근거
│   ├── 02_DATA/           # 데이터 확보: 출처, 이용 조건, 파일 목록, 체크섬
│   ├── 03_EDA/            # EDA 설계 문서, 발견 사항, 그림
│   ├── 04_PROBLEM/        # 문제 정의, 가상 현장 설계, 지표 정의
│   ├── 05_PIPELINE/       # 전처리 규칙
│   ├── 06_MODELING/       # 실험 설정과 결과 기록
│   ├── 07_SIMULATOR/
│   ├── 08_DECISION/
│   ├── 09_BACKEND/
│   ├── 10_FRONTEND/
│   ├── 11_AGENT/
│   └── 12_INTEGRATION/
│
├── DATA/                  # Git 무시. 데이터와 산출물 전부
│   ├── raw/bosch/         # Bosch 원본. 절대 수정하지 않음. 별도 무시 규칙 적용
│   ├── interim/           # 중간 변환 (예: Parquet 변환본, 샘플)
│   ├── processed/         # 학습용 피처 테이블
│   ├── models/            # 저장된 모델 아티팩트
│   ├── experiments/       # 실험 결과 (지표, 설정 스냅샷)
│   └── simulated/         # 시뮬레이터 출력
│
├── src/
│   └── ml/                # Python 패키지. pyproject.toml의 name과 일치
│       ├── data/          # 로딩, 변환 (단계 3~5에서 추가)
│       ├── features/      # 전처리, 피처 생성 (단계 5)
│       ├── models/        # 학습, 평가, 추론 (단계 6)
│       ├── simulator/     # (단계 7)
│       └── decision/      # 위험도, 권고 (단계 8)
│
├── notebooks/             # 탐색용. 단계 접두어 사용 (예: 03_eda_structure.ipynb)
├── scripts/               # 재현 가능한 실행 진입점 (다운로드, 변환, 학습)
└── tests/
```

## 2. 규칙

### 문서 (`DOCS/`)

- 모든 문서는 `DOCS/` 아래에 둔다.
- 단계별 디렉터리 이름은 `NN_이름` 형식이며, 번호는 `ROADMAP.md`의 단계 번호와 일치한다.
- 각 단계 디렉터리에는 그 단계의 설계 문서와 산출물 문서를 함께 둔다.
- 단계 디렉터리는 해당 단계에 진입할 때 만든다. 미리 만들지 않는다.
- 여러 단계에 영향을 주는 결정(도구 선택 등)은 `DOCS/DECISIONS.md`에 기록한다.

### 데이터 (`DATA/`)

- `DATA/` 전체가 `.gitignore`에 포함되어 있다. 데이터, 모델, 실험 결과는 Git에 올라가지 않는다.
- `DATA/raw/bosch/`는 `.gitignore`에 별도로 한 번 더 등록되어 있다. 나중에 산출물을 올리기 위해 `DATA/` 줄을 주석 처리하더라도 Bosch 원본은 계속 무시된다. Kaggle 데이터는 재배포가 허용되지 않기 때문이다.
- `raw/` 아래 파일은 수정하지 않는다. 변환 결과는 `interim/` 또는 `processed/`에 둔다.
- 원본의 출처, 버전, 파일 목록, 체크섬은 `DOCS/02_DATA/`에 기록한다. `DATA/` 안에는 문서를 두지 않는다.

### 코드 (`src/ml/`)

- 코드는 단계가 아니라 기능별로 나눈다. 단계별로 나누면 이후 단계에서 앞 단계 코드를 import할 때 구조가 꼬인다.
- 하위 패키지는 해당 기능이 필요한 단계에서 추가한다.
- Backend와 Frontend는 단계 9~10에서 최상위 `backend/`, `frontend/` 디렉터리로 추가하는 것을 염두에 둔다. 지금은 만들지 않는다.

### 노트북과 스크립트

- 노트북은 탐색용이며, 재현이 필요한 로직은 `src/ml/` 또는 `scripts/`로 옮긴다.
- 노트북 파일명은 단계 접두어를 붙인다 (예: `03_eda_structure.ipynb`).

## 3. 현재 생성된 항목 (2026-10-22)

- `DOCS/ROADMAP.md`, `DOCS/01_SETUP/DIRECTORY_STRUCTURE.md`
- `DATA/` 하위 6개 디렉터리 (비어 있음, Git 무시)
- `src/ml/__init__.py`
- `notebooks/`, `scripts/`, `tests/` (비어 있음. Git은 빈 디렉터리를 추적하지 않으므로 파일이 생기기 전까지 원격에는 나타나지 않는다)

## 4. 미결 사항

- `src/ml`을 import 가능하게 만드는 방식. 현재 `pyproject.toml`은 `uv init --bare`로 만든 가상 프로젝트라 빌드 설정이 없다. 빌드 설정을 추가해 editable 설치하거나, `PYTHONPATH`로 처리하는 방법이 있다. 코드를 처음 작성하는 시점에 결정한다.
- `README`를 `README.md`로 바꿀지 여부.
