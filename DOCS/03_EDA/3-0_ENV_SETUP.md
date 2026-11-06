# 3-0 EDA 실행 환경 준비

- 상태: 완료 (구현 2026-11-01, 문서 정리 2026-11-06)
- 관련: `EDA_PLAN.md` 3.4절·3.5절·8절(결정 1~3, 5), 커밋 `1a696c9`(의존성·빌드 설정·gitignore), `3b0413e`(구조 문서 갱신 포함)
- 참고: 이 문서는 템플릿이 생기기 전에 끝난 단계를 사후에 정리한 것이다. 설계와 컨펌 사항(1~4절)은 `EDA_PLAN.md`의 승인 내용과 커밋 내용을 옮겨 적었다. 구현 당시의 시행착오 기록은 남아 있지 않다.

## 1. 목표

EDA 구현(3-1 이후)에 필요한 의존성과 `src/ml` import 설정을 준비하고, 변환본이 실수로 Git에 올라가지 않게 막는다.

## 2. 확인하려는 것

1. 의존성이 `pyproject.toml`과 `uv.lock`에 기록되어 `uv sync`로 재현되는가
2. 노트북과 테스트에서 `import ml`이 되는가(`src/ml`이 editable로 설치되는가)
3. Parquet 변환본이 Git에 올라가지 않도록 `.gitignore`에 이중 장치가 있는가

## 3. 구현 방법

- `pyproject.toml`에 의존성 추가: 본 의존성 polars, pandas, pyarrow / dev 의존성 matplotlib, ipykernel
- `pyproject.toml`에 빌드 설정 추가: hatchling, wheel 대상 `src/ml`. `uv sync` 시 `ml`이 editable로 설치된다.
- `.gitignore`에 `DATA/raw/bosch_parquet/` 추가. 기존 `DATA/`, `DATA/raw/bosch/`와 함께 이중 장치가 된다.
- 구조 문서(`DIRECTORY_STRUCTURE.md`)에 변환본 위치와 규칙 반영

## 4. 컨펌 사항

| # | 항목 | 옵션 (장단점) | 왜 컨펌이 필요했나 | 선택 |
|---|---|---|---|---|
| 1 | 의존성 구성 | **계획안(polars·pandas·pyarrow + matplotlib·ipykernel)**: EDA에 필요한 기능을 모두 갖춤, pandas와 polars를 함께 가져감 / 최소(polars만): 가볍지만 노트북 시각화와 pandas 탐색을 못 함 / 확장(seaborn, plotly, jupyterlab 추가): 편하지만 지금 쓰임이 없음 | 의존성은 `uv.lock`에 남고 이후 모든 단계에 영향을 준다. 데이터 로딩 도구(polars) 결정은 3-1 문서 4절 #1에서 했고 이 구성은 그 결정을 따른다 | 계획안대로. seaborn, plotly, jupyterlab은 필요해질 때 검토 (2026-11-01) |
| 2 | `src/ml` import 방식 | **hatchling 빌드 설정(editable 설치)**: 노트북과 테스트 어디서나 `import ml`, `pyproject.toml`이 길어짐 / `PYTHONPATH=src`: 설정이 단순하지만 노트북과 테스트마다 따로 설정 | 프로젝트 설정 파일의 구조를 바꾸는 결정이라 되돌리기 번거롭다 | hatchling (2026-11-01) |
| 3 | Git 무시 규칙 | **`DATA/` + `DATA/raw/bosch_parquet/` 이중**: `DATA/` 줄을 풀어도 변환본은 계속 무시, 규칙이 중복됨 / `DATA/`만: 단순하지만 산출물을 올리려고 풀면 변환본이 같이 노출될 위험 | Kaggle 데이터는 재배포가 허용되지 않아 한 번의 실수가 되돌릴 수 없다(올라간 데이터는 지워도 남을 수 있다) | 이중 장치 (2026-11-01) |





---





## 5. 결과

| 항목 | 내용 |
|---|---|
| 의존성 | 본: pandas>=3.1.0, polars>=2.1.0, pyarrow>=26.0.0 / dev: ipykernel>=7.4.0, matplotlib>=3.11.2 |
| 빌드 설정 | `[build-system]` hatchling, `[tool.hatch.build.targets.wheel] packages = ["src/ml"]` |
| lock 파일 | `uv.lock` 갱신(커밋 `1a696c9`, +1,231줄) |
| `.gitignore` | `DATA/raw/bosch_parquet/` 1줄 추가 |
| 구조 문서 | 변환본 위치(`DATA/raw/bosch_parquet/`)와 `raw/` 규칙이 `DIRECTORY_STRUCTURE.md`에 반영됨. 커밋은 3-0 커밋이 아니라 `3b0413e`(EDA 설계 커밋)에 들어 있다 |
| Python | 3.12 (`.python-version`, `requires-python = ">=3.12"`) |

**사후 확인 (2026-11-06)**

- `uv sync` 후 `ml`, polars, pandas, pyarrow, matplotlib, ipykernel을 모두 import할 수 있다. `ml`은 `src/ml/__init__.py`를 가리킨다(editable). 버전은 polars 2.1.0, pandas 3.1.0, pyarrow 26.0.0이다.
- 이 확인은 3-1에서 새로 만든 `.venv` 기준이며, 3-0 당시 환경에서 한 검증이 아니다.

**한계**

- `.gitignore` 이중 장치는 규칙이 파일에 있다는 것까지만 확인했다. 현재 환경은 `DATA`가 심볼릭 링크라 `git check-ignore`가 "beyond a symbolic link" 오류를 내서 규칙의 동작을 검증하지 못했다. 현재는 `.git/info/exclude`의 `/DATA`가 링크를 막고 있다(3-1 문서 6절).
- `src/ml/`에는 `__init__.py`만 있다. `src/ml/data/`, `tests/`, `notebooks/`는 아직 없다.





---





## 6. 실험일지

| 날짜 | 문제 | 해결 아이디어 | 결과 |
|---|---|---|---|
| 2026-11-05 (사후 발견) | `git status`에 9개 파일이 수정됨으로 표시된다. `git diff --ignore-cr-at-eol`로 보면 실제 내용 변경은 없다. 인덱스는 LF, 워킹 트리는 CRLF이고(`git ls-files --eol`), `core.autocrlf`는 설정되어 있지 않다 | 후보 두 가지: `.gitattributes`(`* text=auto eol=lf` 등)로 저장소 차원에서 줄바꿈을 고정 / 로컬 `core.autocrlf=input` 설정. 어느 쪽도 적용하지 않았다 | **미해결(원인 확인까지).** 수정 표시된 파일 중 `pyproject.toml`, `uv.lock`, `.gitignore`는 이 단계의 파일이다. 커밋할 때 줄바꿈이 섞이지 않게 정리 방법을 정해야 한다 |





---





## 7. 다음 단계로 넘기는 것

- 미해결: 워킹 트리 CRLF와 인덱스 LF의 불일치. 정리 방법(`.gitattributes` 또는 `autocrlf`)을 정해야 한다.
- 3-2에서 생성: `src/ml/data/`(컬럼 파싱 함수), `src/ml/stats.py`(Wilson 구간), `tests/`
- 확인 필요: 다른 환경에서 클론했을 때 `DATA`가 심볼릭 링크가 아니라 디렉터리여야 `.gitignore` 이중 장치가 그대로 동작한다.
