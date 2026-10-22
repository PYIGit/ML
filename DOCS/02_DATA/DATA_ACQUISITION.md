# 데이터 확보 기록: Bosch Production Line Performance

- 작성일: 2026-10-22
- 상태: 완료 (단계 2)
- 관련 문서: `DOCS/ROADMAP.md`, `DOCS/01_SETUP/DIRECTORY_STRUCTURE.md`

## 1. 출처

- 데이터셋: Kaggle 대회 "Bosch Production Line Performance" (2016)
- 데이터 페이지: https://www.kaggle.com/c/bosch-production-line-performance/data
- 대회 개요: https://www.kaggle.com/competitions/bosch-production-line-performance/overview
- 대회 규칙: https://www.kaggle.com/competitions/bosch-production-line-performance/rules
- Kaggle 서버상 파일 생성일: 2019-12-11 (파일 목록 API 기준)

## 2. 이용 조건

- 대회 규칙에 동의한 Kaggle 계정으로만 내려받을 수 있다. 본 프로젝트는 계정 소유자가 규칙에 동의한 뒤 API 토큰으로 내려받았다.
- Kaggle 대회 데이터는 일반적으로 비상업적 목적(대회 참여, 학술 연구, 교육)으로만 사용할 수 있으며 재배포가 허용되지 않는다. 정확한 문구는 위 규칙 페이지에서 계정 소유자가 직접 확인한다.
- 따라서 원본 데이터와 그 직접 변환본은 GitHub에 올리지 않는다. `.gitignore`에 `DATA/` 전체와 `DATA/raw/bosch/`가 이중으로 등록되어 있다.
- 본 프로젝트는 개인 포트폴리오 목적이며 상업적 이용이 아니다.

## 3. 확보 방법

1. Kaggle 계정 설정에서 API 토큰(`KGAT_` 형식)을 발급해 `~/.kaggle/access_token`에 저장 (권한 600, 프로젝트 외부)
2. Kaggle CLI 2.2.4를 `uv tool install kaggle`로 설치. 프로젝트 의존성(`pyproject.toml`)에는 추가하지 않음
3. 다운로드. 실행 위치는 프로젝트 루트.

```bash
kaggle competitions download -c bosch-production-line-performance -p DATA/raw/bosch
```

4. 압축 해제 및 검증. 외부 zip 안에 내부 zip 7개가 들어 있는 2중 구조라 `scripts/extract_bosch.py`로 처리했다. 이 스크립트는 압축을 풀고 각 파일의 크기, SHA-256, 컬럼 수를 `DATA/raw/bosch/MANIFEST.json`에 기록한다.

```bash
.venv/bin/python -I scripts/extract_bosch.py DATA/raw/bosch
```

- 소요 시간: 다운로드 약 1분 20초 (약 9 MB/s), 압축 해제 및 체크섬 약 4분
- 환경 `unzip` 명령이 없어 Python 표준 라이브러리 `zipfile`을 사용했다.

## 4. 파일 목록

### 외부 zip

| 파일 | 크기 | SHA-256 |
|---|---|---|
| bosch-production-line-performance.zip | 680.0 MiB | `2c7a57cf7c03670199cc49f2c237eb86726fdbbe1ca39d25e640dcb8e5f67bd6` |

### 내부 zip

| 파일 | 크기 | SHA-256 |
|---|---|---|
| sample_submission.csv.zip | 1.6 MiB | `c019ce7a687f088cb27ff2be597a292dafca7bcbccd1ab447b0c4add6c63a58a` |
| test_categorical.csv.zip | 18.1 MiB | `acde1b5869fb029d9b5a1e112641ec9e33599d4861e87092448ccb2369303f6c` |
| test_date.csv.zip | 58.8 MiB | `2801007419b705f18fb002aa2d7bf5a67b211b7c9f2aaeeddfaacc48986c8407` |
| test_numeric.csv.zip | 270.3 MiB | `12686d633475dcc6f4a1c408847e027f372191dbbdde49c0b67cd139dfe6aaed` |
| train_categorical.csv.zip | 18.1 MiB | `a02ef93d98fb77edf5f7e417fbfa9a15088a8e6b051ace7a3e5b2326c0dd51ce` |
| train_date.csv.zip | 58.8 MiB | `a9ebea04c935f40b2402ad9dcd354fbb39b235a6ad06e6b86924f29c715fd519` |
| train_numeric.csv.zip | 270.0 MiB | `03147ce75f9bc8c402fa4df518b1fe37c6833f30491dcc7e01c847e35b6e24b8` |

### CSV (압축 해제 결과)

| 파일 | 크기 | 컬럼 수 | SHA-256 |
|---|---|---|---|
| sample_submission.csv | 10.8 MiB | 2 | `13f432ffbf3eb9cb97ce131418679959ee3ae2afe63470ecd3198035ee160ece` |
| test_categorical.csv | 2.49 GiB | 2,141 | `5ee6fa10b5846d00ab6d045911660b624684a5d90d7b19b01ad386772cd10e92` |
| test_date.csv | 2.69 GiB | 1,157 | `44f82b882412df567af80cc4a4bd76b7db2af3ed635776f0b2fec65466c75a61` |
| test_numeric.csv | 1.99 GiB | 969 | `86c0bd0b1fafd1cdc209798dcf5423970430636b5487cb4e880e9a22b441d5da` |
| train_categorical.csv | 2.49 GiB | 2,141 | `39fc61ee2d53668ef6669768ed86fca566c0dcb6c087f8ae1b8fff2e4d2d4bf6` |
| train_date.csv | 2.69 GiB | 1,157 | `bc94f169493dd7b3e98e29ba1cacc6b786259d7f56953ede70105927214ec84b` |
| train_numeric.csv | 1.99 GiB | 970 | `3391821b6f3e291a2a91656a4531fe39a8fd500d301a58d18541815360372bca` |

- 디스크 사용량 합계: 약 16 GB (`DATA/raw/bosch/`, zip 포함)
- 저장 위치: WSL 디스크 (`/home/youngin/projects/ML/DATA/raw/bosch/`). 여유 공간 약 930 GB.
- zip 파일은 재검증을 위해 삭제하지 않고 보관한다.

## 5. 첫 관찰 (헤더만 확인한 사실)

이 절은 파일 헤더 한 줄만 읽어서 확인한 내용이다. 본격적인 분석은 단계 3 EDA에서 한다.

- 컬럼 이름 형식은 `L{라인}_S{스테이션}_{종류}{번호}`이다. 예: `L0_S0_F0`, `L0_S0_D1`, `L0_S1_F25`.
  - numeric 파일은 `F`, date 파일은 `D` 접미사를 쓴다. categorical 파일도 `F`를 쓴다.
  - 이름이 라인과 스테이션 구조를 암시하지만, 실제 공정 흐름은 데이터 분석으로 확인해야 하며 이름만으로 단정하지 않는다.
- `train_numeric.csv`는 970개 컬럼이고 `test_numeric.csv`는 969개다. 차이 1개는 학습 파일에만 있는 타깃 컬럼 `Response`로 추정되며, EDA에서 확인한다.
- categorical은 2,141개, date는 1,157개 컬럼이다.
- 각 CSV가 2~2.7 GiB이므로 메모리 7.6 GiB 환경에서 pandas로 한 번에 읽을 수 없다. 단계 3에서 청크 단위로 읽어 Parquet 등 열 지향 포맷으로 변환하는 것을 우선 검토한다.

## 6. 미결 사항

- 대회 규칙 페이지의 정확한 이용 조건 문구를 계정 소유자가 확인해 이 문서에 반영한다.
- 행 수는 아직 세지 않았다. 단계 3에서 청크 읽기와 함께 확인한다.
