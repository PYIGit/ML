"""Bosch CSV -> Parquet 무손실 변환 및 검증 (EDA 계획 3.1절).

- CSV 1개당 Parquet 1개, 행 순서 유지.
- 자료형: Id=Int64, Response=Int8, numeric/date 값=Float64, categorical 값=Utf8.
- 엄격 파싱: 숫자로 해석되지 않는 값이 있으면 실패한다(조용히 null로 바꾸지 않음).
- 스트리밍 변환(scan_csv -> sink_parquet).
- 검증(CSV 원문을 csv 모듈로 한 번 훑어 독립적으로 확인):
  1) 행 수, 컬럼 수  2) 모든 행의 필드 수가 헤더와 같은지
  3) 고정 시드 무작위 행 N개의 모든 셀 비교
  4) 무작위 컬럼 K개의 null 수와 합계 비교
- 결과는 out_dir/MANIFEST.json 에 파일별로 병합 기록한다.

사용 (프로젝트 루트에서, 파일 1개씩 실행해 파일별 피크 메모리를 얻는다):
  .venv/bin/python scripts/convert_bosch_to_parquet.py DATA/raw/bosch DATA/raw/bosch_parquet train_numeric
  .venv/bin/python scripts/convert_bosch_to_parquet.py DATA/raw/bosch DATA/raw/bosch_parquet  # 7개 전부
"""
import csv
import json
import math
import random
import resource
import sys
import time
from pathlib import Path

import polars as pl

SEED = 20261105
N_SAMPLE_ROWS = 50
N_AGG_COLS = 8
NAMES = [
    "sample_submission",
    "train_numeric", "train_date", "train_categorical",
    "test_numeric", "test_date", "test_categorical",
]


def read_header(path: Path) -> list[str]:
    with path.open(newline="") as f:
        return next(csv.reader(f))


def build_schema(name: str, header: list[str]) -> dict:
    kind = name.split("_")[-1]
    schema = {}
    for col in header:
        if col == "Id":
            schema[col] = pl.Int64
        elif col == "Response":
            schema[col] = pl.Int8
        elif kind == "categorical":
            schema[col] = pl.Utf8
        else:  # numeric, date
            schema[col] = pl.Float64
    return schema


def count_data_lines(path: Path) -> int:
    n = 0
    last = b""
    with path.open("rb") as f:
        while chunk := f.read(1 << 24):
            n += chunk.count(b"\n")
            last = chunk[-1:]
    if last != b"\n":
        n += 1
    return n - 1  # 헤더 제외


def convert(csv_path: Path, pq_path: Path, schema: dict) -> None:
    # schema 전체를 명시하고 ignore_errors=False(기본)로 엄격 파싱한다.
    lf = pl.scan_csv(csv_path, schema=schema, ignore_errors=False)
    lf.sink_parquet(pq_path, compression="zstd", row_group_size=50_000)


def cell_matches(raw: str, val, dtype) -> bool:
    if raw == "":
        return val is None
    if val is None:
        return False
    if dtype == pl.Utf8:
        return raw == val
    if dtype == pl.Float64:
        return float(raw) == val
    return int(raw) == val


def verify(csv_path: Path, pq_path: Path, header: list[str], schema: dict, n_rows_csv: int) -> dict:
    rng = random.Random(SEED)
    scan = pl.scan_parquet(pq_path)
    n_rows_pq = scan.select(pl.len()).collect().item()
    pq_cols = scan.collect_schema().names()
    res = {
        "rows_csv": n_rows_csv, "rows_parquet": n_rows_pq,
        "cols_csv": len(header), "cols_parquet": len(pq_cols),
        "rows_match": n_rows_csv == n_rows_pq,
        "cols_match": header == pq_cols,
    }
    sample_idx = set(rng.sample(range(n_rows_csv), min(N_SAMPLE_ROWS, n_rows_csv)))
    value_cols = [c for c in header if c != "Id" and schema[c] != pl.Utf8]
    agg_cols = rng.sample(value_cols, min(N_AGG_COLS, len(value_cols))) if value_cols else []
    agg_pos = {c: header.index(c) for c in agg_cols}
    csv_null = {c: 0 for c in agg_cols}
    csv_vals = {c: [] for c in agg_cols}
    sample_rows = {}
    bad_width = 0
    n = 0
    with csv_path.open(newline="") as f:
        r = csv.reader(f)
        next(r)
        for row in r:
            if len(row) != len(header):
                bad_width += 1
            if n in sample_idx:
                sample_rows[n] = row
            for c, p in agg_pos.items():
                v = row[p]
                if v == "":
                    csv_null[c] += 1
                else:
                    csv_vals[c].append(float(v))
            n += 1
    res["rows_counted_by_csv_reader"] = n
    res["rows_with_wrong_field_count"] = bad_width

    # 샘플 행 셀 단위 비교
    mismatches = 0
    for i, raw in sorted(sample_rows.items()):
        got = scan.slice(i, 1).collect().row(0)
        for c, rv, gv in zip(header, raw, got):
            if not cell_matches(rv, gv, schema[c]):
                mismatches += 1
    res["sample_rows_checked"] = len(sample_rows)
    res["sample_cell_mismatches"] = mismatches

    # 컬럼 집계 비교
    agg = {}
    ok = True
    if agg_cols:
        pq_df = scan.select(
            [pl.col(c).null_count().alias(f"{c}__nulls") for c in agg_cols]
            + [pl.col(c).sum().alias(f"{c}__sum") for c in agg_cols]
        ).collect()
        for c in agg_cols:
            s_csv = math.fsum(csv_vals[c])
            s_pq = pq_df[f"{c}__sum"][0] or 0.0
            nulls_pq = pq_df[f"{c}__nulls"][0]
            tol = 1e-9 * max(1.0, abs(s_csv))
            match = (nulls_pq == csv_null[c]) and abs(s_csv - s_pq) <= tol
            ok &= match
            agg[c] = {"nulls_csv": csv_null[c], "nulls_parquet": nulls_pq,
                      "sum_csv": s_csv, "sum_parquet": s_pq, "match": match}
    res["agg_columns"] = agg
    res["agg_all_match"] = ok
    res["passed"] = (res["rows_match"] and res["cols_match"] and n == n_rows_csv
                     and bad_width == 0 and mismatches == 0 and ok)
    return res


def response_values(pq_path: Path) -> dict | None:
    scan = pl.scan_parquet(pq_path)
    if "Response" not in scan.collect_schema().names():
        return None
    df = scan.group_by("Response").len().sort("Response").collect()
    return {str(k): v for k, v in zip(df["Response"], df["len"])}


def main(csv_dir: Path, out_dir: Path, names: list[str]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "MANIFEST.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    manifest["_meta"] = {"seed": SEED, "polars": pl.__version__,
                         "compression": "zstd", "strict_parsing": True}
    for name in names:
        csv_path = csv_dir / f"{name}.csv"
        pq_path = out_dir / f"{name}.parquet"
        header = read_header(csv_path)
        schema = build_schema(name, header)
        print(f"[{name}] 변환 시작 ({len(header)} 컬럼)", flush=True)
        t0 = time.time()
        convert(csv_path, pq_path, schema)
        t_conv = time.time() - t0
        peak_mib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
        print(f"[{name}] 변환 완료 {t_conv:.0f}s, 피크 RSS {peak_mib:.0f} MiB. 검증 시작", flush=True)
        t1 = time.time()
        n_rows = count_data_lines(csv_path)
        res = verify(csv_path, pq_path, header, schema, n_rows)
        entry = {
            "parquet_bytes": pq_path.stat().st_size,
            "csv_bytes": csv_path.stat().st_size,
            "convert_seconds": round(t_conv, 1),
            "verify_seconds": round(time.time() - t1, 1),
            "peak_rss_mib_convert": round(peak_mib),
            "response_counts": response_values(pq_path),
            "verification": res,
        }
        manifest[name] = entry
        manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
        status = "통과" if res["passed"] else "실패"
        print(f"[{name}] 검증 {status}: 행 {res['rows_parquet']:,}, 컬럼 {res['cols_parquet']}, "
              f"Parquet {entry['parquet_bytes']/2**30:.2f} GiB", flush=True)
        if not res["passed"]:
            sys.exit(f"{name} 검증 실패. MANIFEST.json 확인")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    sel = sys.argv[3:] or NAMES
    for s in sel:
        if s not in NAMES:
            sys.exit(f"알 수 없는 이름: {s} (가능: {NAMES})")
    main(Path(sys.argv[1]), Path(sys.argv[2]), sel)
