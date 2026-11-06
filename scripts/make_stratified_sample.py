"""층화 샘플 생성: 불량 전체 + 정상 무작위 20만 개 (EDA 계획 3.2절, 3-2 문서 3절).

입력: DATA/raw/bosch_parquet/train_numeric.parquet, train_date.parquet
출력: DATA/interim/eda/ (Git 무시)
  sample_ids.parquet            Id, Response, stratum, weight (원본 행 순서 유지)
  sample_stratified.parquet     샘플 Id의 train_numeric 전체 컬럼
  sample_date_stratified.parquet 샘플 Id의 train_date 전체 컬럼 (같은 Id, 같은 순서)
  sample_manifest.json          시드, 크기, 검증 결과

정상 샘플은 numpy.random.default_rng(SEED)로 정렬된 정상 Id에서 비복원 추출한다
(polars 버전과 무관하게 재현). 샘플로 비율을 계산할 때는 weight를 쓰거나 전체 집계를 쓴다.

사용: .venv/bin/python scripts/make_stratified_sample.py
"""
import json
import resource
import time
from pathlib import Path

import numpy as np
import polars as pl

SEED = 20261105
N_NORMAL = 200_000
PQ = Path("DATA/raw/bosch_parquet")
OUT = Path("DATA/interim/eda")


def main() -> None:
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    ids = pl.scan_parquet(PQ / "train_numeric.parquet").select("Id", "Response").collect()
    if ids["Response"].null_count():
        raise SystemExit("Response에 null이 있어 층화 기준을 정할 수 없음")
    normal_all = np.sort(ids.filter(pl.col("Response") == 0)["Id"].to_numpy())
    n_defect = int((ids["Response"] == 1).sum())
    n_normal_total = len(normal_all)
    rng = np.random.default_rng(SEED)
    picked = rng.choice(normal_all, size=N_NORMAL, replace=False)
    weight_normal = n_normal_total / N_NORMAL

    selected = pl.Series("Id", picked, dtype=pl.Int64)
    sample_ids = (ids.filter((pl.col("Response") == 1) | pl.col("Id").is_in(selected.implode()))
                  .with_columns(
                      pl.when(pl.col("Response") == 1).then(pl.lit("defect")).otherwise(pl.lit("normal")).alias("stratum"),
                      pl.when(pl.col("Response") == 1).then(pl.lit(1.0)).otherwise(pl.lit(weight_normal)).alias("weight")))
    sample_ids.write_parquet(OUT / "sample_ids.parquet")
    keep = sample_ids["Id"]
    print(f"sample_ids: {sample_ids.height:,}행 (불량 {n_defect:,}, 정상 {N_NORMAL:,}), 정상 가중치 {weight_normal:.4f}", flush=True)

    outputs = {"sample_stratified.parquet": "train_numeric.parquet",
               "sample_date_stratified.parquet": "train_date.parquet"}
    info = {}
    for out_name, src in outputs.items():
        pl.scan_parquet(PQ / src).filter(pl.col("Id").is_in(keep.implode())).sink_parquet(
            OUT / out_name, compression="zstd", row_group_size=50_000)
        got = pl.scan_parquet(OUT / out_name)
        n_rows = got.select(pl.len()).collect().item()
        got_ids = got.select("Id").collect()["Id"]
        info[out_name] = {
            "source": src, "rows": n_rows, "columns": len(got.collect_schema().names()),
            "bytes": (OUT / out_name).stat().st_size,
            "ids_equal_sample_ids_in_order": bool(got_ids.equals(keep)),
        }
        print(f"{out_name}: {n_rows:,}행, {info[out_name]['columns']}컬럼, "
              f"{info[out_name]['bytes']/2**30:.3f} GiB, Id·순서 일치={info[out_name]['ids_equal_sample_ids_in_order']}",
              flush=True)
    resp = (pl.scan_parquet(OUT / "sample_stratified.parquet").group_by("Response").len().sort("Response")
            .collect())
    manifest = {
        "seed": SEED, "n_normal_requested": N_NORMAL, "n_normal_total": n_normal_total,
        "n_defect": n_defect, "weight_normal": weight_normal, "weight_defect": 1.0,
        "sample_rows": sample_ids.height,
        "sample_response_counts": {str(k): int(v) for k, v in resp.iter_rows()},
        "weighted_total": float(sample_ids["weight"].sum()),
        "files": info,
        "seconds": round(time.time() - t0, 1),
        "peak_rss_mib": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024),
        "polars": pl.__version__, "numpy": np.__version__,
    }
    (OUT / "sample_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    ok = all(v["ids_equal_sample_ids_in_order"] and v["rows"] == sample_ids.height for v in info.values())
    print(f"완료 {manifest['seconds']}s, 피크 RSS {manifest['peak_rss_mib']} MiB, 검증 {'통과' if ok else '실패'}")
    if not ok:
        raise SystemExit("샘플 파일의 Id·순서가 sample_ids와 다름")


if __name__ == "__main__":
    main()
