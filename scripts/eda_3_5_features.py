"""3-5 numeric과 categorical 피처 특성 (Q6, Q7). 설계: DOCS/03_EDA/3-5_NUMERIC_AND_CATEGORICAL.md

입력: DATA/raw/bosch_parquet/{train_numeric.parquet, train_categorical.parquet},
      DATA/interim/eda/{row_features.parquet, row_time_features.parquet, station_visits_train.parquet}
출력: DATA/interim/eda/ (Git 무시)
  numeric_column_stats.parquet, numeric_duplicate_columns.csv, duplicate_rows_numeric.parquet,
  l3_s32_value_summary.csv, categorical_column_stats.parquet, categorical_duplicate_columns.csv,
  q6_q7_summary.json

단계(순서대로): numeric_stats -> numeric_dup_cols -> dup_rows -> s32 -> categorical -> verify
사용: .venv/bin/python scripts/eda_3_5_features.py <stage|all>
"""
import json
import resource
import sys
from pathlib import Path

import numpy as np
import polars as pl
import pyarrow.parquet as pq

from ml.data.features import (candidate_groups, classify_categorical, confirm_row_groups, group_identical_columns,
                              series_stats)
from ml.stats import rate_table

PQ = Path("DATA/raw/bosch_parquet")
OUT = Path("DATA/interim/eda")
NUM = PQ / "train_numeric.parquet"
CAT = PQ / "train_categorical.parquet"
SEED = 20261112
N_VERIFY_COLS = 30
# 센티널 후보 기준(후보를 고르는 용도)
SENT_MIN_UNIQUE = 100
SENT_TOP_SHARE = 0.20
SENT_EDGE_SHARE = 0.05
S32_COL = "L3_S32_F3850"


def peak_mib() -> int:
    return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)


def update_summary(key: str, value: dict) -> None:
    p = OUT / "q6_q7_summary.json"
    cur = json.loads(p.read_text()) if p.exists() else {}
    value["peak_rss_mib"] = peak_mib()
    cur[key] = value
    p.write_text(json.dumps(cur, indent=2, ensure_ascii=False))


def feature_columns(path: Path) -> list[str]:
    return [c for c in pq.ParquetFile(path).schema_arrow.names if c not in ("Id", "Response")]


def station_of(col: str) -> str:
    return "_".join(col.split("_")[:2])


def stats_table(path: Path, cols: list[str], chunk: int) -> pl.DataFrame:
    rows = []
    for i in range(0, len(cols), chunk):
        df = pl.read_parquet(path, columns=cols[i:i + chunk])
        for c in df.columns:
            rows.append({"column": c, "station_key": station_of(c), **series_stats(df[c])})
        del df
        print(f"  stats {min(i + chunk, len(cols))}/{len(cols)}", flush=True)
    return pl.DataFrame(rows, infer_schema_length=None)


def stage_numeric_stats() -> None:
    cols = feature_columns(NUM)
    st = stats_table(NUM, cols, 50)
    st = st.with_columns(
        (pl.col("top_count") / pl.col("n_nonnull")).alias("top_share"),
        (pl.col("n_at_min") / pl.col("n_nonnull")).alias("share_at_min"),
        (pl.col("n_at_max") / pl.col("n_nonnull")).alias("share_at_max"),
        (pl.col("n_unique") == 1).alias("single_value"),
    ).with_columns(
        ((pl.col("n_unique") > SENT_MIN_UNIQUE) & (pl.col("top_share") >= SENT_TOP_SHARE)).alias("cand_top_value"),
        ((pl.col("n_unique") > SENT_MIN_UNIQUE)
         & ((pl.col("share_at_min") >= SENT_EDGE_SHARE) | (pl.col("share_at_max") >= SENT_EDGE_SHARE))).alias("cand_edge"),
    )
    st.write_parquet(OUT / "numeric_column_stats.parquet")
    top_vals = (st.group_by("top_value").agg(pl.len().alias("n_columns")).sort("n_columns", descending=True).head(5))
    res = {
        "n_columns": st.height,
        "n_single_value": int(st["single_value"].sum()),
        "min_of_min": float(st["min"].min()), "max_of_max": float(st["max"].max()),
        "n_cols_min_is_minus1": int((st["min"] == -1.0).sum()), "n_cols_max_is_plus1": int((st["max"] == 1.0).sum()),
        "mean_range": [float(st["mean"].min()), float(st["mean"].max())],
        "n_cols_abs_mean_gt_0.01": int((st["mean"].abs() > 0.01).sum()),
        "std_range": [float(st["std"].min()), float(st["std"].max())],
        "std_median": float(st["std"].median()),
        "n_unique_range": [int(st["n_unique"].min()), int(st["n_unique"].max())],
        "sentinel_candidates_top_value": int(st["cand_top_value"].sum()),
        "sentinel_candidates_edge": int(st["cand_edge"].sum()),
        "most_common_top_values": top_vals.to_dicts(),
        "n_cols_top_share_ge_0.5": int((st["top_share"] >= 0.5).sum()),
    }
    update_summary("numeric_stats", res)
    print(json.dumps(res, ensure_ascii=False, indent=1))


def stage_numeric_dup_cols() -> None:
    st = pl.read_parquet(OUT / "numeric_column_stats.parquet").drop("duplicate_of", strict=False)
    sig = {r["column"]: (r["n_nonnull"], r["n_unique"], r["min"], r["max"], r["sum"], r["top_value"], r["top_count"])
           for r in st.to_dicts()}
    cands = candidate_groups(sig)
    groups: list[list[str]] = []
    for g in cands:
        groups += group_identical_columns(pl.read_parquet(NUM, columns=g))
    rows = [{"group": i, "column": c, "is_representative": j == 0} for i, g in enumerate(groups) for j, c in enumerate(g)]
    pl.DataFrame(rows, schema={"group": pl.Int64, "column": pl.String, "is_representative": pl.Boolean}).write_csv(
        OUT / "numeric_duplicate_columns.csv")
    dup_of = {c: g[0] for g in groups for c in g[1:]}
    st = st.with_columns(pl.col("column").replace_strict(dup_of, default=None, return_dtype=pl.String).alias("duplicate_of"))
    st.write_parquet(OUT / "numeric_column_stats.parquet")
    removable = st.filter(pl.col("single_value") | pl.col("duplicate_of").is_not_null())
    res = {
        "candidate_groups": len(cands), "candidate_columns": sum(len(g) for g in cands),
        "confirmed_groups": len(groups), "columns_in_groups": sum(len(g) for g in groups),
        "removable_as_duplicate": len(dup_of),
        "groups_across_stations": sum(len({station_of(c) for c in g}) > 1 for g in groups),
        "group_sizes": sorted((len(g) for g in groups), reverse=True)[:10],
        "n_single_value": int(st["single_value"].sum()),
        "n_removable_total": removable.height, "n_remaining": st.height - removable.height,
    }
    update_summary("numeric_dup_cols", res)
    print(json.dumps(res, ensure_ascii=False, indent=1))
    for g in groups[:15]:
        print("  ", g)


def stage_dup_rows() -> None:
    cols = feature_columns(NUM)
    pf = pq.ParquetFile(NUM)
    hashes, nonnull = [], []
    for i in range(pf.num_row_groups):
        df = pl.from_arrow(pf.read_row_group(i, columns=cols))
        hashes.append(df.hash_rows(seed=1).to_numpy())
        nonnull.append(df.select(pl.sum_horizontal(pl.all().is_not_null())).to_series().to_numpy())
        del df
    h = np.concatenate(hashes)
    has_value = np.concatenate(nonnull) > 0
    n = len(h)
    uniq, inv, cnt = np.unique(h, return_inverse=True, return_counts=True)
    cand = (cnt[inv] > 1) & has_value
    cand_idx = np.flatnonzero(cand)
    print(f"  candidate rows (same hash, has value): {len(cand_idx):,}", flush=True)
    if len(cand_idx) > 350_000:
        raise SystemExit("후보 행이 너무 많다. 나눠서 비교하도록 바꿔야 한다")
    # 후보 행의 값을 모아 직접 비교
    vals = np.empty((len(cand_idx), len(cols)), dtype=np.float64)
    pos = start = 0
    for i in range(pf.num_row_groups):
        nrows = pf.metadata.row_group(i).num_rows
        local = cand_idx[(cand_idx >= start) & (cand_idx < start + nrows)] - start
        if len(local):
            df = pl.from_arrow(pf.read_row_group(i, columns=cols))[local.tolist()]
            vals[pos:pos + len(local)] = df.to_numpy()
            pos += len(local)
            del df
        start += nrows
    gid = confirm_row_groups(h[cand_idx], vals)
    del vals
    feats = pl.read_parquet(OUT / "row_time_features.parquet", columns=["Id", "Response", "t_first"])
    assert feats.height == n
    d = feats.with_row_index("row")[cand_idx.tolist()].with_columns(pl.Series("group", gid)).filter(pl.col("group") >= 0)
    d = d.sort(["group", "row"])
    d.select("group", "Id", "Response", "t_first", "row").write_parquet(OUT / "duplicate_rows_numeric.parquet")
    g = d.group_by("group").agg(
        pl.len().alias("size"), pl.col("Response").sum().alias("k"),
        pl.col("t_first").n_unique().alias("n_times"),
        (pl.col("row").diff().drop_nulls() == 1).all().alias("rows_consecutive"),
        (pl.col("Id").diff().drop_nulls().max()).alias("max_id_gap"),
    )
    mixed = g.filter((pl.col("k") > 0) & (pl.col("k") < pl.col("size")))
    n_def = int(d["Response"].sum())
    res = {
        "candidate_rows_same_hash": int(len(cand_idx)),
        "hash_collisions_rows": int(len(cand_idx) - d.height),
        "n_groups": g.height, "n_parts_in_groups": d.height, "share_of_parts": d.height / n,
        "n_extra_copies": d.height - g.height,
        "group_size_max": int(g["size"].max()), "groups_of_size_2": int((g["size"] == 2).sum()),
        "defects_in_groups": n_def, "defect_rate_in_groups": n_def / d.height,
        "groups_mixed_response": mixed.height, "parts_in_mixed_groups": int(mixed["size"].sum()),
        "groups_all_defect": int((g["k"] == g["size"]).sum()),
        "groups_same_first_time": int((g["n_times"] == 1).sum()),
        "share_groups_same_first_time": float((g["n_times"] == 1).mean()),
        "groups_rows_consecutive_in_train": int(g["rows_consecutive"].sum()),
        "share_groups_rows_consecutive": float(g["rows_consecutive"].mean()),
        "groups_max_id_gap_le_2": int((g["max_id_gap"] <= 2).sum()),
        "share_groups_max_id_gap_le_2": float((g["max_id_gap"] <= 2).mean()),
    }
    # 묶음 안에서 Id가 가장 작은 부품과 나머지의 불량률
    dd = d.with_columns((pl.col("row") == pl.col("row").min().over("group")).alias("is_first"))
    for flag, name in ((True, "first_in_group"), (False, "later_in_group")):
        x = dd.filter(pl.col("is_first") == flag)
        res[name] = {"n": x.height, "k": int(x["Response"].sum()), "rate": float(x["Response"].mean())}
    # 3-4의 "L3_S32를 거치고 L3_S33을 거치지 않은 부품"과 겹치는 정도
    vis = pl.read_parquet(OUT / "station_visits_train.parquet", columns=["Id", "Response", "numeric__L3_S32", "numeric__L3_S33"])
    x = vis.with_columns((pl.col("numeric__L3_S32") & ~pl.col("numeric__L3_S33")).alias("s32_skip"),
                         pl.col("Id").is_in(d["Id"].to_list()).alias("dup"))
    cross = x.group_by(["s32_skip", "dup"]).agg(pl.len().alias("n"), pl.col("Response").sum().alias("k")).sort(["s32_skip", "dup"])
    res["cross_with_s32_skip"] = rate_table(cross, ["s32_skip", "dup"]).to_dicts()
    res["total_defects"] = int(vis["Response"].sum())
    res["share_of_all_defects"] = n_def / res["total_defects"]
    # 묶음에 든 부품이 값이 있는 컬럼 수가 적은 부품인지
    nn = np.concatenate(nonnull)
    res["median_nonnull_cols_in_groups"] = float(np.median(nn[d["row"].to_numpy()]))
    res["median_nonnull_cols_all"] = float(np.median(nn[has_value]))
    update_summary("dup_rows", res)
    print(json.dumps(res, ensure_ascii=False, indent=1))


def stage_s32() -> None:
    v = pl.read_parquet(NUM, columns=["Id", S32_COL, "Response"])
    vis = pl.read_parquet(OUT / "station_visits_train.parquet", columns=["Id", "numeric__L3_S32", "numeric__L3_S33"])
    d = v.join(vis, on="Id").filter(pl.col(S32_COL).is_not_null())
    assert d["numeric__L3_S32"].all()
    t = (d.group_by(["numeric__L3_S33", S32_COL]).agg(pl.len().alias("n"), pl.col("Response").sum().alias("k"))
         .sort(["numeric__L3_S33", S32_COL]))
    rate_table(t, ["numeric__L3_S33", S32_COL]).rename({"numeric__L3_S33": "visited_L3_S33", S32_COL: "value"}).write_csv(
        OUT / "l3_s32_value_summary.csv")
    skip = set(d.filter(~pl.col("numeric__L3_S33"))[S32_COL].unique().to_list())
    both = set(d.filter(pl.col("numeric__L3_S33"))[S32_COL].unique().to_list())
    res = {
        "n_with_value": d.height, "n_skip_s33": int((~d["numeric__L3_S33"]).sum()),
        "n_unique_values": d[S32_COL].n_unique(),
        "values_skip_s33": len(skip), "values_with_s33": len(both), "values_in_both": len(skip & both),
    }
    # 두 집단의 값 분포(중앙값, 사분위)와 값 구간별 불량률
    edges = [-0.05, -0.01, 0.0, 0.01, 0.05]
    for flag, name in ((False, "skip_s33"), (True, "with_s33")):
        g = d.filter(pl.col("numeric__L3_S33") == flag)
        res[f"quartiles_{name}"] = [float(g[S32_COL].quantile(q, "nearest")) for q in (0.25, 0.5, 0.75)]
        binned = (g.with_columns(pl.Series("bin", np.searchsorted(edges, g[S32_COL].to_numpy(), side="left")))
                  .group_by("bin").agg(pl.len().alias("n"), pl.col("Response").sum().alias("k")).sort("bin"))
        res[f"rate_by_value_range_{name}"] = rate_table(binned, ["bin"]).to_dicts()
    res["value_range_edges"] = edges
    update_summary("s32", res)
    print(json.dumps(res, ensure_ascii=False, indent=1))


def stage_categorical() -> None:
    cols = feature_columns(CAT)
    st = stats_table(CAT, cols, 100)
    n_rows = pq.ParquetFile(CAT).metadata.num_rows
    multi = st.filter(pl.col("n_unique") >= 2)
    sig = {r["column"]: (r["n_nonnull"], r["n_unique"], r["top_value"], r["top_count"]) for r in multi.to_dicts()}
    cands = candidate_groups(sig)
    groups: list[list[str]] = []
    for g in cands:
        for i in range(0, len(g), 200):  # 후보 묶음이 크면 나눠 읽는다(대표 컬럼은 항상 포함)
            part = g[i:i + 200] if i == 0 else [g[0]] + g[i:i + 200]
            found = group_identical_columns(pl.read_parquet(CAT, columns=part))
            for f in found:
                hit = next((x for x in groups if x[0] == f[0]), None)
                if hit is None:
                    groups.append(list(f))
                else:
                    hit += [c for c in f if c not in hit]
        if len(g) > 200:
            raise SystemExit("후보 묶음이 200개를 넘는다. 묶음 사이 비교를 다시 설계해야 한다")
    rows = [{"group": i, "column": c, "is_representative": j == 0} for i, g in enumerate(groups) for j, c in enumerate(g)]
    pl.DataFrame(rows, schema={"group": pl.Int64, "column": pl.String, "is_representative": pl.Boolean}).write_csv(
        OUT / "categorical_duplicate_columns.csv")
    dup_of = {c: g[0] for g in groups for c in g[1:]}
    st = st.with_columns(
        (pl.col("top_count") / pl.col("n_nonnull")).alias("top_share"),
        pl.col("column").replace_strict(dup_of, default=None, return_dtype=pl.String).alias("duplicate_of"),
    )
    st = st.with_columns(pl.Series("verdict", [classify_categorical(r["n_nonnull"], r["n_unique"], r["duplicate_of"])
                                               for r in st.to_dicts()]))
    st.write_parquet(OUT / "categorical_column_stats.parquet")
    keep = st.filter(pl.col("verdict") == "keep")
    # 남길 후보 컬럼에 값이 하나라도 있는 부품
    any_keep = np.zeros(n_rows, dtype=bool)
    any_all = np.zeros(n_rows, dtype=bool)
    kc = keep["column"].to_list()
    lf = pl.scan_parquet(CAT)
    for i in range(0, len(kc), 100):
        r = lf.select(pl.any_horizontal([pl.col(c).is_not_null() for c in kc[i:i + 100]]).alias("a")).collect(engine="streaming")
        any_keep |= r["a"].to_numpy()
    for i in range(0, len(cols), 100):
        r = lf.select(pl.any_horizontal([pl.col(c).is_not_null() for c in cols[i:i + 100]]).alias("a")).collect(engine="streaming")
        any_all |= r["a"].to_numpy()
    counts = {k: int(v) for k, v in st.group_by("verdict").len().iter_rows()}
    res = {
        "n_columns": st.height, "verdict_counts": counts,
        "duplicate_groups": len(groups), "largest_duplicate_groups": sorted((len(g) for g in groups), reverse=True)[:10],
        "keep_top_share_ge_0.99": int((keep["top_share"] >= 0.99).sum()),
        "keep_top_share_lt_0.99": int((keep["top_share"] < 0.99).sum()),
        "keep_n_unique_median": float(keep["n_unique"].median()), "keep_n_unique_max": int(keep["n_unique"].max()),
        "keep_nonnull_lt_1000": int((keep["n_nonnull"] < 1000).sum()),
        "keep_nonnull_ge_1pct_rows": int((keep["n_nonnull"] >= n_rows * 0.01).sum()),
        "keep_stations": keep["station_key"].n_unique(),
        "parts_with_value_in_keep": int(any_keep.sum()), "share_parts_with_value_in_keep": float(any_keep.mean()),
        "parts_with_any_categorical": int(any_all.sum()), "share_parts_with_any_categorical": float(any_all.mean()),
    }
    update_summary("categorical", res)
    print(json.dumps(res, ensure_ascii=False, indent=1))


def _pa_stats(path: Path, col: str) -> dict:
    a = pq.read_table(path, columns=[col]).column(0).drop_null()
    vals = a.to_numpy(zero_copy_only=False)
    out = {"n_nonnull": len(vals), "n_unique": len(set(vals.tolist()))}
    if len(vals) and vals.dtype.kind == "f":
        out.update({"min": float(vals.min()), "max": float(vals.max())})
    return out


def stage_verify() -> None:
    rng = np.random.default_rng(SEED)
    res = {}
    ns = pl.read_parquet(OUT / "numeric_column_stats.parquet")
    cs = pl.read_parquet(OUT / "categorical_column_stats.parquet")
    for name, path, st in (("numeric", NUM, ns), ("categorical", CAT, cs)):
        pick = sorted(rng.choice(st.height, N_VERIFY_COLS, replace=False).tolist())
        if name == "categorical":  # 값이 있는 컬럼이 드물어 남길 후보에서 절반을 고른다
            keep_idx = np.flatnonzero((st["verdict"] == "keep").to_numpy())
            pick = sorted(set(pick[:N_VERIFY_COLS // 2]) | set(rng.choice(keep_idx, N_VERIFY_COLS // 2, replace=False).tolist()))
        bad = 0
        for r in st[pick].to_dicts():
            ref = _pa_stats(path, r["column"])
            bad += any(r[k] != v for k, v in ref.items())
        res[f"{name}_columns_checked"] = len(pick)
        res[f"{name}_columns_mismatch"] = bad
    # 같은 컬럼 묶음: 묶음마다 대표와 나머지를 pyarrow로 다시 비교
    for name, path in (("numeric", NUM), ("categorical", CAT)):
        g = pl.read_csv(OUT / f"{name}_duplicate_columns.csv")
        bad = 0
        for _, part in g.group_by("group", maintain_order=True):
            cols = part["column"].to_list()
            t = pq.read_table(path, columns=cols)
            bad += sum(not t.column(0).equals(t.column(j)) for j in range(1, len(cols)))
        res[f"{name}_duplicate_groups_checked"] = g["group"].n_unique()
        res[f"{name}_duplicate_columns_mismatch"] = bad
    # 같은 부품 묶음: 묶음 300개를 골라 원본 행을 다시 읽어 비교
    d = pl.read_parquet(OUT / "duplicate_rows_numeric.parquet")
    gids = rng.choice(d["group"].unique().to_numpy(), min(300, d["group"].n_unique()), replace=False)
    sub = d.filter(pl.col("group").is_in(gids.tolist()))
    cols = feature_columns(NUM)
    raw = pl.scan_parquet(NUM).filter(pl.col("Id").is_in(sub["Id"].to_list())).select(["Id"] + cols).collect(engine="streaming")
    raw = sub.select("Id", "group").join(raw, on="Id")
    bad = 0
    for _, part in raw.group_by("group"):
        a = part.select(cols).to_numpy()
        bad += int(not (((a == a[0]) | (np.isnan(a) & np.isnan(a[0]))).all()))
    res["duplicate_row_groups_checked"] = len(gids)
    res["duplicate_row_groups_mismatch"] = bad
    # 합계
    counts = {k: int(v) for k, v in cs.group_by("verdict").len().iter_rows()}
    res["categorical_verdict_sum"] = sum(counts.values())
    res["categorical_empty"] = counts.get("empty", 0)
    s = json.loads((OUT / "q6_q7_summary.json").read_text())
    res["s32_n_with_value"] = s["s32"]["n_with_value"]
    res["s32_n_skip_s33"] = s["s32"]["n_skip_s33"]
    ok = (res["numeric_columns_mismatch"] == 0 and res["categorical_columns_mismatch"] == 0
          and res["numeric_duplicate_columns_mismatch"] == 0 and res["categorical_duplicate_columns_mismatch"] == 0
          and res["duplicate_row_groups_mismatch"] == 0 and res["categorical_verdict_sum"] == 2140
          and res["categorical_empty"] == 163 and res["s32_n_with_value"] == 24543 and res["s32_n_skip_s33"] == 6567)
    res["all_passed"] = bool(ok)
    update_summary("verify", res)
    print(json.dumps(res, ensure_ascii=False, indent=1))
    if not ok:
        raise SystemExit("검증 실패: 원인을 찾기 전에는 결과를 쓰지 않는다")


STAGES = {"numeric_stats": stage_numeric_stats, "numeric_dup_cols": stage_numeric_dup_cols, "dup_rows": stage_dup_rows,
          "s32": stage_s32, "categorical": stage_categorical, "verify": stage_verify}

if __name__ == "__main__":
    arg = sys.argv[1] if len(sys.argv) > 1 else "all"
    for name in (STAGES if arg == "all" else [arg]):
        print(f"=== {name} ===", flush=True)
        STAGES[name]()
        print(f"peak {peak_mib()} MiB", flush=True)
