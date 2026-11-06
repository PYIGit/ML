"""3-2: Q1 구조·식별자·무결성, Q2 컬럼 구조, Q3 타깃 분포 집계.

입력: DATA/raw/bosch_parquet/*.parquet
출력: DATA/interim/eda/ (Git 무시)
  q1_integrity.json, column_meta.parquet, station_structure.csv, q2_summary.json,
  target_summary.json, row_features.parquet, row_features_vs_response.csv

단계별로 실행해 단계마다 피크 메모리를 기록한다 (q2가 q3보다 먼저 있어야 한다):
  .venv/bin/python scripts/eda_3_2_structure.py q1
  .venv/bin/python scripts/eda_3_2_structure.py q2
  .venv/bin/python scripts/eda_3_2_structure.py q3
"""
import json
import resource
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import polars as pl
import pyarrow.parquet as pq

from ml.data.columns import build_column_meta
from ml.stats import rate_table, wilson_interval

PQ = Path("DATA/raw/bosch_parquet")
OUT = Path("DATA/interim/eda")
FEATURE_FILES = ["train_numeric", "test_numeric", "train_date", "test_date",
                 "train_categorical", "test_categorical"]
ALL_FILES = FEATURE_FILES + ["sample_submission"]
CAT_RE = r"^T-?\d+$"


def scan(name: str) -> pl.LazyFrame:
    return pl.scan_parquet(PQ / f"{name}.parquet")


def peak_mib() -> int:
    return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)


def pq_info(name: str):
    """(컬럼명, 컬럼별 null 수, 행 수). null 수는 행 그룹 통계에서 읽고 데이터는 읽지 않는다."""
    f = pq.ParquetFile(PQ / f"{name}.parquet")
    md = f.metadata
    nulls: list = [0] * md.num_columns
    for g in range(md.num_row_groups):
        rg = md.row_group(g)
        for j in range(md.num_columns):
            st = rg.column(j).statistics
            if st is None or not st.has_null_count:
                nulls[j] = None
            elif nulls[j] is not None:
                nulls[j] += st.null_count
    return f.schema_arrow.names, nulls, md.num_rows


def dump(obj, path: Path) -> None:
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=int))


def finish(stage: str, t0: float, result: dict, out_name: str) -> None:
    result["_run"] = {"stage": stage, "seconds": round(time.time() - t0, 1), "peak_rss_mib": peak_mib(),
                      "polars": pl.__version__}
    dump(result, OUT / out_name)
    print(f"[{stage}] 완료 {result['_run']['seconds']}s, 피크 RSS {result['_run']['peak_rss_mib']} MiB"
          f" -> {OUT / out_name}", flush=True)


# ------------------------------------------------------------------ Q1
def count_nan_inf(name: str, cols: list[str], batch: int = 200) -> dict:
    lf = scan(name)
    n_nan = n_inf = 0
    nan_cols, inf_cols = [], []
    for i in range(0, len(cols), batch):
        b = cols[i:i + batch]
        exprs = ([pl.col(c).is_nan().sum().alias(f"{c}|nan") for c in b]
                 + [pl.col(c).is_infinite().sum().alias(f"{c}|inf") for c in b])
        row = lf.select(exprs).collect(engine="streaming").row(0, named=True)
        for key, v in row.items():
            c, kind = key.rsplit("|", 1)
            if v:
                (nan_cols if kind == "nan" else inf_cols).append(c)
                if kind == "nan":
                    n_nan += v
                else:
                    n_inf += v
    return {"n_nan": n_nan, "n_inf": n_inf, "n_cols_with_nan": len(nan_cols),
            "n_cols_with_inf": len(inf_cols), "example_cols_nan": nan_cols[:10],
            "example_cols_inf": inf_cols[:10]}


def check_categorical_format(name: str, cols: list[str], batch: int = 50) -> dict:
    lf = scan(name)
    tot = {"n_values": 0, "n_pattern_violation": 0, "n_leading_trailing_space": 0, "n_has_lowercase": 0}
    viol_cols = []
    for i in range(0, len(cols), batch):
        b = cols[i:i + batch]
        exprs = []
        for c in b:
            v = pl.col(c).drop_nulls()
            exprs += [v.len().alias(f"{c}|n"),
                      v.str.contains(CAT_RE).not_().sum().alias(f"{c}|viol"),
                      (v != v.str.strip_chars()).sum().alias(f"{c}|space"),
                      v.str.contains(r"[a-z]").sum().alias(f"{c}|lower")]
        row = lf.select(exprs).collect(engine="streaming").row(0, named=True)
        for key, val in row.items():
            c, kind = key.rsplit("|", 1)
            tot[{"n": "n_values", "viol": "n_pattern_violation", "space": "n_leading_trailing_space",
                 "lower": "n_has_lowercase"}[kind]] += val
            if kind == "viol" and val:
                viol_cols.append(c)
    examples = {}
    for c in viol_cols[:5]:
        ex = (lf.select(c).drop_nulls().filter(~pl.col(c).str.contains(CAT_RE))
              .unique().head(5).collect()[c].to_list())
        examples[c] = ex
    tot["n_cols_with_pattern_violation"] = len(viol_cols)
    tot["violation_examples"] = examples
    return tot


def stage_q1() -> None:
    t0 = time.time()
    res: dict = {}
    info = {n: pq_info(n) for n in ALL_FILES}
    res["rows"] = {n: info[n][2] for n in ALL_FILES}
    res["columns"] = {n: len(info[n][0]) for n in ALL_FILES}

    ids = {n: scan(n).select("Id").collect().to_series() for n in ALL_FILES}
    per = {}
    for n, s in ids.items():
        per[n] = {"n": len(s), "n_unique": s.n_unique(), "is_sorted": bool(s.is_sorted()),
                  "min": int(s.min()), "max": int(s.max()), "n_null": s.null_count()}
    res["id_per_file"] = per
    res["id_equal_across_files"] = {
        "train: numeric==date": bool(ids["train_numeric"].equals(ids["train_date"])),
        "train: numeric==categorical": bool(ids["train_numeric"].equals(ids["train_categorical"])),
        "test: numeric==date": bool(ids["test_numeric"].equals(ids["test_date"])),
        "test: numeric==categorical": bool(ids["test_numeric"].equals(ids["test_categorical"])),
        "sample_submission==test_numeric": bool(ids["sample_submission"].equals(ids["test_numeric"])),
    }
    tr, te = ids["train_numeric"], ids["test_numeric"]
    overlap = int(tr.is_in(te.implode()).sum())
    both = pl.concat([pl.DataFrame({"Id": tr, "is_test": pl.repeat(False, len(tr), eager=True)}),
                      pl.DataFrame({"Id": te, "is_test": pl.repeat(True, len(te), eager=True)})]).sort("Id")
    runs = both["is_test"].rle().struct.unnest()
    run_stats = {}
    for label, val in (("train", False), ("test", True)):
        r = runs.filter(pl.col("value") == val)["len"]
        run_stats[label] = {"n_runs": len(r), "max_run_length": int(r.max()), "mean_run_length": round(float(r.mean()), 2)}
    te_in_train_range = int(((te >= tr.min()) & (te <= tr.max())).sum())
    res["train_vs_test_ids"] = {
        "n_overlapping_ids": overlap,
        "train_min_max": [int(tr.min()), int(tr.max())], "test_min_max": [int(te.min()), int(te.max())],
        "test_ids_inside_train_range": te_in_train_range,
        "split_changes_in_sorted_ids": int((both["is_test"] != both["is_test"].shift(1)).sum()),
        "runs": run_stats,
        "union_size": len(tr) + len(te), "id_span": int(max(tr.max(), te.max()) - min(tr.min(), te.min()) + 1),
    }

    names = {n: info[n][0] for n in ALL_FILES}
    tn, ten = names["train_numeric"], names["test_numeric"]
    res["schema"] = {
        "train_numeric_minus_test_numeric": [c for c in tn if c not in ten],
        "test_numeric_minus_train_numeric": [c for c in ten if c not in tn],
        "response_position_in_train_numeric": tn.index("Response"),
        "numeric_other_columns_same_order": [c for c in tn if c != "Response"] == ten,
        "date_columns_same_order": names["train_date"] == names["test_date"],
        "categorical_columns_same_order": names["train_categorical"] == names["test_categorical"],
        "Id_is_first_column": {n: names[n][0] == "Id" for n in ALL_FILES},
    }
    sub = scan("sample_submission").select("Response").unique().collect()["Response"].to_list()
    res["sample_submission_response_values"] = sorted(sub)

    res["nan_inf"] = {}
    for n in ["train_numeric", "test_numeric", "train_date", "test_date"]:
        cols = [c for c in names[n] if c not in ("Id", "Response")]
        res["nan_inf"][n] = count_nan_inf(n, cols)
        print(f"  NaN/inf {n}: {res['nan_inf'][n]['n_nan']}/{res['nan_inf'][n]['n_inf']}", flush=True)
    res["categorical_format"] = {}
    for n in ["train_categorical", "test_categorical"]:
        cols = [c for c in names[n] if c != "Id"]
        res["categorical_format"][n] = check_categorical_format(n, cols)
        print(f"  categorical 형식 {n}: 위반 {res['categorical_format'][n]['n_pattern_violation']}", flush=True)

    stop = {
        "ids_not_equal_across_files": not all(res["id_equal_across_files"][k] for k in
                                              ("train: numeric==date", "train: numeric==categorical",
                                               "test: numeric==date", "test: numeric==categorical")),
        "id_duplicates": any(p["n"] != p["n_unique"] for p in per.values()),
        "train_test_id_overlap": overlap > 0,
    }
    res["stop_conditions_triggered"] = stop
    OUT.mkdir(parents=True, exist_ok=True)
    finish("q1", t0, res, "q1_integrity.json")
    if any(stop.values()):
        sys.exit(f"중단 조건 발생: {stop}")


# ------------------------------------------------------------------ Q2
def stage_q2() -> None:
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    metas = []
    for n in ALL_FILES:
        names, nulls, n_rows = pq_info(n)
        metas.append(build_column_meta(n, names, nulls, n_rows))
    meta = pl.concat(metas)
    meta.write_parquet(OUT / "column_meta.parquet")

    feat = meta.filter((pl.col("role") == "feature") & (pl.col("split") == "train"))
    by_station: dict = defaultdict(lambda: {"numeric": [], "categorical": [], "date": []})
    st_info = {}
    for r in feat.iter_rows(named=True):
        by_station[r["station_key"]][r["data_kind"]].append(r["number"])
        st_info[r["station_key"]] = (r["line"], r["station"])
    rows = []
    for sk, d in by_station.items():
        line, station = st_info[sk]
        nums, cats, dates = sorted(d["numeric"]), sorted(d["categorical"]), sorted(d["date"])
        f_union = set(nums) | set(cats)
        f_plus1 = {f + 1 for f in f_union}
        rows.append({
            "station_key": sk, "line": line, "station": station,
            "n_numeric": len(nums), "n_categorical": len(cats), "n_date": len(dates),
            "in_numeric": bool(nums), "in_categorical": bool(cats), "in_date": bool(dates),
            "numeric_F": ",".join(map(str, nums)), "categorical_F": ",".join(map(str, cats)),
            "date_D": ",".join(map(str, dates)),
            "numeric_cat_F_disjoint": not (set(nums) & set(cats)),
            "n_D_equal_Fplus1": len(set(dates) & f_plus1),
            "frac_D_equal_Fplus1": (len(set(dates) & f_plus1) / len(dates)) if dates else None,
            "frac_F_with_D_at_Fplus1": (len(f_union & {d - 1 for d in dates}) / len(f_union)) if f_union else None,
        })
    st = pl.DataFrame(rows).sort(["line", "station"])
    st.write_csv(OUT / "station_structure.csv")

    res: dict = {}
    for kind in ("numeric", "date", "categorical"):
        k = feat.filter(pl.col("data_kind") == kind)
        per_station = k.group_by("station_key").len()["len"]
        res[kind] = {
            "n_feature_columns": k.height, "lines": sorted(k["line"].unique().to_list()),
            "n_stations": k["station_key"].n_unique(),
            "stations_per_line": {int(l): int(c) for l, c in
                                  k.group_by("line").agg(pl.col("station_key").n_unique()).sort("line").iter_rows()},
            "features_per_station": {"min": int(per_station.min()), "median": float(per_station.median()),
                                     "max": int(per_station.max())},
        }
    all_st = feat.select("line", "station").unique()
    multi = (all_st.group_by("station").agg(pl.col("line").sort().alias("lines"))
             .filter(pl.col("lines").list.len() > 1).sort("station"))
    res["station_number_used_in_multiple_lines"] = {
        "count": multi.height, "examples": multi.head(10).to_dicts()}
    res["station_sets"] = {
        "only_in_date_not_in_numeric": st.filter(pl.col("in_date") & ~pl.col("in_numeric"))["station_key"].to_list(),
        "in_numeric_not_in_date": st.filter(pl.col("in_numeric") & ~pl.col("in_date"))["station_key"].to_list(),
        "in_numeric_not_in_categorical": st.filter(pl.col("in_numeric") & ~pl.col("in_categorical"))["station_key"].to_list(),
        "only_in_categorical": st.filter(pl.col("in_categorical") & ~pl.col("in_numeric"))["station_key"].to_list(),
        "n_stations_total": st.height,
    }
    both = st.filter(pl.col("in_numeric") & pl.col("in_categorical"))
    res["numeric_vs_categorical_F_numbers"] = {
        "stations_with_both": both.height,
        "stations_with_disjoint_F_sets": int(both["numeric_cat_F_disjoint"].sum()),
    }
    withd = st.filter(pl.col("in_date") & (pl.col("in_numeric") | pl.col("in_categorical")))
    res["date_vs_F_numbers_hypothesis_D_equals_Fplus1"] = {
        "stations_compared": withd.height,
        "stations_all_D_equal_Fplus1": int((withd["frac_D_equal_Fplus1"] == 1.0).sum()),
        "stations_no_D_equal_Fplus1": int((withd["frac_D_equal_Fplus1"] == 0.0).sum()),
        "mean_frac_D_equal_Fplus1": round(float(withd["frac_D_equal_Fplus1"].mean()), 4),
        "mean_frac_F_with_D_at_Fplus1": round(float(withd["frac_F_with_D_at_Fplus1"].mean()), 4),
        "note": "추정. 번호 인접성만 본 것이며 측정 시각과의 대응을 확정하지 않는다",
    }
    tr_names = {n: meta.filter(pl.col("file") == n)["column"].to_list() for n in FEATURE_FILES}
    res["test_schema_matches_train"] = {
        "date": tr_names["train_date"] == tr_names["test_date"],
        "categorical": tr_names["train_categorical"] == tr_names["test_categorical"],
    }
    res["null_count_source"] = "Parquet 행 그룹 통계. 통계가 없는 컬럼 수: " + str(
        int(meta["null_count"].null_count()))
    finish("q2", t0, res, "q2_summary.json")


# ------------------------------------------------------------------ Q3
def stage_q3() -> None:
    t0 = time.time()
    meta = pl.read_parquet(OUT / "column_meta.parquet")
    num = meta.filter((pl.col("file") == "train_numeric") & (pl.col("role") == "feature"))
    st_cols: dict = defaultdict(list)
    st_line = {}
    for r in num.iter_rows(named=True):
        st_cols[r["station_key"]].append(r["column"])
        st_line[r["station_key"]] = r["line"]
    visited = {sk: pl.any_horizontal([pl.col(c).is_not_null() for c in cols]) for sk, cols in st_cols.items()}
    line_visited = {}
    for ln in sorted(set(st_line.values())):
        line_visited[ln] = pl.any_horizontal([visited[sk] for sk in st_cols if st_line[sk] == ln])
    all_cols = num["column"].to_list()
    rf = (scan("train_numeric").select(
        "Id", "Response",
        pl.sum_horizontal([pl.col(c).is_not_null() for c in all_cols]).cast(pl.Int32).alias("n_nonnull_numeric"),
        pl.sum_horizontal(list(visited.values())).cast(pl.Int16).alias("n_stations_visited_numeric"),
        pl.sum_horizontal(list(line_visited.values())).cast(pl.Int8).alias("n_lines_visited_numeric"),
    ).collect(engine="streaming"))
    rf.write_parquet(OUT / "row_features.parquet")

    n = rf.height
    resp = rf.group_by("Response").len().sort("Response")
    counts = {("null" if k is None else str(k)): int(v) for k, v in resp.iter_rows()}
    k1 = counts.get("1", 0)
    lo, hi = wilson_interval(k1, n)
    summary = {"n_rows": n, "response_counts": counts, "n_response_null": rf["Response"].null_count(),
               "defect_rate": k1 / n, "defect_rate_wilson95": [lo, hi],
               "response_dtype": str(rf.schema["Response"])}
    tables = []
    for feat in ("n_stations_visited_numeric", "n_lines_visited_numeric"):
        g = rf.group_by(feat).agg(pl.len().alias("n"), pl.col("Response").sum().alias("k")).sort(feat)
        g = rate_table(g.with_columns(pl.col(feat).cast(pl.Utf8).alias("group"), pl.col(feat).cast(pl.Int64).alias("order")),
                       ["order", "group"]).with_columns(pl.lit(feat).alias("feature"))
        tables.append(g)
    s_nn = rf["n_nonnull_numeric"]
    edges = sorted({int(s_nn.quantile(q / 10, interpolation="lower")) for q in range(1, 10)})
    idx = np.searchsorted(edges, s_nn.to_numpy(), side="left")  # 구간 i: (edges[i-1], edges[i]]
    labels = ([f"[{int(s_nn.min())}, {edges[0]}]"]
              + [f"({edges[i - 1]}, {edges[i]}]" for i in range(1, len(edges))]
              + [f"({edges[-1]}, {int(s_nn.max())}]"])
    b = pl.DataFrame({"order": idx.astype("int64"), "Response": rf["Response"]})
    g = b.group_by("order").agg(pl.len().alias("n"), pl.col("Response").sum().alias("k")).sort("order")
    g = g.with_columns(pl.col("order").map_elements(lambda i: labels[i], return_dtype=pl.Utf8).alias("group"))
    tables.append(rate_table(g, ["order", "group"]).with_columns(pl.lit("n_nonnull_numeric").alias("feature")))
    out = pl.concat([t.select("feature", "order", "group", "n", "k", "rate", "ci_low", "ci_high", "small_group")
                     for t in tables])
    out.write_csv(OUT / "row_features_vs_response.csv")
    summary["row_feature_summary"] = {
        c: {"min": int(rf[c].min()), "max": int(rf[c].max()), "mean": round(float(rf[c].mean()), 3)}
        for c in ("n_nonnull_numeric", "n_stations_visited_numeric", "n_lines_visited_numeric")}
    summary["n_rows_with_zero_numeric_values"] = int((rf["n_nonnull_numeric"] == 0).sum())
    summary["visit_definition"] = "스테이션의 numeric 컬럼 중 하나라도 값이 있으면 방문(프로젝트 가정)"
    finish("q3", t0, summary, "target_summary.json")


if __name__ == "__main__":
    stages = {"q1": stage_q1, "q2": stage_q2, "q3": stage_q3}
    arg = sys.argv[1] if len(sys.argv) > 1 else ""
    if arg not in stages:
        sys.exit(__doc__)
    stages[arg]()
