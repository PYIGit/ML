"""3-6 예측 시점에 쓸 수 있는 정보와 train·test 비교 (Q8, Q9). 설계: DOCS/03_EDA/3-6_LEAKAGE_AND_TRAIN_TEST.md

입력: DATA/raw/bosch_parquet/{test_date.parquet, test_numeric.parquet, train_numeric.parquet},
      DATA/interim/eda/{column_meta.parquet, station_time_train.parquet, row_time_features.parquet,
                        time_bin_summary.csv, duplicate_rows_numeric.parquet}
출력: DATA/interim/eda/ (Git 무시)
  station_time_test.parquet, train_test_comparison.csv, duplicate_rows_combined.parquet,
  last_station_summary.csv, prediction_point_summary.csv, q8_q9_summary.json

단계(순서대로): test_times -> compare -> dup_combined -> last_station -> pred_point -> verify
사용: .venv/bin/python scripts/eda_3_6_leakage.py <stage|all>
"""
import json
import resource
import sys
from pathlib import Path

import numpy as np
import polars as pl
import pyarrow.parquet as pq

from ml.data.features import confirm_row_groups
from ml.data.leakage import group_kinds, last_station_index, ratio_outside, share_recorded_before
from ml.data.timing import assign_time_bin, compute_station_time_range, part_time_features
from ml.data.visits import compute_station_counts, station_columns
from ml.stats import rate_table

PQ = Path("DATA/raw/bosch_parquet")
OUT = Path("DATA/interim/eda")
BIN_WIDTH = 10.0
SEED = 20261112
FLOW_ENDS = ["L3_S37", "L3_S38", "L3_S51"]


def peak_mib() -> int:
    return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)


def update_summary(key: str, value: dict) -> None:
    p = OUT / "q8_q9_summary.json"
    cur = json.loads(p.read_text()) if p.exists() else {}
    value["peak_rss_mib"] = peak_mib()
    cur[key] = value
    p.write_text(json.dumps(cur, indent=2, ensure_ascii=False))


def station_no(key: str) -> int:
    return int(key.split("_S")[1])


def groups_of(kind: str) -> dict[str, list[str]]:
    g = station_columns(pl.read_parquet(OUT / "column_meta.parquet"), kind)
    return {k: g[k] for k in sorted(g, key=station_no)}


def stations_of(times: pl.DataFrame) -> list[str]:
    return [c[:-5] for c in times.columns if c.endswith("_tmin")]


def test_features(times: pl.DataFrame) -> pl.DataFrame:
    f = part_time_features(times, stations_of(times))
    return f.with_columns(assign_time_bin(f["t_first"], BIN_WIDTH))


def stage_test_times() -> None:
    groups = groups_of("date")
    times = compute_station_time_range(PQ / "test_date.parquet", groups)
    ids = pl.read_parquet(PQ / "test_date.parquet", columns=["Id"])["Id"]
    times.insert_column(0, ids).write_parquet(OUT / "station_time_test.parquet")
    f = test_features(times)
    info = {"rows": times.height, "rows_without_time": int(f["t_first"].is_null().sum()),
            "t_first_min": float(f["t_first"].min()), "t_last_max": float(f["t_last"].max()),
            "bytes": (OUT / "station_time_test.parquet").stat().st_size}
    update_summary("test_times", info)
    print(json.dumps(info, ensure_ascii=False, indent=1))


def stage_compare() -> None:
    tr = pl.read_parquet(OUT / "station_time_train.parquet")
    te = pl.read_parquet(OUT / "station_time_test.parquet")
    st = stations_of(tr)
    ftr = pl.read_parquet(OUT / "row_time_features.parquet")
    fte = test_features(te)
    rows = []
    # (가) 시간 구간별 부품 수
    bins = pl.read_csv(OUT / "time_bin_summary.csv").select("time_bin", "t_start", "n", "outside")
    nte = fte.filter(pl.col("time_bin").is_not_null()).group_by("time_bin").len().rename({"len": "n_test"})
    b = bins.join(nte, on="time_bin", how="full", coalesce=True).fill_null(0).sort("time_bin")
    for r in b.to_dicts():
        rows.append({"kind": "time_bin_count", "item": str(r["time_bin"]), "train": r["n"], "test": r["n_test"],
                     "diff": r["n_test"] - r["n"]})
    ro = ratio_outside(b["n"].to_numpy(), b["n_test"].to_numpy(), 0.9, 1.1, 1000)
    high = b.filter(pl.col("outside") == "high")
    time_res = {
        **ro, "bins_train": int((b["n"] > 0).sum()), "bins_test": int((b["n_test"] > 0).sum()),
        "bins_only_in_one": int(((b["n"] > 0) != (b["n_test"] > 0)).sum()),
        "share_train_in_high_bins": float(high["n"].sum() / b["n"].sum()),
        "share_test_in_high_bins": float(high["n_test"].sum() / b["n_test"].sum()),
        "share_train_before_860": float((ftr["t_first"] < 860).sum() / ftr["t_first"].is_not_null().sum()),
        "share_test_before_860": float((fte["t_first"] < 860).sum() / fte["t_first"].is_not_null().sum()),
        "t_first_range_train": [float(ftr["t_first"].min()), float(ftr["t_first"].max())],
        "t_first_range_test": [float(fte["t_first"].min()), float(fte["t_first"].max())],
        "no_time_train": int(ftr["t_first"].is_null().sum()), "no_time_test": int(fte["t_first"].is_null().sum()),
        "duration_median_train": float(ftr["duration"].median()), "duration_median_test": float(fte["duration"].median()),
    }
    # (나) 스테이션 방문율 (date 기준)
    vr = []
    for s in st:
        a, c = float(tr[f"{s}_tmin"].is_not_null().mean()), float(te[f"{s}_tmin"].is_not_null().mean())
        vr.append((s, a, c))
        rows.append({"kind": "station_visit_rate", "item": s, "train": a, "test": c, "diff": c - a})
    worst = max(vr, key=lambda x: abs(x[2] - x[1]))
    visit_res = {"stations": len(st), "max_abs_diff": abs(worst[2] - worst[1]), "max_diff_station": worst[0],
                 "max_diff_train": worst[1], "max_diff_test": worst[2]}
    # (다) 컬럼 결측률
    meta = pl.read_parquet(OUT / "column_meta.parquet").filter(pl.col("station_key").is_not_null())
    name_col = "column" if "column" in meta.columns else "name"
    piv = (meta.select("split", "data_kind", name_col, "null_rate")
           .pivot(on="split", index=["data_kind", name_col], values="null_rate")
           .with_columns((pl.col("test") - pl.col("train")).alias("diff")))
    null_res = {}
    for kind, part in piv.group_by("data_kind"):
        w = part.sort(pl.col("diff").abs(), descending=True).row(0, named=True)
        null_res[kind[0]] = {"columns": part.height, "max_abs_diff": abs(w["diff"]), "column": w[name_col],
                             "train": w["train"], "test": w["test"]}
        rows.append({"kind": f"null_rate_max_diff_{kind[0]}", "item": w[name_col], "train": w["train"], "test": w["test"],
                     "diff": w["diff"]})
    # (라) L3_S32를 거치고 L3_S33을 거치지 않은 부품 (date 기준)
    cond = {}
    for name, t in (("train", tr), ("test", te)):
        c = t["L3_S32_tmin"].is_not_null() & t["L3_S33_tmin"].is_null()
        cond[name] = {"n": int(c.sum()), "share": float(c.mean())}
    rows.append({"kind": "s32_without_s33_share", "item": "L3_S32 & not L3_S33", "train": cond["train"]["share"],
                 "test": cond["test"]["share"], "diff": cond["test"]["share"] - cond["train"]["share"]})
    pl.DataFrame(rows, schema={"kind": pl.String, "item": pl.String, "train": pl.Float64, "test": pl.Float64,
                               "diff": pl.Float64}).write_csv(OUT / "train_test_comparison.csv")
    res = {"time": time_res, "visit_rate": visit_res, "null_rate": null_res, "s32_without_s33": cond}
    update_summary("compare", res)
    print(json.dumps(res, ensure_ascii=False, indent=1))


def _hash_file(path: Path, cols: list[str]) -> tuple[np.ndarray, np.ndarray]:
    pf = pq.ParquetFile(path)
    hs, has = [], []
    for i in range(pf.num_row_groups):
        df = pl.from_arrow(pf.read_row_group(i, columns=cols))
        hs.append(df.hash_rows(seed=1).to_numpy())
        has.append(df.select(pl.any_horizontal(pl.all().is_not_null())).to_series().to_numpy())
        del df
    return np.concatenate(hs), np.concatenate(has)


def _gather(path: Path, cols: list[str], rows: np.ndarray, out: np.ndarray) -> None:
    """파일의 행 번호 rows(오름차순)의 값을 out에 채운다 (Float32: 값이 소수 셋째 자리라 같음 판정이 바뀌지 않는다)."""
    pf = pq.ParquetFile(path)
    pos = start = 0
    for i in range(pf.num_row_groups):
        n = pf.metadata.row_group(i).num_rows
        local = rows[(rows >= start) & (rows < start + n)] - start
        if len(local):
            df = pl.from_arrow(pf.read_row_group(i, columns=cols))[local.tolist()]
            out[pos:pos + len(local)] = df.to_numpy().astype(np.float32)
            pos += len(local)
            del df
        start += n


def stage_dup_combined() -> None:
    cols = [c for c in pq.ParquetFile(PQ / "train_numeric.parquet").schema_arrow.names if c not in ("Id", "Response")]
    assert cols == [c for c in pq.ParquetFile(PQ / "test_numeric.parquet").schema_arrow.names if c != "Id"]
    h_tr, has_tr = _hash_file(PQ / "train_numeric.parquet", cols)
    h_te, has_te = _hash_file(PQ / "test_numeric.parquet", cols)
    n_tr, n_te = len(h_tr), len(h_te)
    h = np.concatenate([h_tr, h_te])
    has = np.concatenate([has_tr, has_te])
    _, inv, cnt = np.unique(h, return_inverse=True, return_counts=True)
    cand = np.flatnonzero((cnt[inv] > 1) & has)
    print(f"  candidate rows: {len(cand):,}", flush=True)
    if len(cand) > 500_000:
        raise SystemExit("후보 행이 너무 많다. 나눠서 비교하도록 바꿔야 한다")
    vals = np.empty((len(cand), len(cols)), dtype=np.float32)
    c_tr = cand[cand < n_tr]
    c_te = cand[cand >= n_tr] - n_tr
    _gather(PQ / "train_numeric.parquet", cols, c_tr, vals[:len(c_tr)])
    _gather(PQ / "test_numeric.parquet", cols, c_te, vals[len(c_tr):])
    gid = confirm_row_groups(h[cand], vals)
    del vals
    ids_tr = pl.read_parquet(OUT / "row_time_features.parquet", columns=["Id", "Response"])
    ids_te = pl.read_parquet(PQ / "test_numeric.parquet", columns=["Id"])
    all_ids = np.concatenate([ids_tr["Id"].to_numpy(), ids_te["Id"].to_numpy()])
    resp = np.concatenate([ids_tr["Response"].to_numpy().astype(np.float64), np.full(n_te, np.nan)])
    keep = gid >= 0
    rows = cand[keep]
    is_train = rows < n_tr
    d = pl.DataFrame({"group": gid[keep], "Id": all_ids[rows], "split": np.where(is_train, "train", "test"),
                      "Response": resp[rows], "kind": group_kinds(gid[keep], is_train)}).with_columns(
        pl.col("Response").cast(pl.Int8, strict=False)).sort(["group", "Id"])
    d.write_parquet(OUT / "duplicate_rows_combined.parquet")
    g = d.group_by("group").agg(pl.len().alias("size"), pl.col("kind").first(),
                                (pl.col("Id").max() - pl.col("Id").min()).alias("id_span"),
                                (pl.col("Id").diff().drop_nulls() == 1).all().alias("ids_consecutive"))
    kinds = {r["kind"]: {"groups": r["groups"], "parts": r["parts"]} for r in
             g.group_by("kind").agg(pl.len().alias("groups"), pl.col("size").sum().alias("parts")).to_dicts()}
    # train 부품을 합친 기준으로 다시 나눈 불량률
    tr_in = d.filter(pl.col("split") == "train")
    old = pl.read_parquet(OUT / "duplicate_rows_numeric.parquet")
    tot_n, tot_k = n_tr, int(ids_tr["Response"].sum())
    k_in = int(tr_in["Response"].sum())
    newly = tr_in.filter(~pl.col("Id").is_in(old["Id"].to_list()))
    # 묶음 안에서 Id가 가장 작은 부품과 나머지 (train 부품만 불량률을 알 수 있다)
    dd = d.with_columns((pl.col("Id") == pl.col("Id").min().over("group")).alias("is_first")).filter(pl.col("split") == "train")
    pos = dd.group_by("is_first").agg(pl.len().alias("n"), pl.col("Response").sum().alias("k")).sort("is_first")
    table = pl.DataFrame({
        "part": ["train_with_twin_combined", "train_twin_only_in_test", "train_twin_in_train(3-5)", "train_without_twin_combined"],
        "n": [tr_in.height, newly.height, old.height, tot_n - tr_in.height],
        "k": [k_in, int(newly["Response"].sum()), int(old["Response"].sum()), tot_k - k_in],
    })
    res = {
        "candidate_rows": int(len(cand)), "hash_collision_rows": int((~keep).sum()),
        "n_groups": g.height, "n_parts": d.height, "group_kinds": kinds,
        "groups_of_size_2": int((g["size"] == 2).sum()), "group_size_max": int(g["size"].max()),
        "share_groups_ids_consecutive": float(g["ids_consecutive"].mean()),
        "share_groups_id_span_le_2": float((g["id_span"] <= 2).mean()),
        "train_parts_with_twin": tr_in.height, "share_train": tr_in.height / n_tr,
        "test_parts_with_twin": d.height - tr_in.height, "share_test": (d.height - tr_in.height) / n_te,
        "train_parts_with_twin_3_5": old.height,
        "train_3_5_parts_missing_in_combined": old.filter(~pl.col("Id").is_in(tr_in["Id"].to_list())).height,
        "rates": rate_table(table, ["part"]).to_dicts(),
        "share_of_all_defects": k_in / tot_k,
        "first_vs_later_train": rate_table(pos, ["is_first"]).to_dicts(),
    }
    update_summary("dup_combined", res)
    print(json.dumps(res, ensure_ascii=False, indent=1))


def stage_last_station() -> None:
    tr = pl.read_parquet(OUT / "station_time_train.parquet")
    st = stations_of(tr)
    resp = pl.read_parquet(OUT / "row_time_features.parquet", columns=["Response"])["Response"].to_numpy()
    mat = tr.select([f"{s}_tmin" for s in st]).to_numpy()
    last = last_station_index(mat)
    ok = last >= 0
    d = pl.DataFrame({"last_station": np.array(st)[last[ok]], "Response": resp[ok]})
    t = d.group_by("last_station").agg(pl.len().alias("n"), pl.col("Response").sum().alias("k"))
    t = rate_table(t, ["last_station"]).with_columns(
        pl.col("last_station").is_in(FLOW_ENDS).alias("is_flow_end"),
        pl.col("last_station").map_elements(station_no, return_dtype=pl.Int64).alias("no")).sort("no").drop("no")
    t.write_csv(OUT / "last_station_summary.csv")
    end = d["last_station"].is_in(FLOW_ENDS).to_numpy()
    r = d["Response"].to_numpy()
    summ = pl.DataFrame({"part": ["ends_at_flow_end", "ends_elsewhere"], "n": [int(end.sum()), int((~end).sum())],
                         "k": [int(r[end].sum()), int(r[~end].sum())]})
    res = {
        "parts_with_time": int(ok.sum()),
        "by_end": rate_table(summ, ["part"]).to_dicts(),
        "share_defects_ending_elsewhere": float(r[~end].sum() / r.sum()),
        "share_good_ending_elsewhere": float(((~end) & (r == 0)).sum() / (r == 0).sum()),
        "top_last_stations": t.sort("n", descending=True).head(8).select("last_station", "n", "k", "rate", "small_group").to_dicts(),
        "elsewhere_top": t.filter(~pl.col("is_flow_end")).sort("n", descending=True).head(8)
                          .select("last_station", "n", "k", "rate", "small_group").to_dicts(),
    }
    update_summary("last_station", res)
    print(json.dumps(res, ensure_ascii=False, indent=1))


def stage_pred_point() -> None:
    groups = groups_of("numeric")
    counts = compute_station_counts(PQ / "train_numeric.parquet", groups)
    pre_cols = [s for s in counts.columns if not s.startswith("L3_")]
    l3_cols = [s for s in counts.columns if s.startswith("L3_")]
    c = counts.select(pl.sum_horizontal(pre_cols).alias("pre"), pl.sum_horizontal(l3_cols).alias("l3"))
    pre, l3 = c["pre"].to_numpy().astype(np.int64), c["l3"].to_numpy().astype(np.int64)
    share = share_recorded_before(pre, pre + l3)
    tr = pl.read_parquet(OUT / "station_time_train.parquet")
    st = stations_of(tr)
    pre_st = [s for s in st if not s.startswith("L3_")]
    l3_st = [s for s in st if s.startswith("L3_")]
    t = tr.select(
        pl.max_horizontal([pl.col(f"{s}_tmax") for s in pre_st]).alias("pre_last"),
        pl.min_horizontal([pl.col(f"{s}_tmin") for s in l3_st]).alias("l3_first"),
        pl.max_horizontal([pl.col(f"{s}_tmax") for s in st]).alias("t_last"),
    ).with_columns((pl.col("t_last") - pl.col("pre_last")).alias("remain_after_pre"),
                   (pl.col("l3_first") - pl.col("pre_last")).alias("wait_before_l3"),
                   (pl.col("t_last") - pl.col("l3_first")).alias("time_in_l3"))
    resp = pl.read_parquet(OUT / "row_time_features.parquet", columns=["Response"])["Response"].to_numpy()
    both = (pre > 0) & (l3 > 0)
    only_l3 = (pre == 0) & (l3 > 0)
    only_pre = (pre > 0) & (l3 == 0)
    tb = t.filter(pl.Series(both))
    rows = [
        {"point": "before_L3", "n_parts": int(both.sum()), "k": int(resp[both].sum()),
         "share_of_numeric_values_recorded_median": float(np.median(share[both])),
         "share_recorded_p10": float(np.percentile(share[both], 10)), "share_recorded_p90": float(np.percentile(share[both], 90)),
         "remaining_time_median": float(tb["remain_after_pre"].median()),
         "remaining_time_p10": float(tb["remain_after_pre"].quantile(0.1)),
         "wait_before_l3_median": float(tb["wait_before_l3"].median()), "time_in_l3_median": float(tb["time_in_l3"].median())},
        {"point": "after_all_stations", "n_parts": int(((pre + l3) > 0).sum()), "k": int(resp[(pre + l3) > 0].sum()),
         "share_of_numeric_values_recorded_median": 1.0, "share_recorded_p10": 1.0, "share_recorded_p90": 1.0,
         "remaining_time_median": 0.0, "remaining_time_p10": 0.0, "wait_before_l3_median": None, "time_in_l3_median": None},
    ]
    pl.DataFrame(rows).write_csv(OUT / "prediction_point_summary.csv")
    res = {"points": rows, "parts_only_l3": int(only_l3.sum()), "defects_only_l3": int(resp[only_l3].sum()),
           "parts_no_l3": int(only_pre.sum()), "defects_no_l3": int(resp[only_pre].sum()),
           "n_numeric_columns_pre_l3": sum(len(groups[s]) for s in pre_cols),
           "n_numeric_columns_l3": sum(len(groups[s]) for s in l3_cols),
           "median_values_pre": float(np.median(pre[both])), "median_values_l3": float(np.median(l3[both]))}
    update_summary("pred_point", res)
    print(json.dumps(res, ensure_ascii=False, indent=1))


def stage_verify() -> None:
    res = {}
    s = json.loads((OUT / "q8_q9_summary.json").read_text())
    # 1) test의 스테이션별 최소·최대 시각: pyarrow로 전체 행을 독립 계산
    te = pl.read_parquet(OUT / "station_time_test.parquet")
    groups = groups_of("date")
    n = te.height
    bad_null = bad_val = 0
    for st, cols in groups.items():
        lo = np.full(n, np.nan)
        hi = np.full(n, np.nan)
        for i in range(0, len(cols), 50):
            tbl = pq.read_table(PQ / "test_date.parquet", columns=cols[i:i + 50])
            for c in tbl.columns:
                a = c.to_numpy(zero_copy_only=False).astype(np.float64)
                lo = np.fmin(lo, a)
                hi = np.fmax(hi, a)
            del tbl
        for mine, ref in ((te[f"{st}_tmin"], lo), (te[f"{st}_tmax"], hi)):
            m = mine.to_numpy().astype(np.float64)
            bad_null += int((np.isnan(m) != np.isnan(ref)).sum())
            both = ~np.isnan(m) & ~np.isnan(ref)
            bad_val += int((np.abs(m[both] - ref[both]) > 0.001).sum())
    res["test_times_full_check"] = {"rows": n, "stations": len(groups), "mismatch_null_cells": bad_null,
                                    "mismatch_value_cells": bad_val}
    # 2) 합친 짝: train 부품끼리의 짝이 3-5와 같은가
    d = pl.read_parquet(OUT / "duplicate_rows_combined.parquet")
    old = pl.read_parquet(OUT / "duplicate_rows_numeric.parquet")
    tr = d.filter(pl.col("split") == "train")
    # 합친 묶음에서 train 부품이 2개 이상인 묶음 = 3-5의 묶음이어야 한다
    multi = tr.group_by("group").agg(pl.col("Id").sort().alias("ids")).filter(pl.col("ids").list.len() >= 2)
    old_g = old.group_by("group").agg(pl.col("Id").sort().alias("ids"))
    a = {tuple(x) for x in multi["ids"].to_list()}
    b = {tuple(x) for x in old_g["ids"].to_list()}
    res["train_groups_vs_3_5"] = {"combined_train_groups": len(a), "groups_3_5": len(b), "same": a == b}
    # 합친 묶음 300개를 원본에서 Float64로 다시 읽어 비교
    rng = np.random.default_rng(SEED)
    gids = rng.choice(d["group"].unique().to_numpy(), 300, replace=False)
    sub = d.filter(pl.col("group").is_in(gids.tolist()))
    cols = [c for c in pq.ParquetFile(PQ / "test_numeric.parquet").schema_arrow.names if c != "Id"]
    raw = []
    for split, path in (("train", "train_numeric.parquet"), ("test", "test_numeric.parquet")):
        ids = sub.filter(pl.col("split") == split)["Id"].to_list()
        raw.append(pl.scan_parquet(PQ / path).filter(pl.col("Id").is_in(ids)).select(["Id"] + cols).collect(engine="streaming"))
    raw = sub.select("Id", "group").join(pl.concat(raw), on="Id")
    bad = 0
    for _, part in raw.group_by("group"):
        v = part.select(cols).to_numpy()
        bad += int(not (((v == v[0]) | (np.isnan(v) & np.isnan(v[0]))).all()))
    res["combined_groups_rechecked"] = {"groups": len(gids), "rows": raw.height, "rows_expected": sub.height, "mismatch": bad}
    # 3) 합계
    cmp_ = pl.read_csv(OUT / "train_test_comparison.csv", schema_overrides={"item": pl.String}).filter(pl.col("kind") == "time_bin_count")
    bins = pl.read_csv(OUT / "time_bin_summary.csv")
    res["time_bin_train_sum"] = {"sum": int(cmp_["train"].sum()), "expected": int(bins["n"].sum())}
    res["time_bin_test_sum"] = {"sum": int(cmp_["test"].sum()), "expected": n - s["test_times"]["rows_without_time"]}
    ls = pl.read_csv(OUT / "last_station_summary.csv")
    res["last_station_sum"] = {"sum": int(ls["n"].sum()), "expected": 1183165, "k_sum": int(ls["k"].sum()), "k_expected": 6877}
    ok = (bad_null == 0 and bad_val == 0 and res["train_groups_vs_3_5"]["same"] and bad == 0
          and raw.height == sub.height
          and all(res[k]["sum"] == res[k]["expected"] for k in ("time_bin_train_sum", "time_bin_test_sum", "last_station_sum"))
          and res["last_station_sum"]["k_sum"] == 6877)
    res["all_passed"] = bool(ok)
    update_summary("verify", res)
    print(json.dumps(res, ensure_ascii=False, indent=1))
    if not ok:
        raise SystemExit("검증 실패: 원인을 찾기 전에는 결과를 쓰지 않는다")


STAGES = {"test_times": stage_test_times, "compare": stage_compare, "dup_combined": stage_dup_combined,
          "last_station": stage_last_station, "pred_point": stage_pred_point, "verify": stage_verify}

if __name__ == "__main__":
    arg = sys.argv[1] if len(sys.argv) > 1 else "all"
    for name in (STAGES if arg == "all" else [arg]):
        print(f"=== {name} ===", flush=True)
        STAGES[name]()
        print(f"peak {peak_mib()} MiB", flush=True)
