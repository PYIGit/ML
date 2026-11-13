"""3-3 결측 구조와 공정 경로 (Q4). 설계: DOCS/03_EDA/3-3_MISSINGNESS_AND_ROUTES.md

입력: DATA/raw/bosch_parquet/train_{numeric,date,categorical}.parquet,
      DATA/interim/eda/{column_meta.parquet, row_features.parquet}
출력: DATA/interim/eda/ (Git 무시)
  station_visits_train.parquet, missing_rate_summary.csv, nonnull_per_row_dist.csv,
  station_visit_rate.csv, column_missing_types.parquet, missing_decomposition_by_station.csv,
  routes_{numeric,date,categorical}.parquet, line_combinations.csv, line_pair_conditional.csv,
  group12_station_share.csv, cross_file_consistency.csv, consistency_patterns.csv, q4_summary.json

단계(순서대로): visits -> missing -> routes -> consistency -> verify
사용: .venv/bin/python scripts/eda_3_3_missing_routes.py <stage|all>
"""
import json
import resource
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl
import pyarrow.parquet as pq

from ml.data.missing import cross_table, decompose_missing, missing_rate_bins
from ml.data.routes import (decode_route, line_combination_table, line_pair_conditional,
                            line_visited, route_keys, route_table, top_share)
from ml.data.visits import compute_station_counts, station_columns, visited_from_counts

PQ = Path("DATA/raw/bosch_parquet")
OUT = Path("DATA/interim/eda")
KINDS = ["numeric", "date", "categorical"]
MAX_COLS = {"numeric": 100, "date": 100, "categorical": 50}
TOP_NS = [1, 10, 100, 1000]


def peak_mib() -> int:
    return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)


def update_summary(key: str, value: dict) -> None:
    p = OUT / "q4_summary.json"
    cur = json.loads(p.read_text()) if p.exists() else {}
    cur[key] = value
    p.write_text(json.dumps(cur, indent=2, ensure_ascii=False))


def meta() -> pl.DataFrame:
    return pl.read_parquet(OUT / "column_meta.parquet")


def vcol(kind: str, station: str) -> str:
    return f"{kind}__{station}"


def load_visits() -> pl.DataFrame:
    return pl.read_parquet(OUT / "station_visits_train.parquet")


def kind_stations(v: pl.DataFrame, kind: str) -> list[str]:
    pre = f"{kind}__"
    return [c[len(pre):] for c in v.columns if c.startswith(pre)]


def kind_visited(v: pl.DataFrame, kind: str, stations: list[str] | None = None) -> pl.DataFrame:
    st = stations if stations is not None else kind_stations(v, kind)
    return v.select([pl.col(vcol(kind, s)).alias(s) for s in st])


def line_of(station: str) -> int:
    return int(station.split("_")[0][1:])


def stage_visits() -> None:
    t0 = time.time()
    m = meta()
    ids = pl.scan_parquet(PQ / "train_numeric.parquet").select("Id", "Response").collect()
    out = {"Id": ids["Id"], "Response": ids["Response"]}
    info = {}
    for kind in KINDS:
        tk = time.time()
        groups = station_columns(m, kind)
        counts = compute_station_counts(PQ / f"train_{kind}.parquet", groups, MAX_COLS[kind])
        if counts.height != ids.height:
            raise SystemExit(f"{kind}: 행 수 {counts.height} != {ids.height}")
        vis = visited_from_counts(counts)
        for s in vis.columns:
            out[vcol(kind, s)] = vis[s]
        total = counts.select(pl.sum_horizontal(pl.all().cast(pl.Int32)).alias("n")).to_series()
        out[f"n_nonnull_{kind}"] = total
        info[kind] = {"stations": len(groups), "columns": sum(len(c) for c in groups.values()),
                      "seconds": round(time.time() - tk, 1), "peak_rss_mib_so_far": peak_mib()}
        print(f"{kind}: 스테이션 {len(groups)}개, 컬럼 {info[kind]['columns']}개, "
              f"{info[kind]['seconds']}s, 피크 {info[kind]['peak_rss_mib_so_far']} MiB", flush=True)
        del counts
    df = pl.DataFrame(out)
    df.write_parquet(OUT / "station_visits_train.parquet", compression="zstd")
    info["rows"] = df.height
    info["columns"] = df.width
    info["bytes"] = (OUT / "station_visits_train.parquet").stat().st_size
    info["seconds"] = round(time.time() - t0, 1)
    info["peak_rss_mib"] = peak_mib()
    update_summary("visits", info)
    print(f"visits 완료 {info['seconds']}s, 피크 {info['peak_rss_mib']} MiB, "
          f"{df.height:,}행 x {df.width}열, {info['bytes']/2**20:.1f} MiB", flush=True)


def stage_missing() -> None:
    t0 = time.time()
    m, v = meta(), load_visits()
    n = v.height
    bins = missing_rate_bins(m)
    bins.write_csv(OUT / "missing_rate_summary.csv")
    print(bins, flush=True)

    dist, rate_rows, nv, summ = [], [], {}, {}
    for kind in KINDS:
        c = v[f"n_nonnull_{kind}"]
        d = c.value_counts().sort(c.name).rename({c.name: "n_nonnull"}).with_columns(
            pl.lit(kind).alias("data_kind"))
        dist.append(d.select("data_kind", "n_nonnull", "count"))
        q = c.quantile
        summ[kind] = {"min": int(c.min()), "median": float(c.median()), "mean": round(float(c.mean()), 2),
                      "p95": float(q(0.95)), "max": int(c.max()), "n_zero_rows": int((c == 0).sum())}
        for s in kind_stations(v, kind):
            nvis = int(v[vcol(kind, s)].sum())
            nv[(kind, s)] = n - nvis
            rate_rows.append({"data_kind": kind, "station_key": s, "line": line_of(s),
                              "n_visited": nvis, "visit_rate": nvis / n})
    pl.concat(dist).write_csv(OUT / "nonnull_per_row_dist.csv")
    pl.DataFrame(rate_rows).write_csv(OUT / "station_visit_rate.csv")

    dec = decompose_missing(m, nv)
    dec.write_parquet(OUT / "column_missing_types.parquet")
    agg = (dec.group_by("data_kind")
           .agg(pl.len().alias("n_columns"), pl.col("null_count").sum().alias("null_total"),
                pl.col("structural_missing").sum().alias("structural"),
                pl.col("sporadic_missing").sum().alias("sporadic"),
                (pl.col("n_rows") * 1).sum().alias("cells"))
           .with_columns((pl.col("structural") / pl.col("null_total")).alias("structural_share"),
                         (pl.col("sporadic") / pl.col("null_total")).alias("sporadic_share"))
           .sort("data_kind"))
    # 산발적 결측이 있는 컬럼 수
    ncol_spor = dec.group_by("data_kind").agg(
        (pl.col("sporadic_missing") > 0).sum().alias("columns_with_sporadic"),
        (pl.col("sporadic_missing") == 0).sum().alias("columns_no_sporadic"))
    agg = agg.join(ncol_spor, on="data_kind")
    print(agg, flush=True)
    by_station = (dec.group_by(["data_kind", "station_key"])
                  .agg(pl.len().alias("n_columns"), pl.col("null_count").sum().alias("null_total"),
                       pl.col("structural_missing").sum().alias("structural"),
                       pl.col("sporadic_missing").sum().alias("sporadic"))
                  .with_columns((pl.col("sporadic") / pl.col("null_total")).alias("sporadic_share"))
                  .sort(["data_kind", "station_key"]))
    by_station.write_csv(OUT / "missing_decomposition_by_station.csv")
    info = {"bins": bins.to_dicts(), "nonnull_per_row": summ,
            "decomposition_by_kind": agg.to_dicts(),
            "sum_check_ok": bool(((dec["structural_missing"] + dec["sporadic_missing"])
                                  == dec["null_count"]).all()),
            "negative_sporadic_columns": int((dec["sporadic_missing"] < 0).sum()),
            "seconds": round(time.time() - t0, 1), "peak_rss_mib": peak_mib()}
    update_summary("missing", info)
    print(json.dumps(summ, ensure_ascii=False), flush=True)


def stage_routes() -> None:
    t0 = time.time()
    v = load_visits()
    y = v["Response"].to_numpy()
    info = {}
    for kind in KINDS:
        st = kind_stations(v, kind)
        keys = route_keys(kind_visited(v, kind, st))
        tab = route_table(keys, y, st)
        tab.write_parquet(OUT / f"routes_{kind}.parquet")
        small = tab.filter(pl.col("small_group"))
        r12 = tab.filter(pl.col("n_stations") == 12)
        info[kind] = {
            "n_routes": tab.height, "n_total": int(tab["n"].sum()), "k_total": int(tab["k"].sum()),
            "top_share": {str(k): round(top_share(tab, k), 4) for k in TOP_NS},
            "routes_small": small.height, "parts_in_small": int(small["n"].sum()),
            "defects_in_small": int(small["k"].sum()),
            "routes_n1": int(tab.filter(pl.col("n") == 1).height),
            "n_stations_max": int(tab["n_stations"].max()),
            "top5": tab.head(5).select("n_stations", "n", "k", "rate", "ci_low", "ci_high").to_dicts(),
        }
        if kind == "numeric":
            tot12 = int(r12["n"].sum())
            info[kind]["n_stations_12"] = {
                "n_routes": r12.height, "n_parts": tot12, "n_defects": int(r12["k"].sum()),
                "top_route_share_in_group": float(r12["n"].max() / tot12) if tot12 else None,
                "top3": r12.head(3).select("n", "k", "rate", "ci_low", "ci_high", "stations").to_dicts()}
        print(f"{kind}: 경로 {tab.height:,}개, 상위 10 점유 {info[kind]['top_share']['10']:.3f}, "
              f"소표본 경로 {small.height:,}개({int(small['n'].sum()):,}부품)", flush=True)

    # 방문 스테이션 수 12인 그룹(3-2에서 불량률 6.09%)의 스테이션별 방문 비율과 라인 조합
    st = kind_stations(v, "numeric")
    vis = kind_visited(v, "numeric", st)
    n_st = vis.select(pl.sum_horizontal(pl.all().cast(pl.Int16))).to_series()
    mask = (n_st == 12).to_numpy()
    g12 = vis.filter(pl.Series(mask))
    share = {c: float(g12[c].mean()) for c in st}
    always = {c: round(x, 4) for c, x in share.items() if x >= 0.9}
    all_share = {c: float(vis[c].mean()) for c in st}
    lines12 = line_visited(g12, {s_: line_of(s_) for s_ in st})
    comb12 = line_combination_table(lines12, y[mask])
    info["numeric"]["n_stations_12"].update({
        "stations_visited_by_90pct": always,
        "stations_visited_by_90pct_overall": {c: round(all_share[c], 4) for c in always},
        "line_combinations": comb12.head(6).to_dicts()})
    pl.DataFrame({"station_key": st, "share_in_group12": [share[c] for c in st],
                  "share_overall": [all_share[c] for c in st]}).write_csv(OUT / "group12_station_share.csv")

    # 라인 조합(대표: numeric)
    lines = line_visited(kind_visited(v, "numeric", st), {s: line_of(s) for s in st})
    comb = line_combination_table(lines, y)
    comb.write_csv(OUT / "line_combinations.csv")
    line_pair_conditional(lines).write_csv(OUT / "line_pair_conditional.csv")
    info["line_combinations"] = comb.to_dicts()
    info["line_pair_conditional"] = line_pair_conditional(lines).to_dicts()
    print(comb, flush=True)
    info["seconds"] = round(time.time() - t0, 1)
    info["peak_rss_mib"] = peak_mib()
    update_summary("routes", info)


def stage_consistency() -> None:
    t0 = time.time()
    v = load_visits()
    n = v.height
    st = {k: kind_stations(v, k) for k in KINDS}
    pairs = [("numeric", "date"), ("numeric", "categorical"), ("date", "categorical")]
    rows = []
    for a, b in pairs:
        for s in sorted(set(st[a]) & set(st[b]), key=lambda s: (line_of(s), int(s.split("_S")[1]))):
            rows.append({"pair": f"{a}-{b}", "station_key": s,
                         **cross_table(v[vcol(a, s)], v[vcol(b, s)])})
    cons = pl.DataFrame(rows).with_columns(
        (pl.col("a_only") + pl.col("b_only")).alias("mismatch"))
    cons = cons.with_columns((pl.col("mismatch") / n).alias("mismatch_rate"))
    cons.write_csv(OUT / "cross_file_consistency.csv")

    only_one = {k: sorted(set(st[k]) - set().union(*[set(st[o]) for o in KINDS if o != k]))
                for k in KINDS}
    info = {"station_level": {}, "only_in_one_file": only_one}
    for a, b in pairs:
        sub = cons.filter(pl.col("pair") == f"{a}-{b}")
        info["station_level"][f"{a}-{b}"] = {
            "common_stations": sub.height,
            "stations_all_consistent": int((sub["mismatch"] == 0).sum()),
            "mismatch_cells": int(sub["mismatch"].sum()),
            "a_only": int(sub["a_only"].sum()), "b_only": int(sub["b_only"].sum()),
            "worst": sub.sort("mismatch", descending=True).head(5)
                        .select("station_key", "both", "a_only", "b_only", "neither").to_dicts()}

    # 부품 단위: 공통 스테이션의 경로가 같은가
    pat_rows = []
    info["part_level"] = {}
    for a, b in pairs:
        common = sorted(set(st[a]) & set(st[b]), key=st[a].index)
        ka = route_keys(kind_visited(v, a, common))
        kb = route_keys(kind_visited(v, b, common))
        diff = ka ^ kb
        info["part_level"][f"{a}-{b}"] = {
            "common_stations": len(common), "parts_same_route": int((diff == 0).sum()),
            "same_share": float((diff == 0).mean())}
        g = (pl.DataFrame({"x": diff}).filter(pl.col("x") != 0).group_by("x").len()
             .sort("len", descending=True).head(10))
        for x, c in g.iter_rows():
            pat_rows.append({"pair": f"{a}-{b}", "n_parts": c,
                             "mismatch_stations": ",".join(decode_route(x, common))})
    pl.DataFrame(pat_rows).write_csv(OUT / "consistency_patterns.csv")
    info["patterns_top3"] = {p: [r for r in pat_rows if r["pair"] == p][:3]
                             for p in [f"{a}-{b}" for a, b in pairs]}
    # 세 파일 모두에 있는 스테이션에서 세 기준이 모두 같은 부품
    common3 = [s for s in st["numeric"] if s in st["date"] and s in st["categorical"]]
    k3 = [route_keys(kind_visited(v, k, common3)) for k in KINDS]
    info["part_level"]["all_three"] = {
        "common_stations": len(common3),
        "parts_same_route": int(((k3[0] == k3[1]) & (k3[1] == k3[2])).sum())}

    # numeric 값이 하나도 없는 행: date·categorical 기준 방문
    z = v.filter(pl.col("n_nonnull_numeric") == 0)
    n_date_st = z.select(pl.sum_horizontal([pl.col(vcol("date", s)).cast(pl.Int16) for s in st["date"]])).to_series()
    n_cat_st = z.select(pl.sum_horizontal([pl.col(vcol("categorical", s)).cast(pl.Int16) for s in st["categorical"]])).to_series()
    info["no_numeric_rows"] = {
        "n_rows": z.height, "n_defects": int(z["Response"].sum()),
        "date_visit_stations_mean": round(float(n_date_st.mean()), 3),
        "date_visit_stations_max": int(n_date_st.max()),
        "rows_with_date_visit": int((n_date_st > 0).sum()),
        "rows_with_categorical_visit": int((n_cat_st > 0).sum()),
        "rows_all_empty": int(((z["n_nonnull_date"] == 0) & (z["n_nonnull_categorical"] == 0)).sum())}
    info["seconds"] = round(time.time() - t0, 1)
    info["peak_rss_mib"] = peak_mib()
    update_summary("consistency", info)
    print(json.dumps({k: info[k] for k in ["part_level", "no_numeric_rows", "only_in_one_file"]},
                     ensure_ascii=False, indent=1), flush=True)


def stage_verify() -> None:
    t0 = time.time()
    v = load_visits()
    n = v.height
    res = {}
    rf = pl.read_parquet(OUT / "row_features.parquet")
    res["id_order_equal_row_features"] = bool(v["Id"].equals(rf["Id"]))
    st = kind_stations(v, "numeric")
    vis = kind_visited(v, "numeric", st)
    n_st = vis.select(pl.sum_horizontal(pl.all().cast(pl.Int16))).to_series()
    n_ln = line_visited(vis, {s: line_of(s) for s in st}).select(
        pl.sum_horizontal(pl.all().cast(pl.Int16))).to_series()
    res["numeric_vs_3_2"] = {
        "n_nonnull_equal": bool((v["n_nonnull_numeric"].cast(pl.Int32) == rf["n_nonnull_numeric"].cast(pl.Int32)).all()),
        "n_stations_equal": bool((n_st == rf["n_stations_visited_numeric"].cast(pl.Int16)).all()),
        "n_lines_equal": bool((n_ln == rf["n_lines_visited_numeric"].cast(pl.Int16)).all()),
        "rows": n}
    for kind in KINDS:
        t = pl.read_parquet(OUT / f"routes_{kind}.parquet")
        res[f"routes_{kind}"] = {
            "n_sum_ok": int(t["n"].sum()) == n, "k_sum_ok": int(t["k"].sum()) == int(v["Response"].sum()),
            "k_total": int(t["k"].sum())}
    # 경로의 스테이션 수 분포 == 행별 방문 스테이션 수 분포 (numeric)
    t = pl.read_parquet(OUT / "routes_numeric.parquet")
    d1 = t.group_by("n_stations").agg(pl.col("n").sum()).sort("n_stations")
    d2 = n_st.value_counts().sort(n_st.name).rename({n_st.name: "n_stations", "count": "n"})
    res["route_station_count_dist_equals_row_dist"] = bool(
        d1["n_stations"].cast(pl.Int64).to_list() == d2["n_stations"].cast(pl.Int64).to_list()
        and d1["n"].to_list() == d2["n"].to_list())
    res["decomposition_sum_ok"] = json.loads((OUT / "q4_summary.json").read_text())["missing"]["sum_check_ok"]

    # date, categorical: pyarrow로 전체 행을 독립 계산 (polars 코드와 다른 경로)
    m = meta()
    for kind in ["date", "categorical"]:
        tk = time.time()
        groups = station_columns(m, kind)
        pf = PQ / f"train_{kind}.parquet"
        bad_vis = bad_tot_rows = 0
        total = np.zeros(n, dtype=np.int32)
        for s, cols in groups.items():
            cnt = np.zeros(n, dtype=np.int16)
            for i in range(0, len(cols), 50):
                tbl = pq.read_table(pf, columns=cols[i:i + 50])
                for c in tbl.columns:
                    cnt += np.asarray(c.is_valid().to_numpy(zero_copy_only=False), dtype=np.int16)
                del tbl
            bad_vis += int(((cnt > 0) != v[vcol(kind, s)].to_numpy()).sum())
            total += cnt
        bad_tot_rows = int((total != v[f"n_nonnull_{kind}"].to_numpy()).sum())
        res[f"full_check_{kind}"] = {"rows": n, "stations": len(groups),
                                     "mismatch_visit_cells": bad_vis, "mismatch_total_rows": bad_tot_rows,
                                     "seconds": round(time.time() - tk, 1)}
        print(kind, res[f"full_check_{kind}"], flush=True)
    ok = (res["id_order_equal_row_features"] and all(res["numeric_vs_3_2"][k] for k in
          ["n_nonnull_equal", "n_stations_equal", "n_lines_equal"])
          and all(res[f"routes_{k}"]["n_sum_ok"] and res[f"routes_{k}"]["k_sum_ok"] for k in KINDS)
          and res["route_station_count_dist_equals_row_dist"] and res["decomposition_sum_ok"]
          and all(res[f"full_check_{k}"]["mismatch_visit_cells"] == 0
                  and res[f"full_check_{k}"]["mismatch_total_rows"] == 0 for k in ["date", "categorical"]))
    res["all_passed"] = bool(ok)
    res["seconds"] = round(time.time() - t0, 1)
    res["peak_rss_mib"] = peak_mib()
    update_summary("verify", res)
    print(json.dumps(res, ensure_ascii=False, indent=1))
    if not ok:
        raise SystemExit("검증 실패: 원인을 찾기 전에는 결과를 쓰지 않는다")


STAGES = {"visits": stage_visits, "missing": stage_missing, "routes": stage_routes,
          "consistency": stage_consistency, "verify": stage_verify}

if __name__ == "__main__":
    arg = sys.argv[1] if len(sys.argv) > 1 else "all"
    for name in (STAGES if arg == "all" else [arg]):
        print(f"=== {name} ===", flush=True)
        STAGES[name]()
