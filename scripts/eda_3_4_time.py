"""3-4 시간(date) 구조 (Q5). 설계: DOCS/03_EDA/3-4_TIME_STRUCTURE.md

입력: DATA/raw/bosch_parquet/train_date.parquet,
      DATA/interim/eda/{column_meta.parquet, row_features.parquet, station_visits_train.parquet}
출력: DATA/interim/eda/ (Git 무시)
  station_time_train.parquet, row_time_features.parquet, station_time_consistency.csv,
  station_transitions.csv, time_bin_summary.csv, group12_time.csv, q5_summary.json

단계(순서대로): station_times -> features -> flow -> trend -> id_order -> group12 -> verify
사용: .venv/bin/python scripts/eda_3_4_time.py <stage|all>
"""
import json
import resource
import sys
from pathlib import Path

import numpy as np
import polars as pl
import pyarrow.parquet as pq

from ml.data.timing import (assign_time_bin, autocorrelation, burst_test, compute_station_time_range,
                            dominant_period, part_time_features, station_consistency, transition_table,
                            transitions_long)
from ml.data.visits import station_columns
from ml.stats import rate_table, wilson_interval

PQ = Path("DATA/raw/bosch_parquet")
OUT = Path("DATA/interim/eda")
BIN_WIDTH = 10.0
N_SIM = 5000
SEED = 20261112
GROUP_N_STATIONS = 12
PAIRS = [("L3_S35", "L3_S36"), ("L1_S24", "L1_S25"), ("L2_S26", "L2_S27")]


def peak_mib() -> int:
    return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)


def update_summary(key: str, value: dict) -> None:
    p = OUT / "q5_summary.json"
    cur = json.loads(p.read_text()) if p.exists() else {}
    value["peak_rss_mib"] = peak_mib()
    cur[key] = value
    p.write_text(json.dumps(cur, indent=2, ensure_ascii=False))


def station_no(key: str) -> int:
    return int(key.split("_S")[1])


def date_groups() -> dict[str, list[str]]:
    """date 파일의 스테이션별 컬럼 목록. 스테이션 번호순."""
    g = station_columns(pl.read_parquet(OUT / "column_meta.parquet"), "date")
    return {k: g[k] for k in sorted(g, key=station_no)}


def load_times() -> tuple[pl.DataFrame, list[str]]:
    t = pl.read_parquet(OUT / "station_time_train.parquet")
    return t, [c[:-5] for c in t.columns if c.endswith("_tmin")]


def load_feats() -> pl.DataFrame:
    return pl.read_parquet(OUT / "row_time_features.parquet")


def stage_station_times() -> None:
    groups = date_groups()
    stations = list(groups)
    times = compute_station_time_range(PQ / "train_date.parquet", groups)
    ids = pl.read_parquet(PQ / "train_date.parquet", columns=["Id"])["Id"]
    times.insert_column(0, ids).write_parquet(OUT / "station_time_train.parquet")
    cons = station_consistency(times, stations)
    cons = cons.with_columns(pl.Series("n_date_columns", [len(groups[s]) for s in stations]))
    cons.write_csv(OUT / "station_time_consistency.csv")
    # 값의 범위와 가장 작은 간격
    vals = np.unique(np.concatenate([times[c].drop_nulls().unique().to_numpy() for c in times.columns if c != "Id"]))
    vals = np.unique(np.round(vals.astype(np.float64), 2))
    gaps = np.round(np.diff(vals), 2)
    multi = cons.filter(pl.col("n_date_columns") > 1)
    info = {
        "rows": times.height, "stations": len(stations),
        "value_min": float(vals.min()), "value_max": float(vals.max()),
        "n_distinct_station_times": int(len(vals)), "smallest_gap": float(gaps.min()),
        "share_of_gaps_equal_smallest": float((gaps == gaps.min()).mean()),
        "stations_with_multiple_columns": multi.height,
        "stations_all_parts_same_value": int((multi["n_differ"] == 0).sum()),
        "stations_with_differences": cons.filter(pl.col("n_differ") > 0).select(
            "station_key", "n_date_columns", "n_with_time", "n_differ", "max_diff",
            "median_diff_when_differ").to_dicts(),
    }
    update_summary("station_times", info)
    print(json.dumps(info, ensure_ascii=False, indent=1), flush=True)


def stage_features() -> None:
    times, stations = load_times()
    rf = pl.read_parquet(OUT / "row_features.parquet", columns=["Id", "Response", "n_stations_visited_numeric"])
    if not times["Id"].equals(rf["Id"]):
        raise SystemExit("Id 순서가 row_features.parquet과 다름")
    f = part_time_features(times, stations)
    f = f.with_columns(assign_time_bin(f["t_first"], BIN_WIDTH))
    f = pl.concat([rf.select("Id", "Response"), f], how="horizontal")
    f.write_parquet(OUT / "row_time_features.parquet")
    has = f.filter(pl.col("t_first").is_not_null())
    # 생산량의 반복 주기: 폭 0.1 구간의 부품 수 계열의 자기상관
    w = 0.1
    idx = (has["t_first"] / w).round(0).cast(pl.Int64).to_numpy()
    counts = np.bincount(idx, minlength=idx.max() + 1)
    acf = autocorrelation(counts, max_lag=600)
    peak = dominant_period(acf)
    peaks = [int(lag) for lag in range(2, 599) if acf[lag] > 0.15 and acf[lag] >= acf[lag - 1] and acf[lag] > acf[lag + 1]
             and acf[lag] == acf[max(1, lag - 15):lag + 16].max()]
    info = {
        "rows": f.height, "rows_without_time": f.height - has.height,
        "defects_without_time": int(f.filter(pl.col("t_first").is_null())["Response"].sum()),
        "t_first_min": float(has["t_first"].min()), "t_first_max": float(has["t_first"].max()),
        "duration_median": float(has["duration"].median()),
        "duration_p90": float(has["duration"].quantile(0.9)),
        "share_duration_zero": float((has["duration"] == 0).mean()),
        "period_bin_width": w,
        "period": None if peak is None else round(peak * w, 2),
        "period_acf": None if peak is None else round(float(acf[peak]), 3),
        "acf_at_one_seventh_of_period": None if peak is None else round(float(acf[round(peak / 7)]), 3),
        "value_range_in_periods": None if peak is None else round(float(has["t_last"].max()) / (peak * w), 1),
        "acf_local_peaks": [{"lag": round(p * w, 2), "acf": round(float(acf[p]), 3)} for p in peaks[:12]],
        "n_time_bins": int(has["time_bin"].n_unique()), "bin_width": BIN_WIDTH,
    }
    update_summary("features", info)
    print(json.dumps(info, ensure_ascii=False, indent=1), flush=True)


def stage_flow() -> None:
    times, stations = load_times()
    mat = times.select([f"{s}_tmin" for s in stations]).to_numpy()
    n_with = (~np.isnan(mat)).sum(axis=1)
    long = transitions_long(mat)
    del mat
    table = transition_table(long, stations)
    table.drop("from_idx", "to_idx").write_csv(OUT / "station_transitions.csv")
    total = long.height
    out_n = table.group_by("from_station").agg(pl.col("n").sum().alias("out_total"))
    in_n = table.group_by("to_station").agg(pl.col("n").sum().alias("in_total"))
    nxt = (table.join(out_n, on="from_station").with_columns((pl.col("n") / pl.col("out_total")).alias("share"))
           .sort("n", descending=True))
    prv = (table.join(in_n, on="to_station").with_columns((pl.col("n") / pl.col("in_total")).alias("share"))
           .sort("n", descending=True))

    def top(df, key, s, other, k=3):
        d = df.filter(pl.col(key) == s).head(k)
        return [{"station": r[other], "share": round(r["share"], 4), "n": r["n"], "dt_median": r["dt_median"],
                 "same_time_share": round(r["same_time_share"], 4)} for r in d.to_dicts()]

    info = {
        "transitions_total": total,
        "expected_total": int(np.clip(n_with - 1, 0, None).sum()),
        "distinct_transitions": table.height,
        "share_against_number_order": float(table.filter(pl.col("against_number_order"))["n"].sum() / total),
        "share_same_time": float((long["dt"] == 0).mean()),
        "dt_median_all": float(long["dt"].median()),
        "top_against_number_order": table.filter(pl.col("against_number_order")).head(8).select(
            "from_station", "to_station", "n", "dt_median").to_dicts(),
        "next_station": {s: top(nxt, "from_station", s, "to_station") for s in stations},
        "prev_station": {s: top(prv, "to_station", s, "from_station") for s in stations},
        "pairs": {f"{a}|{b}": {a: {"prev": top(prv, "to_station", a, "from_station", 2),
                                   "next": top(nxt, "from_station", a, "to_station", 2)},
                               b: {"prev": top(prv, "to_station", b, "from_station", 2),
                                   "next": top(nxt, "from_station", b, "to_station", 2)}} for a, b in PAIRS},
    }
    update_summary("flow", info)
    print(json.dumps({k: v for k, v in info.items() if k not in ("next_station", "prev_station")},
                     ensure_ascii=False, indent=1), flush=True)


def bin_counts(f: pl.DataFrame) -> pl.DataFrame:
    return (f.filter(pl.col("time_bin").is_not_null()).group_by("time_bin")
            .agg(pl.len().alias("n"), pl.col("Response").cast(pl.Int64).sum().alias("k")).sort("time_bin"))


def runs(flags: list[bool], bins: list[int]) -> list[tuple[int, int]]:
    """True가 이어진 구간(구간 번호가 연속인 것)의 (시작, 끝) 목록."""
    out, start, prev = [], None, None
    for b, fl in zip(bins, flags):
        if fl and start is not None and b == prev + 1:
            prev = b
        elif fl:
            if start is not None:
                out.append((start, prev))
            start = prev = b
        else:
            if start is not None:
                out.append((start, prev))
            start = prev = None
    if start is not None:
        out.append((start, prev))
    return out


def stage_trend() -> None:
    f = load_feats()
    b = bin_counts(f)
    bt = burst_test(b["n"].to_numpy(), b["k"].to_numpy(), n_sim=N_SIM, seed=SEED)
    t = rate_table(b, ["time_bin"]).with_columns(
        (pl.col("time_bin") * BIN_WIDTH).alias("t_start"),
        pl.Series("expected_k", bt["expected"]), pl.Series("sim_lo", bt["sim_lo"]), pl.Series("sim_hi", bt["sim_hi"]),
    ).with_columns(
        pl.when(pl.col("k") > pl.col("sim_hi")).then(pl.lit("high"))
        .when(pl.col("k") < pl.col("sim_lo")).then(pl.lit("low")).otherwise(pl.lit("")).alias("outside"))
    t.select("time_bin", "t_start", "n", "k", "rate", "ci_low", "ci_high", "small_group", "expected_k", "sim_lo",
             "sim_hi", "outside").write_csv(OUT / "time_bin_summary.csv")
    bins = t["time_bin"].to_list()
    high = runs([o == "high" for o in t["outside"].to_list()], bins)
    hi_t = t.filter(pl.col("outside") == "high")

    def run_info(s, e):
        d = t.filter(pl.col("time_bin").is_between(s, e))
        return {"t_start": s * BIN_WIDTH, "t_end": (e + 1) * BIN_WIDTH, "n_bins": e - s + 1, "n": int(d["n"].sum()),
                "k": int(d["k"].sum()), "rate": float(d["k"].sum() / d["n"].sum())}

    total_n, total_k = int(t["n"].sum()), int(t["k"].sum())
    info = {
        "bin_width": BIN_WIDTH, "n_bins": t.height, "n_small_bins": int(t["small_group"].sum()),
        "total_n": total_n, "total_k": total_k, "overall_rate": total_k / total_n,
        "statistic": bt["statistic"], "sim_statistic_median": bt["sim_statistic_median"],
        "sim_statistic_max": bt["sim_statistic_max"], "p_value": bt["p_value"], "n_sim": N_SIM, "seed": SEED,
        "n_bins_high": hi_t.height, "n_bins_low": t.filter(pl.col("outside") == "low").height,
        "bins_expected_outside_by_chance": round(t.height * 0.05, 1),
        "share_of_parts_in_high_bins": float(hi_t["n"].sum() / total_n),
        "share_of_defects_in_high_bins": float(hi_t["k"].sum() / total_k),
        "rate_in_high_bins": float(hi_t["k"].sum() / hi_t["n"].sum()) if hi_t.height else None,
        "rate_outside_high_bins": float((total_k - hi_t["k"].sum()) / (total_n - hi_t["n"].sum())),
        "high_runs": sorted([run_info(s, e) for s, e in high], key=lambda r: -r["k"])[:12],
        "top_rate_bins_not_small": t.filter(~pl.col("small_group")).sort("rate", descending=True).head(8).select(
            "t_start", "n", "k", "rate", "ci_low", "ci_high").to_dicts(),
        "rate_first_half": float(t.filter(pl.col("t_start") < 860)["k"].sum() / t.filter(pl.col("t_start") < 860)["n"].sum()),
        "rate_second_half": float(t.filter(pl.col("t_start") >= 860)["k"].sum() / t.filter(pl.col("t_start") >= 860)["n"].sum()),
    }
    update_summary("trend", info)
    print(json.dumps(info, ensure_ascii=False, indent=1), flush=True)


def stage_id_order() -> None:
    f = load_feats().filter(pl.col("t_first").is_not_null()).sort("Id")
    rho = f.select(pl.corr(pl.col("Id").rank(), pl.col("t_first").rank())).item()
    t = f["t_first"].to_numpy()
    adj = np.abs(np.diff(t))
    rng = np.random.default_rng(SEED)
    rnd = np.abs(t - rng.permutation(t))
    info = {
        "rows": f.height, "spearman_id_vs_t_first": float(rho),
        "adjacent_id_abs_dt_median": float(np.median(adj)),
        "random_pair_abs_dt_median": float(np.median(rnd)),
        "adjacent_id_share_same_time": float((adj == 0).mean()),
        "random_pair_share_same_time": float((rnd == 0).mean()),
        "adjacent_id_share_time_not_decreasing": float((np.diff(t) >= 0).mean()),
    }
    update_summary("id_order", info)
    print(json.dumps(info, ensure_ascii=False, indent=1), flush=True)


def stage_group12() -> None:
    f = load_feats()
    rf = pl.read_parquet(OUT / "row_features.parquet", columns=["Id", "n_stations_visited_numeric"])
    f = f.with_columns(rf["n_stations_visited_numeric"]).filter(pl.col("time_bin").is_not_null())
    g = f.filter(pl.col("n_stations_visited_numeric") == GROUP_N_STATIONS)
    allb = pl.read_csv(OUT / "time_bin_summary.csv").select(
        "time_bin", "t_start", pl.col("n").alias("n_all"), pl.col("k").alias("k_all"), "outside")
    gb = rate_table(bin_counts(g), ["time_bin"]).join(allb, on="time_bin", how="left").with_columns(
        (pl.col("n") / pl.col("n_all")).alias("share_of_bin_parts"),
        (pl.col("k") / pl.col("k_all")).alias("share_of_bin_defects")).sort("time_bin")
    gb.select("time_bin", "t_start", "n", "k", "rate", "ci_low", "ci_high", "small_group", "n_all", "k_all",
              "share_of_bin_parts", "share_of_bin_defects", "outside").write_csv(OUT / "group12_time.csv")
    n, k = int(gb["n"].sum()), int(gb["k"].sum())
    hi = gb.filter(pl.col("outside") == "high")
    all_hi = allb.filter(pl.col("outside") == "high")
    srt = gb.sort("k", descending=True)
    cum = np.cumsum(srt["k"].to_numpy())
    rest = f.filter(pl.col("n_stations_visited_numeric") != GROUP_N_STATIONS)
    rest_hi = rest.join(all_hi.select("time_bin"), on="time_bin", how="inner")
    info = {
        "group_n": n, "group_k": k, "group_rate": k / n, "n_bins_with_group_parts": gb.height,
        "t_min": float(g["t_first"].min()), "t_max": float(g["t_first"].max()),
        "share_group_parts_in_high_bins": float(hi["n"].sum() / n),
        "share_group_defects_in_high_bins": float(hi["k"].sum() / k),
        "group_rate_in_high_bins": float(hi["k"].sum() / hi["n"].sum()) if hi.height else None,
        "group_rate_outside_high_bins": float((k - hi["k"].sum()) / (n - hi["n"].sum())),
        "bins_holding_half_of_group_defects": int(np.searchsorted(cum, k / 2) + 1),
        "group_share_of_defects_in_high_bins_overall": float(hi["k"].sum() / all_hi["k_all"].sum()),
        "rest_rate_in_high_bins": float(rest_hi["Response"].sum() / rest_hi.height),
        "rest_rate_overall": float(rest["Response"].sum() / rest.height),
        "top_bins_by_group_defects": srt.head(8).select("t_start", "n", "k", "rate", "n_all", "k_all", "outside").to_dicts(),
    }
    # 그룹을 L3_S32 방문 여부로 나눠 본다 (시기로 설명되지 않아 추가, 3-4 문서 6절)
    v = pl.read_parquet(OUT / "station_visits_train.parquet",
                        columns=["Id", "numeric__L3_S32", "numeric__L3_S33", "numeric__L3_S34"])
    full = load_feats().with_columns(rf["n_stations_visited_numeric"], v["numeric__L3_S32"].alias("s32"),
                                     v["numeric__L3_S33"].alias("s33"), v["numeric__L3_S34"].alias("s34"))

    def nk(df):
        n_, k_ = df.height, int(df["Response"].sum())
        lo_, hi_ = wilson_interval(k_, n_)
        return {"n": n_, "k": k_, "rate": k_ / n_ if n_ else None, "ci_low": lo_, "ci_high": hi_}

    is_g = pl.col("n_stations_visited_numeric") == GROUP_N_STATIONS
    skip = pl.col("s32") & ~pl.col("s33")
    timed = full.filter(pl.col("time_bin").is_not_null())
    high_bins = all_hi["time_bin"].to_list()
    in_high = pl.col("time_bin").is_in(high_bins)
    rest_b = bin_counts(timed.filter(~skip))
    bt_rest = burst_test(rest_b["n"].to_numpy(), rest_b["k"].to_numpy(), n_sim=N_SIM, seed=SEED)
    info["s32"] = {
        "group_with_s32": nk(full.filter(is_g & pl.col("s32"))),
        "group_without_s32": nk(full.filter(is_g & ~pl.col("s32"))),
        "all_s32": nk(full.filter(pl.col("s32"))),
        "all_s32_with_s33": nk(full.filter(pl.col("s32") & pl.col("s33"))),
        "all_s32_without_s33": nk(full.filter(skip)),
        "all_s32_without_s33_also_without_s34": int(full.filter(skip & ~pl.col("s34")).height),
        "all_others": nk(full.filter(~skip)),
        "share_of_all_defects_in_s32_without_s33": float(full.filter(skip)["Response"].sum() / full["Response"].sum()),
        "s32_without_s33_share_in_high_bins": float(timed.filter(skip & in_high).height / timed.filter(skip).height),
        "s32_without_s33_rate_in_high_bins": nk(timed.filter(skip & in_high))["rate"],
        "s32_without_s33_rate_outside_high_bins": nk(timed.filter(skip & ~in_high))["rate"],
        "others_rate_in_high_bins": nk(timed.filter(~skip & in_high))["rate"],
        "others_rate_outside_high_bins": nk(timed.filter(~skip & ~in_high))["rate"],
        "burst_p_value_excluding_s32_without_s33": bt_rest["p_value"],
        "burst_statistic_excluding_s32_without_s33": bt_rest["statistic"],
        "burst_sim_statistic_max_excluding": bt_rest["sim_statistic_max"],
    }
    update_summary("group12", info)
    print(json.dumps(info, ensure_ascii=False, indent=1), flush=True)


def stage_verify() -> None:
    times, stations = load_times()
    f = load_feats()
    groups = date_groups()
    n = times.height
    res = {}
    # 1) 스테이션별 최소·최대 시각: pyarrow로 전체 행을 독립 계산
    pf = PQ / "train_date.parquet"
    bad_cells = bad_null = 0
    first = np.full(n, np.nan)
    last = np.full(n, np.nan)
    for s, cols in groups.items():
        lo = np.full(n, np.nan)
        hi = np.full(n, np.nan)
        for i in range(0, len(cols), 50):
            tbl = pq.read_table(pf, columns=cols[i:i + 50])
            for c in tbl.columns:
                a = c.to_numpy(zero_copy_only=False).astype(np.float64)
                lo = np.fmin(lo, a)
                hi = np.fmax(hi, a)
            del tbl
        for mine, ref in ((times[f"{s}_tmin"], lo), (times[f"{s}_tmax"], hi)):
            m = mine.to_numpy().astype(np.float64)
            bad_null += int((np.isnan(m) != np.isnan(ref)).sum())
            both = ~np.isnan(m) & ~np.isnan(ref)
            bad_cells += int((np.abs(m[both] - ref[both]) > 0.001).sum())
        first = np.fmin(first, lo)
        last = np.fmax(last, hi)
    res["station_times_full_check"] = {"rows": n, "stations": len(groups), "mismatch_null_cells": bad_null,
                                       "mismatch_value_cells": bad_cells}
    tf = f["t_first"].to_numpy()
    tl = f["t_last"].to_numpy()
    res["part_times_full_check"] = {
        "mismatch_null": int((np.isnan(tf) != np.isnan(first)).sum()),
        "mismatch_first": int((np.abs(tf - first)[~np.isnan(first)] > 0.001).sum()),
        "mismatch_last": int((np.abs(tl - last)[~np.isnan(last)] > 0.001).sum())}
    # 2) 구간 표 합계
    t = pl.read_csv(OUT / "time_bin_summary.csv")
    rf = pl.read_parquet(OUT / "row_features.parquet", columns=["Id", "Response", "n_stations_visited_numeric"])
    no_time = f.filter(pl.col("t_first").is_null())
    res["bin_sums"] = {"n_sum": int(t["n"].sum()), "n_expected": n - no_time.height,
                       "k_sum": int(t["k"].sum()), "k_expected": int(rf["Response"].sum()) - int(no_time["Response"].sum())}
    # 3) 전이 표 합계
    tr = pl.read_csv(OUT / "station_transitions.csv")
    n_with = f["n_stations_with_time"].to_numpy().astype(np.int64)
    res["transition_sum"] = {"n_sum": int(tr["n"].sum()), "n_expected": int(np.clip(n_with - 1, 0, None).sum())}
    # 4) 12인 그룹
    g = pl.read_csv(OUT / "group12_time.csv")
    g_no_time = no_time.join(rf.filter(pl.col("n_stations_visited_numeric") == GROUP_N_STATIONS), on="Id").height
    res["group12"] = {"n": int(g["n"].sum()), "k": int(g["k"].sum()), "n_expected_3_3": 7221, "k_expected_3_3": 440,
                      "group_parts_without_time": g_no_time}
    ok = (bad_null == 0 and bad_cells == 0 and all(v == 0 for v in res["part_times_full_check"].values())
          and res["bin_sums"]["n_sum"] == res["bin_sums"]["n_expected"]
          and res["bin_sums"]["k_sum"] == res["bin_sums"]["k_expected"]
          and res["transition_sum"]["n_sum"] == res["transition_sum"]["n_expected"]
          and res["group12"]["n"] + g_no_time == 7221 and res["group12"]["k"] == 440)
    res["all_passed"] = bool(ok)
    update_summary("verify", res)
    print(json.dumps(res, ensure_ascii=False, indent=1))
    if not ok:
        raise SystemExit("검증 실패: 원인을 찾기 전에는 결과를 쓰지 않는다")


STAGES = {"station_times": stage_station_times, "features": stage_features, "flow": stage_flow,
          "trend": stage_trend, "id_order": stage_id_order, "group12": stage_group12, "verify": stage_verify}

if __name__ == "__main__":
    arg = sys.argv[1] if len(sys.argv) > 1 else "all"
    for name in (STAGES if arg == "all" else [arg]):
        print(f"=== {name} ===", flush=True)
        STAGES[name]()
