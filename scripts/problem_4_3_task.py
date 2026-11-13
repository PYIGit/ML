"""4-3 예측 과제: 입력 컬럼 수, L3 전의 값이 없는 부품, L3 전의 값만으로 찾은 짝. 설계: DOCS/04_PROBLEM/4-3_PREDICTION_TASK.md

입력: DATA/raw/bosch_parquet/{train_numeric.parquet, test_numeric.parquet},
      DATA/interim/eda/{column_meta.parquet, numeric_duplicate_columns.csv, categorical_column_stats.parquet,
                        station_time_train.parquet, station_time_test.parquet, duplicate_rows_combined.parquet}
출력: DATA/interim/problem/ (Git 무시)
  pre_l3_pairs.parquet, q4_3_summary.json

단계(순서대로): columns -> pairs -> verify
사용: .venv/bin/python scripts/problem_4_3_task.py <stage|all>
"""
import json
import math
import resource
import sys
from pathlib import Path

import numpy as np
import polars as pl
import pyarrow.parquet as pq

from ml.data.features import confirm_row_groups
from ml.data.pairs import has_adjacent_id, has_mate_within, has_same_times

PQ = Path("DATA/raw/bosch_parquet")
EDA = Path("DATA/interim/eda")
OUT = Path("DATA/interim/problem")
PRE_L3_LINES = [0, 1, 2]
CHUNK_BYTES = 1_000_000_000  # 후보 행의 값을 한 번에 올리는 한도
SEED = 20261113


def peak_mib() -> int:
    return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)


def update_summary(key: str, value: dict) -> None:
    p = OUT / "q4_3_summary.json"
    cur = json.loads(p.read_text()) if p.exists() else {}
    value["peak_rss_mib"] = peak_mib()
    cur[key] = value
    p.write_text(json.dumps(cur, indent=2, ensure_ascii=False))


def pre_l3_numeric_columns() -> list[str]:
    meta = pl.read_parquet(EDA / "column_meta.parquet")
    return (meta.filter((pl.col("file") == "train_numeric") & (pl.col("role") == "feature")
                        & pl.col("line").is_in(PRE_L3_LINES)).sort("position")["column"].to_list())


def stage_columns() -> None:
    meta = pl.read_parquet(EDA / "column_meta.parquet")
    num = meta.filter((pl.col("file") == "train_numeric") & (pl.col("role") == "feature"))
    dup = pl.read_csv(EDA / "numeric_duplicate_columns.csv").filter(~pl.col("is_representative"))["column"].to_list()
    cat = (pl.read_parquet(EDA / "categorical_column_stats.parquet").filter(pl.col("verdict") == "keep")
           .join(meta.filter(pl.col("file") == "train_categorical").select("column", "line"), on="column"))
    res = {}
    for name, lines in (("pre_l3", PRE_L3_LINES), ("l3", [3])):
        n = num.filter(pl.col("line").is_in(lines))
        res[name] = {
            "numeric_all": n.height,
            "numeric_kept": n.filter(~pl.col("column").is_in(dup)).height,
            "categorical_kept": cat.filter(pl.col("line").is_in(lines)).height,
            "stations": meta.filter((pl.col("file") == "train_date") & (pl.col("role") == "feature")
                                    & pl.col("line").is_in(lines))["station_key"].n_unique(),
        }
    res["numeric_kept_by_line"] = {str(r["line"]): r["len"] for r in
                                   num.filter(~pl.col("column").is_in(dup)).group_by("line").len().sort("line").to_dicts()}
    update_summary("columns", res)
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


def _confirm_in_chunks(h: np.ndarray, cand: np.ndarray, cols: list[str], n_tr: int) -> np.ndarray:
    """후보 행(cand, 오름차순)을 해시 값으로 나눠 묶음 번호를 붙인다. 같은 해시는 같은 조각에 들어간다."""
    k = max(1, math.ceil(len(cand) * len(cols) * 4 / CHUNK_BYTES))
    part = (h[cand] % np.uint64(k)).astype(np.int64)
    gid = np.full(len(cand), -1, dtype=np.int64)
    nxt = 0
    for j in range(k):
        sel = np.flatnonzero(part == j)
        rows = cand[sel]
        vals = np.empty((len(rows), len(cols)), dtype=np.float32)
        r_tr = rows[rows < n_tr]
        _gather(PQ / "train_numeric.parquet", cols, r_tr, vals[:len(r_tr)])
        _gather(PQ / "test_numeric.parquet", cols, rows[rows >= n_tr] - n_tr, vals[len(r_tr):])
        g = confirm_row_groups(h[rows], vals)
        del vals
        ok = g >= 0
        gid[sel[ok]] = g[ok] + nxt
        nxt += int(g.max()) + 1 if ok.any() else 0
        print(f"  chunk {j + 1}/{k}: rows {len(rows):,}, groups so far {nxt:,}, peak {peak_mib()} MiB", flush=True)
    return gid


def _times(rows_tr: np.ndarray, rows_te: np.ndarray) -> np.ndarray:
    """후보 행의 L0 ~ L2 스테이션 시각(tmin, tmax). train 행이 먼저, test 행이 다음."""
    names = pq.ParquetFile(EDA / "station_time_train.parquet").schema_arrow.names
    cols = [c for c in names if c != "Id" and int(c[1]) in PRE_L3_LINES]
    tr = pl.read_parquet(EDA / "station_time_train.parquet", columns=cols)[rows_tr.tolist()].to_numpy()
    te = pl.read_parquet(EDA / "station_time_test.parquet", columns=cols)[rows_te.tolist()].to_numpy()
    return np.concatenate([tr, te]).astype(np.float32)


def _rate(n: int, k: int, k_all: int) -> dict:
    return {"n": int(n), "k": int(k), "rate": k / n if n else None, "share_of_all_defects": k / k_all}


def stage_pairs() -> None:
    cols = pre_l3_numeric_columns()
    h_tr, has_tr = _hash_file(PQ / "train_numeric.parquet", cols)
    h_te, has_te = _hash_file(PQ / "test_numeric.parquet", cols)
    n_tr, n_te = len(h_tr), len(h_te)
    h = np.concatenate([h_tr, h_te])
    has = np.concatenate([has_tr, has_te])
    del h_tr, h_te
    _, inv, cnt = np.unique(h, return_inverse=True, return_counts=True)
    cand = np.flatnonzero((cnt[inv] > 1) & has)
    del inv, cnt
    print(f"  pre-L3 numeric columns: {len(cols)}, candidate rows: {len(cand):,}", flush=True)
    gid = _confirm_in_chunks(h, cand, cols, n_tr)

    tr = pl.read_parquet(PQ / "train_numeric.parquet", columns=["Id", "Response"])
    te = pl.read_parquet(PQ / "test_numeric.parquet", columns=["Id"])
    # station_time_*.parquet의 행 순서가 numeric 파일과 같은지 확인한다 (행 번호로 시각을 읽기 때문)
    assert (pl.read_parquet(EDA / "station_time_train.parquet", columns=["Id"])["Id"] == tr["Id"]).all()
    assert (pl.read_parquet(EDA / "station_time_test.parquet", columns=["Id"])["Id"] == te["Id"]).all()
    all_ids = np.concatenate([tr["Id"].to_numpy(), te["Id"].to_numpy()])
    resp = tr["Response"].to_numpy().astype(np.int64)

    keep = gid >= 0
    rows, g = cand[keep], gid[keep]
    is_train = rows < n_tr
    times = _times(rows[is_train], rows[~is_train] - n_tr)
    flags = {"a": np.ones(len(rows), dtype=bool),
             "b": has_adjacent_id(g, all_ids[rows]),
             "c": has_same_times(g, times)}
    flags["bc"] = _same_time_adjacent(g, all_ids[rows], times)
    # train 부품끼리만 찾았을 때 (test를 읽지 않는 경우)
    g_tr = np.where(is_train, g, -1)
    flags_tr = {"a": has_mate_within(g, is_train),
                "b": has_adjacent_id(g_tr, all_ids[rows]),
                "c": has_same_times(g_tr, times)}

    full_ids = pl.read_parquet(EDA / "duplicate_rows_combined.parquet").filter(pl.col("split") == "train")["Id"]
    out = tr.with_columns(pl.Series("has_pre_l3", has_tr), pl.col("Id").is_in(full_ids.to_list()).alias("pair_full"))
    for name, f in flags.items():
        col = np.zeros(n_tr, dtype=bool)
        col[rows[is_train & f]] = True
        out = out.with_columns(pl.Series(f"pair_{name}", col))
    for name, f in flags_tr.items():
        col = np.zeros(n_tr, dtype=bool)
        col[rows[is_train & f]] = True
        out = out.with_columns(pl.Series(f"pair_{name}_train_only", col))
    OUT.mkdir(parents=True, exist_ok=True)
    out.write_parquet(OUT / "pre_l3_pairs.parquet")

    k_all = int(resp.sum())
    sizes = np.bincount(g)
    res = {
        "n_pre_l3_numeric_columns": len(cols), "candidate_rows": int(len(cand)), "n_groups": int(len(sizes)),
        "group_size_max": int(sizes.max()), "groups_of_size_2": int((sizes == 2).sum()),
        "parts_in_groups_ge_10": int(sizes[sizes >= 10].sum()),
        "train": _rate(n_tr, k_all, k_all),
        "no_pre_l3": _rate(int((~has_tr).sum()), int(resp[~has_tr].sum()), k_all),
        "with_pre_l3": _rate(int(has_tr.sum()), int(resp[has_tr].sum()), k_all),
        "pair_full": _summ(out, "pair_full", k_all),
        "pair_full_with_pre_l3": _rate(out.filter(pl.col("pair_full") & pl.col("has_pre_l3")).height,
                                       int(out.filter(pl.col("pair_full") & pl.col("has_pre_l3"))["Response"].sum()), k_all),
    }
    for name in list(flags) + [f"{n}_train_only" for n in flags_tr]:
        c = f"pair_{name}"
        res[c] = _summ(out, c, k_all)
        res[c]["overlap_with_full"] = out.filter(pl.col(c) & pl.col("pair_full")).height
        x = out.filter(pl.col(c) & ~pl.col("pair_full"))
        res[c]["not_full"] = _rate(x.height, int(x["Response"].sum()), k_all)
        y = out.filter(~pl.col(c) & pl.col("has_pre_l3"))
        res[c]["no_pair_with_pre_l3"] = _rate(y.height, int(y["Response"].sum()), k_all)
    update_summary("pairs", res)
    print(json.dumps(res, ensure_ascii=False, indent=1))


def _same_time_adjacent(g: np.ndarray, ids: np.ndarray, times: np.ndarray) -> np.ndarray:
    """행마다: 같은 묶음에서 Id가 바로 이웃이고 시각도 전부 같은 행이 있는가."""
    out = np.zeros(len(g), dtype=bool)
    order = np.lexsort((ids, g))
    a, b = order[:-1], order[1:]
    ta, tb = times[a], times[b]
    same_t = ((ta == tb) | (np.isnan(ta) & np.isnan(tb))).all(axis=1) & ~np.isnan(ta).all(axis=1)
    adj = (g[a] == g[b]) & (ids[b] - ids[a] == 1) & same_t
    out[a[adj]] = True
    out[b[adj]] = True
    return out


def _summ(out: pl.DataFrame, col: str, k_all: int) -> dict:
    x = out.filter(pl.col(col))
    return _rate(x.height, int(x["Response"].sum()), k_all)


def stage_verify() -> None:
    cols = pre_l3_numeric_columns()
    out = pl.read_parquet(OUT / "pre_l3_pairs.parquet")
    res = {}
    # 1) 합계
    res["n_train"] = out.height
    res["n_defects"] = int(out["Response"].sum())
    assert out.height == 1_183_747 and res["n_defects"] == 6_879
    # 2) 968개 기준의 짝 중 L3 전의 값이 있는 부품은 (가)에도 짝이어야 한다
    miss = out.filter(pl.col("pair_full") & pl.col("has_pre_l3") & ~pl.col("pair_a")).height
    res["full_pairs_missing_in_a"] = miss
    assert miss == 0
    # 3) 조건을 더한 정의는 (가)의 부분이어야 한다
    for c in ("pair_b", "pair_c", "pair_bc"):
        assert out.filter(pl.col(c) & ~pl.col("pair_a")).height == 0
    assert out.filter(pl.col("pair_bc") & ~(pl.col("pair_b") & pl.col("pair_c"))).height == 0
    # 4) (나) 부품 300개: 이웃 Id의 L0 ~ L2 값을 원본에서 다시 읽어 같은지 확인한다 (polars 스캔, 다른 읽기 경로)
    rng = np.random.default_rng(SEED)
    b_ids = out.filter(pl.col("pair_b"))["Id"].to_numpy()
    pick = rng.choice(b_ids, size=300, replace=False)
    near = np.unique(np.concatenate([pick - 1, pick, pick + 1]))
    lf = pl.concat([pl.scan_parquet(PQ / "train_numeric.parquet").select(["Id"] + cols),
                    pl.scan_parquet(PQ / "test_numeric.parquet").select(["Id"] + cols)])
    df = lf.filter(pl.col("Id").is_in(near.tolist())).collect()
    vals = {i: v for i, v in zip(df["Id"].to_list(), df.drop("Id").to_numpy())}
    bad = 0
    for i in pick.tolist():
        same = [n for n in (i - 1, i + 1) if n in vals and np.array_equal(vals[i], vals[n], equal_nan=True)]
        bad += not same
    res["sample_b_checked"] = 300
    res["sample_b_mismatch"] = bad
    assert bad == 0
    update_summary("verify", res)
    print(json.dumps(res, ensure_ascii=False, indent=1))


STAGES = {"columns": stage_columns, "pairs": stage_pairs, "verify": stage_verify}

if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    OUT.mkdir(parents=True, exist_ok=True)
    for name in (STAGES if which == "all" else [which]):
        print(f"== {name}", flush=True)
        STAGES[name]()
