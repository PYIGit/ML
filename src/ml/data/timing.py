"""date 컬럼으로 시각을 정리하는 계산 (3-4 문서).

가정: date 컬럼의 값은 그 스테이션에서 기록된 시각이다(단위는 모른다).
스테이션의 시각은 그 스테이션 D 컬럼들의 최솟값과 최댓값으로 요약한다.
"""
import numpy as np
import polars as pl

from ml.data.visits import plan_chunks


def _combine(a: pl.Series, b: pl.Series, how: str) -> pl.Series:
    f = pl.min_horizontal if how == "min" else pl.max_horizontal
    return pl.DataFrame({"a": a, "b": b}).select(f("a", "b")).to_series()


def compute_station_time_range(path, groups: dict[str, list[str]], max_cols: int = 100) -> pl.DataFrame:
    """스테이션별로 행마다 D 컬럼의 최솟값과 최댓값을 구한다 (Float32, 파일의 행 순서).

    반환 컬럼은 `{스테이션}_tmin`, `{스테이션}_tmax`. 값이 하나도 없으면 null.
    """
    lf = pl.scan_parquet(path)
    lo: dict[str, pl.Series] = {}
    hi: dict[str, pl.Series] = {}
    for chunk in plan_chunks(groups, max_cols):
        exprs, names = [], []
        for idx, (key, cols) in enumerate(chunk):
            exprs.append(pl.min_horizontal([pl.col(c) for c in cols]).cast(pl.Float32).alias(f"{key}|{idx}|min"))
            exprs.append(pl.max_horizontal([pl.col(c) for c in cols]).cast(pl.Float32).alias(f"{key}|{idx}|max"))
            names.append((key, idx))
        res = lf.select(exprs).collect(engine="streaming")
        for key, idx in names:
            a, b = res[f"{key}|{idx}|min"], res[f"{key}|{idx}|max"]
            lo[key] = a if key not in lo else _combine(lo[key], a, "min")
            hi[key] = b if key not in hi else _combine(hi[key], b, "max")
    cols = []
    for key in groups:
        cols.append(lo[key].alias(f"{key}_tmin"))
        cols.append(hi[key].alias(f"{key}_tmax"))
    return pl.DataFrame(cols)


def station_consistency(times: pl.DataFrame, stations: list[str]) -> pl.DataFrame:
    """스테이션별로 '한 부품 안에서 D 컬럼 값이 모두 같은가'(최소 == 최대)를 센다."""
    rows = []
    for s in stations:
        lo, hi = times[f"{s}_tmin"], times[f"{s}_tmax"]
        diff = (hi - lo).drop_nulls()
        differ = diff.filter(diff > 0)
        rows.append({"station_key": s, "n_with_time": diff.len(), "n_all_same": int((diff == 0).sum()),
                     "n_differ": differ.len(),
                     "max_diff": float(differ.max()) if differ.len() else 0.0,
                     "median_diff_when_differ": float(differ.median()) if differ.len() else 0.0})
    return pl.DataFrame(rows)


def part_time_features(times: pl.DataFrame, stations: list[str]) -> pl.DataFrame:
    """부품별 최초 시각, 최종 시각, 소요 시간, 시각이 있는 스테이션 수. 값은 소수 둘째 자리로 맞춘다."""
    lo = [pl.col(f"{s}_tmin") for s in stations]
    hi = [pl.col(f"{s}_tmax") for s in stations]
    return times.select(
        pl.min_horizontal(lo).cast(pl.Float64).round(2).alias("t_first"),
        pl.max_horizontal(hi).cast(pl.Float64).round(2).alias("t_last"),
        pl.sum_horizontal([c.is_not_null() for c in lo]).cast(pl.Int16).alias("n_stations_with_time"),
    ).with_columns((pl.col("t_last") - pl.col("t_first")).round(2).alias("duration"))


def assign_time_bin(t: pl.Series, width: float) -> pl.Series:
    """고정 폭 구간 번호: floor(t / width). null은 null."""
    if width <= 0:
        raise ValueError("width는 0보다 커야 함")
    return (t / width).floor().cast(pl.Int32).alias("time_bin")


def transitions_long(mat: np.ndarray, chunk: int = 200_000) -> pl.DataFrame:
    """부품마다 스테이션을 시각순으로 놓고, 이어진 두 스테이션(전이)을 한 행으로 만든다.

    mat: (부품 수, 스테이션 수) 시각 행렬, 값이 없으면 NaN. 열 순서는 스테이션 번호순이어야 한다.
    시각이 같은 스테이션은 열 순서(번호순)로 놓는다. 이것은 가정이다.
    반환: from_idx, to_idx(열 번호), dt(뒤 시각 - 앞 시각).
    """
    fr, to, dt = [], [], []
    for s in range(0, mat.shape[0], chunk):
        m = mat[s:s + chunk]
        order = np.argsort(m, axis=1, kind="stable")  # NaN은 맨 뒤
        ts = np.take_along_axis(m, order, axis=1)
        valid = ~np.isnan(ts[:, 1:])
        fr.append(order[:, :-1][valid].astype(np.int16))
        to.append(order[:, 1:][valid].astype(np.int16))
        dt.append((ts[:, 1:] - ts[:, :-1])[valid].astype(np.float32))
    return pl.DataFrame({"from_idx": np.concatenate(fr), "to_idx": np.concatenate(to), "dt": np.concatenate(dt)})


def transition_table(long: pl.DataFrame, stations: list[str]) -> pl.DataFrame:
    """전이별 부품 수, 같은 시각인 비율, 시각 차이의 중앙값. 부품 수 내림차순."""
    name = pl.Series("s", stations)
    t = (long.group_by("from_idx", "to_idx")
         .agg(pl.len().alias("n"), (pl.col("dt") == 0).mean().alias("same_time_share"),
              pl.col("dt").median().cast(pl.Float64).round(2).alias("dt_median"))
         .sort("n", descending=True))
    return t.with_columns(
        name.gather(t["from_idx"].cast(pl.Int64)).alias("from_station"),
        name.gather(t["to_idx"].cast(pl.Int64)).alias("to_station"),
        (pl.col("to_idx") < pl.col("from_idx")).alias("against_number_order"),
    ).select("from_station", "to_station", "n", "same_time_share", "dt_median", "against_number_order",
             "from_idx", "to_idx")


def burst_test(n_bins: np.ndarray, k_bins: np.ndarray, n_sim: int = 5000, seed: int = 0) -> dict:
    """불량이 시간 구간에 우연보다 많이 몰려 있는지 본다.

    기준선: 구간별 부품 수(n_bins)는 그대로 두고, 불량 전체(k_bins의 합)를 부품에 무작위로 놓는다.
    통계량: 구간별 (불량 수 - 기대값)^2 / 기대값의 합. p값은 무작위 배치가 관측값 이상인 비율.
    반환에는 구간별 무작위 배치의 95% 범위(sim_lo, sim_hi)가 들어 있다.
    """
    n_bins = np.asarray(n_bins, dtype=np.int64)
    k_bins = np.asarray(k_bins, dtype=np.int64)
    total_n, total_k = int(n_bins.sum()), int(k_bins.sum())
    if total_n == 0 or total_k == 0:
        raise ValueError("부품 수와 불량 수는 0보다 커야 함")
    expected = total_k * n_bins / total_n
    use = expected > 0

    def stat(k):
        return (((k - expected) ** 2)[..., use] / expected[use]).sum(axis=-1)

    rng = np.random.default_rng(seed)
    sims = rng.multivariate_hypergeometric(n_bins, total_k, size=n_sim, method="marginals")
    obs = float(stat(k_bins))
    sim_stat = stat(sims)
    return {
        "statistic": obs,
        "sim_statistic_median": float(np.median(sim_stat)),
        "sim_statistic_max": float(sim_stat.max()),
        "p_value": float((np.sum(sim_stat >= obs) + 1) / (n_sim + 1)),
        "n_sim": n_sim, "seed": seed,
        "expected": expected,
        "sim_lo": np.percentile(sims, 2.5, axis=0),
        "sim_hi": np.percentile(sims, 97.5, axis=0),
    }


def autocorrelation(x: np.ndarray, max_lag: int) -> np.ndarray:
    """평균을 뺀 계열의 자기상관(간격 0 ~ max_lag). 간격 0은 1."""
    x = np.asarray(x, dtype=np.float64)
    x = x - x.mean()
    denom = float((x * x).sum())
    if denom == 0:
        return np.zeros(max_lag + 1)
    return np.array([1.0] + [float((x[:-lag] * x[lag:]).sum() / denom) for lag in range(1, max_lag + 1)])


def dominant_period(acf: np.ndarray, min_value: float = 0.1) -> int | None:
    """반복 주기로 볼 간격: 자기상관이 min_value 아래로 한 번 내려간 뒤의 구간에서 가장 높은 간격.

    그 값이 min_value보다 낮으면 주기가 없다고 보고 None을 돌려준다.
    """
    below = np.where(acf[1:] < min_value)[0]
    if len(below) == 0:
        return None
    start = 1 + int(below[0])
    lag = start + int(np.argmax(acf[start:]))
    return lag if acf[lag] >= min_value else None
