"""소표본 집계용 통계 도구 (EDA 계획 3.6절).

그룹별 비율에는 그룹 크기와 Wilson 신뢰구간을 함께 기록하고,
최소 표본 수 미만 그룹은 별도로 표시한다.
"""
import math

import polars as pl

MIN_GROUP_SIZE = 1000


def wilson_interval(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """이항 비율의 Wilson 구간. n == 0이면 (nan, nan)."""
    if n <= 0:
        return (math.nan, math.nan)
    if not 0 <= k <= n:
        raise ValueError(f"k={k}는 0~n={n} 범위여야 함")
    p = k / n
    z2 = z * z
    denom = 1 + z2 / n
    center = (p + z2 / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z2 / (4 * n * n)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def rate_table(df: pl.DataFrame, group_cols: list[str], n_col: str = "n", k_col: str = "k",
               min_size: int = MIN_GROUP_SIZE) -> pl.DataFrame:
    """그룹별 n, k가 있는 표에 비율, Wilson 구간, 소표본 표시를 붙인다."""
    ci = [wilson_interval(k, n) for k, n in zip(df[k_col].to_list(), df[n_col].to_list())]
    return df.with_columns(
        (pl.col(k_col) / pl.col(n_col)).alias("rate"),
        pl.Series("ci_low", [c[0] for c in ci], dtype=pl.Float64),
        pl.Series("ci_high", [c[1] for c in ci], dtype=pl.Float64),
        (pl.col(n_col) < min_size).alias("small_group"),
    ).select(group_cols + [n_col, k_col, "rate", "ci_low", "ci_high", "small_group"])
