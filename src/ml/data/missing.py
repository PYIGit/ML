"""결측 구조 분석 (3-3 문서): 결측률 구간, 구조적·산발적 결측, 교차 파일 일관성."""
import polars as pl

# (이름, 하한 포함, 상한 미포함). 결측률은 0~1.
MISSING_BINS = [
    ("100% (전부 결측)", 1.0, None),
    ("99% 이상 100% 미만", 0.99, 1.0),
    ("90% 이상 99% 미만", 0.90, 0.99),
    ("50% 이상 90% 미만", 0.50, 0.90),
    ("50% 미만", 0.0, 0.50),
]


def missing_rate_bins(meta: pl.DataFrame) -> pl.DataFrame:
    """파일 종류(train)별 피처 컬럼의 결측률 구간별 컬럼 수. 구간은 서로 겹치지 않는다."""
    feat = meta.filter((pl.col("role") == "feature") & (pl.col("split") == "train"))
    rows = []
    for kind in ["numeric", "date", "categorical"]:
        rates = feat.filter(pl.col("data_kind") == kind)["null_rate"]
        for name, lo, hi in MISSING_BINS:
            if hi is None:
                n = int((rates >= lo).sum())
            else:
                n = int(((rates >= lo) & (rates < hi)).sum())
            rows.append({"data_kind": kind, "bin": name, "n_columns": n})
    return pl.DataFrame(rows)


def decompose_missing(meta: pl.DataFrame, not_visited: dict[tuple[str, str], int]) -> pl.DataFrame:
    """컬럼별 결측을 구조적(스테이션 미방문)과 산발적(방문했는데 빈 값)으로 나눈다.

    not_visited[(data_kind, station_key)] = 그 파일 기준으로 스테이션을 방문하지 않은 행 수.
    방문하지 않으면 스테이션의 모든 컬럼이 비므로 구조적 결측 = 미방문 행 수다.
    산발적 결측이 음수면 방문 계산이 잘못된 것이므로 예외를 낸다.
    """
    feat = meta.filter((pl.col("role") == "feature") & (pl.col("split") == "train"))
    structural = [not_visited[(k, s)] for k, s in zip(feat["data_kind"], feat["station_key"])]
    out = feat.select("data_kind", "column", "station_key", "n_rows", "null_count").with_columns(
        pl.Series("structural_missing", structural, dtype=pl.Int64))
    out = out.with_columns((pl.col("null_count") - pl.col("structural_missing")).alias("sporadic_missing"))
    if (out["sporadic_missing"] < 0).any():
        bad = out.filter(pl.col("sporadic_missing") < 0).head(5)
        raise ValueError(f"산발적 결측이 음수인 컬럼이 있음:\n{bad}")
    visited_rows = pl.col("n_rows") - pl.col("structural_missing")
    return out.with_columns(
        (pl.col("sporadic_missing") / visited_rows).alias("sporadic_rate_in_visited"))


def cross_table(a: pl.Series, b: pl.Series) -> dict[str, int]:
    """두 방문 여부(bool) 열의 교차표: 둘 다, a만, b만, 둘 다 아님."""
    both = int((a & b).sum())
    a_only = int((a & ~b).sum())
    b_only = int((~a & b).sum())
    return {"both": both, "a_only": a_only, "b_only": b_only,
            "neither": a.len() - both - a_only - b_only}
