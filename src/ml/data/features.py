"""numeric·categorical 컬럼의 특성을 정리하는 계산 (3-5 문서).

- 컬럼 통계: 값의 종류 수, 가장 많이 나온 값, (숫자 컬럼은) 범위와 양 끝 값의 개수
- 같은 컬럼: 모든 행에서 값이 같고 빈 칸의 위치도 같은 컬럼
- 같은 행: 모든 컬럼에서 값이 같고 빈 칸의 위치도 같은 행
"""
import numpy as np
import polars as pl


def series_stats(s: pl.Series) -> dict:
    """컬럼 하나의 통계. 빈 칸(null)은 세지 않는다.

    공통: n_nonnull, n_unique, top_value(가장 많이 나온 값. 동률이면 작은 값), top_count
    숫자 컬럼만: min, max, mean, std, sum, n_at_min, n_at_max
    """
    v = s.drop_nulls()
    numeric = s.dtype.is_numeric()
    out = {"n_nonnull": v.len(), "n_unique": 0, "top_value": None, "top_count": 0}
    if numeric:
        out.update({"min": None, "max": None, "mean": None, "std": None, "sum": None, "n_at_min": 0, "n_at_max": 0})
    if v.len() == 0:
        return out
    name = v.name or "value"
    vc = v.alias(name).value_counts().sort(["count", name], descending=[True, False])
    out["n_unique"] = vc.height
    out["top_value"] = vc[0, name]
    out["top_count"] = int(vc[0, "count"])
    if numeric:
        lo, hi = v.min(), v.max()
        out.update({
            "min": lo, "max": hi, "mean": v.mean(), "std": v.std() if v.len() > 1 else 0.0, "sum": v.sum(),
            "n_at_min": int(vc.filter(pl.col(name) == lo)["count"][0]),
            "n_at_max": int(vc.filter(pl.col(name) == hi)["count"][0]),
        })
    return out


def group_identical_columns(df: pl.DataFrame) -> list[list[str]]:
    """df의 컬럼 중 값과 빈 칸 위치가 완전히 같은 것끼리 묶는다. 2개 이상인 묶음만 반환한다.

    값을 직접 비교한다(`Series.equals`, null끼리는 같다고 본다). 묶음 안과 묶음 사이의 순서는 df의 컬럼 순서다.
    """
    groups: list[list[str]] = []
    for c in df.columns:
        for g in groups:
            if df[g[0]].equals(df[c], null_equal=True, check_names=False):
                g.append(c)
                break
        else:
            groups.append([c])
    return [g for g in groups if len(g) > 1]


def candidate_groups(signatures: dict[str, tuple]) -> list[list[str]]:
    """요약값(signature)이 같은 컬럼끼리 묶는다. 2개 이상인 묶음만. 같은 컬럼의 '후보'일 뿐이다."""
    by: dict[tuple, list[str]] = {}
    for c, sig in signatures.items():
        by.setdefault(sig, []).append(c)
    return [g for g in by.values() if len(g) > 1]


def confirm_row_groups(hashes: np.ndarray, values: np.ndarray) -> np.ndarray:
    """해시가 같은 행들 가운데 값이 실제로 같은 행끼리 묶음 번호를 붙인다.

    hashes: 행별 해시(길이 n), values: 행별 값(n x 컬럼 수, 빈 칸은 NaN).
    반환: 길이 n의 묶음 번호. 같은 값의 행이 2개 이상일 때만 0부터 번호를 주고, 아니면 -1.
    해시가 같아도 값이 다르면 같은 묶음이 아니다.
    """
    n = len(hashes)
    out = np.full(n, -1, dtype=np.int64)
    if n == 0:
        return out
    order = np.argsort(hashes, kind="stable")
    h = hashes[order]
    starts = np.flatnonzero(np.r_[True, h[1:] != h[:-1]])
    ends = np.r_[starts[1:], n]
    nxt = 0
    for a, b in zip(starts, ends):
        if b - a < 2:
            continue
        idx = order[a:b]
        left = np.ones(len(idx), dtype=bool)  # 아직 묶이지 않은 행
        while left.sum() >= 2:
            cand = idx[left]
            ref = values[cand[0]]
            block = values[cand]
            same = ((block == ref) | (np.isnan(block) & np.isnan(ref))).all(axis=1)
            if same.sum() >= 2:
                out[cand[same]] = nxt
                nxt += 1
            pos = np.flatnonzero(left)
            left[pos[same]] = False
    return out


def classify_categorical(n_nonnull: int, n_unique: int, duplicate_of: str | None) -> str:
    """categorical 컬럼의 판정. 순서: 전부 빔 -> 값 한 가지 -> 다른 컬럼과 같음 -> 남길 후보."""
    if n_nonnull == 0:
        return "empty"
    if n_unique == 1:
        return "single_value"
    if duplicate_of is not None:
        return "duplicate"
    return "keep"
