"""스테이션 방문 계산 (3-3 문서).

"방문"은 프로젝트의 가정이다: 한 부품이 한 스테이션의 컬럼 중 하나라도 값이 있으면 그 스테이션을 방문한 것으로 본다.
컬럼 수가 많은 파일을 한 번에 읽으면 메모리가 크므로, 컬럼을 묶음 단위로 나눠 스트리밍으로 센다.
"""
import polars as pl


def station_columns(meta: pl.DataFrame, data_kind: str) -> dict[str, list[str]]:
    """column_meta 표에서 파일 종류(numeric/date/categorical)의 스테이션별 피처 컬럼 목록.

    스테이션 순서는 파일에서 처음 나타나는 순서, 컬럼 순서는 파일 순서다.
    """
    sub = (meta.filter((pl.col("data_kind") == data_kind) & (pl.col("role") == "feature")
                       & (pl.col("split") == "train"))
           .sort("position"))
    out: dict[str, list[str]] = {}
    for key, col in zip(sub["station_key"].to_list(), sub["column"].to_list()):
        out.setdefault(key, []).append(col)
    return out


def plan_chunks(groups: dict[str, list[str]], max_cols: int) -> list[list[tuple[str, list[str]]]]:
    """스테이션 컬럼 목록을 읽기 묶음으로 나눈다. 한 묶음의 컬럼 수는 max_cols 이하.

    큰 스테이션은 여러 묶음에 걸쳐 나뉘고, 작은 스테이션은 한 묶음에 함께 담긴다.
    """
    if max_cols < 1:
        raise ValueError("max_cols는 1 이상이어야 함")
    chunks: list[list[tuple[str, list[str]]]] = []
    cur: list[tuple[str, list[str]]] = []
    room = max_cols
    for key, cols in groups.items():
        i = 0
        while i < len(cols):
            take = cols[i:i + room]
            cur.append((key, take))
            i += len(take)
            room -= len(take)
            if room == 0:
                chunks.append(cur)
                cur, room = [], max_cols
    if cur:
        chunks.append(cur)
    return chunks


def compute_station_counts(path, groups: dict[str, list[str]], max_cols: int = 100) -> pl.DataFrame:
    """스테이션별로 "행마다 값이 있는 컬럼 수"를 센다 (Int16, 파일의 행 순서).

    반환 표의 컬럼은 groups의 스테이션 키다. 방문 여부는 `> 0`이다.
    """
    lf = pl.scan_parquet(path)
    parts: dict[str, pl.Series] = {}
    for chunk in plan_chunks(groups, max_cols):
        exprs, names = [], []
        for idx, (key, cols) in enumerate(chunk):
            name = f"{key}|{idx}"
            exprs.append(pl.sum_horizontal([pl.col(c).is_not_null() for c in cols])
                         .cast(pl.Int16).alias(name))
            names.append((name, key))
        res = lf.select(exprs).collect(engine="streaming")
        for name, key in names:
            s = res[name]
            parts[key] = s if key not in parts else (parts[key] + s)
    return pl.DataFrame({key: parts[key].alias(key) for key in groups})


def visited_from_counts(counts: pl.DataFrame) -> pl.DataFrame:
    """스테이션별 값이 있는 컬럼 수 -> 방문 여부(bool)."""
    return counts.select([(pl.col(c) > 0).alias(c) for c in counts.columns])
