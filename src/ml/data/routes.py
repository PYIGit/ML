"""경로(방문 스테이션 집합)와 라인 조합 계산 (3-3 문서).

경로는 스테이션 집합이며 방문 순서는 다루지 않는다. 집합은 비트마스크(UInt64, 64개 이내)로 센다.
"""
import numpy as np
import polars as pl

from ml.stats import MIN_GROUP_SIZE, rate_table


def route_keys(visited: pl.DataFrame) -> np.ndarray:
    """bool 열 표 -> 행마다 비트마스크(UInt64). i번째 열이 i번째 비트."""
    if visited.width > 64:
        raise ValueError("스테이션이 64개를 넘음")
    key = np.zeros(visited.height, dtype=np.uint64)
    for i, c in enumerate(visited.columns):
        key |= visited[c].to_numpy().astype(np.uint64) << np.uint64(i)
    return key


def decode_route(key: int, station_names: list[str]) -> list[str]:
    return [n for i, n in enumerate(station_names) if (int(key) >> i) & 1]


def route_table(keys: np.ndarray, response: np.ndarray, station_names: list[str],
                min_size: int = MIN_GROUP_SIZE) -> pl.DataFrame:
    """경로별 부품 수, 불량 수, 불량률, Wilson 구간. 부품 수 내림차순."""
    df = pl.DataFrame({"route_key": keys, "y": response.astype(np.int64)})
    g = (df.group_by("route_key").agg(pl.len().alias("n"), pl.col("y").sum().alias("k"))
         .sort(["n", "route_key"], descending=[True, False]))
    g = rate_table(g, ["route_key"], min_size=min_size)
    keys_list = g["route_key"].to_list()
    stations = [decode_route(k, station_names) for k in keys_list]
    return g.with_columns(
        pl.Series("n_stations", [len(s) for s in stations], dtype=pl.Int16),
        pl.Series("stations", [",".join(s) for s in stations], dtype=pl.Utf8),
    )


def top_share(table: pl.DataFrame, top_n: int) -> float:
    """상위 top_n개 경로가 전체 부품에서 차지하는 비율 (table은 n 내림차순)."""
    total = table["n"].sum()
    return float(table["n"].head(top_n).sum() / total) if total else float("nan")


def line_visited(visited: pl.DataFrame, line_of_station: dict[str, int], n_lines: int = 4) -> pl.DataFrame:
    """스테이션 방문 -> 라인 방문(그 라인 스테이션 중 하나라도 방문하면 True). 열 이름 L0..L{n-1}."""
    cols = {}
    for line in range(n_lines):
        members = [c for c in visited.columns if line_of_station[c] == line]
        cols[f"L{line}"] = (pl.any_horizontal([pl.col(c) for c in members]) if members
                            else pl.lit(False))
    return visited.select([e.alias(n) for n, e in cols.items()])


def line_combination_table(lines: pl.DataFrame, response: np.ndarray,
                           min_size: int = MIN_GROUP_SIZE) -> pl.DataFrame:
    """라인 방문 조합별 부품 수, 불량 수, 불량률. combo 열은 'L0+L1' 같은 문자열('none'은 없음)."""
    names = lines.columns
    df = lines.with_columns(pl.Series("y", response.astype(np.int64)))
    g = df.group_by(names).agg(pl.len().alias("n"), pl.col("y").sum().alias("k"))
    combo = [("+".join(n for n, v in zip(names, row) if v) or "none")
             for row in g.select(names).iter_rows()]
    g = g.with_columns(pl.Series("combo", combo)).sort("n", descending=True)
    return rate_table(g, ["combo"], min_size=min_size)


def line_pair_conditional(lines: pl.DataFrame) -> pl.DataFrame:
    """라인 쌍마다 P(j 방문 | i 방문). i == j는 제외."""
    rows = []
    for i in lines.columns:
        n_i = int(lines[i].sum())
        for j in lines.columns:
            if i == j:
                continue
            both = int((lines[i] & lines[j]).sum())
            rows.append({"given": i, "then": j, "n_given": n_i, "n_both": both,
                         "p": both / n_i if n_i else float("nan")})
    return pl.DataFrame(rows)
