"""Bosch 컬럼명 파싱과 컬럼 메타 표 생성.

컬럼명 형식: L{라인}_S{스테이션}_{F|D}{번호}. 예: L0_S0_F0, L3_S51_D4261.
`Id`, `Response`는 이 형식이 아니므로 특수 컬럼으로 다룬다.
이름이 암시하는 라인·스테이션 구조는 익명화된 것이며 실제 공정을 설명한다고 주장하지 않는다.
"""
import re
from dataclasses import dataclass

import polars as pl

COLUMN_RE = re.compile(r"^L(\d+)_S(\d+)_([FD])(\d+)$")
SPECIAL_COLUMNS = {"Id": "id", "Response": "target"}


@dataclass(frozen=True)
class ColumnInfo:
    line: int
    station: int
    kind: str  # "F" 또는 "D"
    number: int

    @property
    def station_key(self) -> str:
        return f"L{self.line}_S{self.station}"


def parse_column_name(name: str) -> ColumnInfo:
    """형식에 맞지 않으면 ValueError."""
    m = COLUMN_RE.match(name)
    if m is None:
        raise ValueError(f"컬럼명 형식이 아님: {name!r}")
    line, station, kind, number = m.groups()
    return ColumnInfo(int(line), int(station), kind, int(number))


def build_column_meta(file_name: str, columns: list[str], null_counts: list[int | None],
                      n_rows: int) -> pl.DataFrame:
    """한 파일의 컬럼 목록으로 메타 표를 만든다.

    file_name: 예 "train_numeric". 마지막 `_` 뒤가 데이터 종류, 앞이 split.
    null_counts: 컬럼 순서대로의 null 수. 알 수 없으면 None.
    """
    if len(columns) != len(null_counts):
        raise ValueError("columns와 null_counts의 길이가 다름")
    split, _, data_kind = file_name.partition("_")
    rows = []
    for pos, (col, nc) in enumerate(zip(columns, null_counts)):
        if col in SPECIAL_COLUMNS:
            info = None
            role = SPECIAL_COLUMNS[col]
        else:
            info = parse_column_name(col)
            role = "feature"
        rows.append({
            "file": file_name, "split": split, "data_kind": data_kind,
            "position": pos, "column": col, "role": role,
            "line": info.line if info else None,
            "station": info.station if info else None,
            "station_key": info.station_key if info else None,
            "kind": info.kind if info else None,
            "number": info.number if info else None,
            "n_rows": n_rows, "null_count": nc,
            "null_rate": (nc / n_rows) if (nc is not None and n_rows) else None,
        })
    return pl.DataFrame(rows, schema={
        "file": pl.Utf8, "split": pl.Utf8, "data_kind": pl.Utf8, "position": pl.Int32,
        "column": pl.Utf8, "role": pl.Utf8, "line": pl.Int8, "station": pl.Int16,
        "station_key": pl.Utf8, "kind": pl.Utf8, "number": pl.Int32,
        "n_rows": pl.Int64, "null_count": pl.Int64, "null_rate": pl.Float64,
    })
