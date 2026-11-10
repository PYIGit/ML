import tempfile
import unittest
from pathlib import Path

import polars as pl

from ml.data.columns import build_column_meta
from ml.data.visits import (compute_station_counts, plan_chunks, station_columns,
                            visited_from_counts)


def _write(tmp: Path) -> Path:
    df = pl.DataFrame({
        "Id": [1, 2, 3, 4],
        "L0_S0_F0": [1.0, None, None, 2.0],
        "L0_S0_F1": [None, None, 3.0, 4.0],
        "L0_S1_F2": [None, None, None, 5.0],
        "L1_S5_F9": [7.0, 8.0, None, None],
    })
    p = tmp / "t.parquet"
    df.write_parquet(p)
    return p


class StationColumnsTest(unittest.TestCase):
    def test_group_order_and_roles(self):
        cols = ["Id", "L0_S0_F0", "L0_S1_F2", "L0_S0_F1", "L1_S5_F9"]
        meta = build_column_meta("train_numeric", cols, [0] * 5, 4)
        g = station_columns(meta, "numeric")
        self.assertEqual(list(g), ["L0_S0", "L0_S1", "L1_S5"])
        self.assertEqual(g["L0_S0"], ["L0_S0_F0", "L0_S0_F1"])


class PlanChunksTest(unittest.TestCase):
    def test_chunks_respect_limit_and_cover_all(self):
        groups = {"A": [f"a{i}" for i in range(5)], "B": ["b0"], "C": [f"c{i}" for i in range(3)]}
        chunks = plan_chunks(groups, 4)
        self.assertTrue(all(sum(len(c) for _, c in ch) <= 4 for ch in chunks))
        flat = [(k, c) for ch in chunks for k, cols in ch for c in cols]
        self.assertEqual(flat, [(k, c) for k, cols in groups.items() for c in cols])

    def test_invalid(self):
        with self.assertRaises(ValueError):
            plan_chunks({"A": ["a"]}, 0)


class ComputeCountsTest(unittest.TestCase):
    def test_counts_and_visits_independent_of_chunk_size(self):
        with tempfile.TemporaryDirectory() as d:
            p = _write(Path(d))
            groups = {"L0_S0": ["L0_S0_F0", "L0_S0_F1"], "L0_S1": ["L0_S1_F2"], "L1_S5": ["L1_S5_F9"]}
            expected = {"L0_S0": [1, 0, 1, 2], "L0_S1": [0, 0, 0, 1], "L1_S5": [1, 1, 0, 0]}
            for max_cols in (1, 2, 100):
                c = compute_station_counts(p, groups, max_cols)
                self.assertEqual({k: c[k].to_list() for k in c.columns}, expected, max_cols)
            v = visited_from_counts(compute_station_counts(p, groups))
            self.assertEqual(v["L0_S0"].to_list(), [True, False, True, True])
            self.assertEqual(c["L0_S0"].dtype, pl.Int16)


if __name__ == "__main__":
    unittest.main()
