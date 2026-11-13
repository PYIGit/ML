import unittest

import numpy as np
import polars as pl

from ml.data.features import (candidate_groups, classify_categorical, confirm_row_groups, group_identical_columns,
                              series_stats)


class SeriesStatsTest(unittest.TestCase):
    def test_numeric(self):
        s = series_stats(pl.Series("a", [0.5, None, -1.0, 0.5, 1.0, -1.0, 0.5]))
        self.assertEqual((s["n_nonnull"], s["n_unique"], s["top_value"], s["top_count"]), (6, 3, 0.5, 3))
        self.assertEqual((s["min"], s["max"], s["n_at_min"], s["n_at_max"]), (-1.0, 1.0, 2, 1))
        self.assertAlmostEqual(s["sum"], 0.5)

    def test_tie_takes_smaller_value(self):
        s = series_stats(pl.Series("a", [2.0, 1.0, 2.0, 1.0]))
        self.assertEqual((s["top_value"], s["top_count"]), (1.0, 2))

    def test_string(self):
        s = series_stats(pl.Series("c", ["T1", None, "T1", "T2"]))
        self.assertEqual((s["n_nonnull"], s["n_unique"], s["top_value"], s["top_count"]), (3, 2, "T1", 2))
        self.assertNotIn("min", s)

    def test_all_null(self):
        s = series_stats(pl.Series("a", [None, None], dtype=pl.Float64))
        self.assertEqual((s["n_nonnull"], s["n_unique"], s["top_value"]), (0, 0, None))
        t = series_stats(pl.Series("c", [None], dtype=pl.String))
        self.assertEqual(t["n_nonnull"], 0)

    def test_single_value(self):
        s = series_stats(pl.Series("a", [0.1, 0.1, None]))
        self.assertEqual((s["n_unique"], s["n_at_min"], s["n_at_max"]), (1, 2, 2))


class IdenticalColumnsTest(unittest.TestCase):
    def test_same_values_and_null_positions(self):
        df = pl.DataFrame({
            "a": [1.0, None, 3.0],
            "b": [1.0, None, 3.0],   # a와 같음
            "c": [1.0, 3.0, None],   # 값은 같지만 빈 칸 위치가 다름
            "d": [1.0, None, 4.0],   # 값이 다름
            "e": [1.0, None, 3.0],   # a와 같음
        })
        self.assertEqual(group_identical_columns(df), [["a", "b", "e"]])

    def test_strings(self):
        df = pl.DataFrame({"x": ["T1", None], "y": ["T1", None], "z": [None, "T1"]})
        self.assertEqual(group_identical_columns(df), [["x", "y"]])

    def test_candidates(self):
        g = candidate_groups({"a": (3, 1.0), "b": (3, 1.0), "c": (2, 1.0)})
        self.assertEqual(g, [["a", "b"]])


class RowGroupsTest(unittest.TestCase):
    def test_same_hash_but_different_values_are_not_grouped(self):
        nan = np.nan
        values = np.array([
            [1.0, nan],   # 0: 2와 같음
            [1.0, 2.0],   # 1: 해시는 0과 같지만 값이 다름
            [1.0, nan],   # 2
            [5.0, 5.0],   # 3: 4와 같음
            [5.0, 5.0],   # 4
            [9.0, nan],   # 5: 혼자
        ])
        hashes = np.array([7, 7, 7, 8, 8, 9], dtype=np.uint64)
        g = confirm_row_groups(hashes, values)
        self.assertEqual(g[0], g[2])
        self.assertEqual(g[3], g[4])
        self.assertNotEqual(g[0], g[3])
        self.assertEqual((g[1], g[5]), (-1, -1))

    def test_two_real_groups_under_one_hash(self):
        values = np.array([[1.0], [2.0], [1.0], [2.0]])
        g = confirm_row_groups(np.zeros(4, dtype=np.uint64), values)
        self.assertEqual(g[0], g[2])
        self.assertEqual(g[1], g[3])
        self.assertNotEqual(g[0], g[1])

    def test_empty(self):
        self.assertEqual(len(confirm_row_groups(np.array([], dtype=np.uint64), np.empty((0, 3)))), 0)


class ClassifyTest(unittest.TestCase):
    def test_order(self):
        self.assertEqual(classify_categorical(0, 0, None), "empty")
        self.assertEqual(classify_categorical(10, 1, "x"), "single_value")
        self.assertEqual(classify_categorical(10, 2, "x"), "duplicate")
        self.assertEqual(classify_categorical(10, 2, None), "keep")


if __name__ == "__main__":
    unittest.main()
