import unittest

import polars as pl

from ml.data.columns import build_column_meta
from ml.data.missing import cross_table, decompose_missing, missing_rate_bins


def _meta(null_counts):
    cols = ["Id", "L0_S0_F0", "L0_S0_F1", "L0_S1_F2"]
    return build_column_meta("train_numeric", cols, null_counts, 10)


class MissingTest(unittest.TestCase):
    def test_bins_are_exclusive_and_sum_to_columns(self):
        meta = _meta([0, 10, 9, 0])
        t = missing_rate_bins(meta).filter(pl.col("data_kind") == "numeric")
        self.assertEqual(t["n_columns"].sum(), 3)
        d = dict(zip(t["bin"], t["n_columns"]))
        self.assertEqual(d["100% (전부 결측)"], 1)
        self.assertEqual(d["90% 이상 99% 미만"], 1)
        self.assertEqual(d["50% 미만"], 1)

    def test_decompose(self):
        meta = _meta([0, 6, 8, 3])
        out = decompose_missing(meta, {("numeric", "L0_S0"): 5, ("numeric", "L0_S1"): 3})
        r = out.filter(pl.col("column") == "L0_S0_F0").row(0, named=True)
        self.assertEqual((r["structural_missing"], r["sporadic_missing"]), (5, 1))
        self.assertAlmostEqual(r["sporadic_rate_in_visited"], 1 / 5)
        self.assertEqual((out["structural_missing"] + out["sporadic_missing"]).to_list(),
                         out["null_count"].to_list())

    def test_negative_sporadic_raises(self):
        with self.assertRaises(ValueError):
            decompose_missing(_meta([0, 2, 2, 2]), {("numeric", "L0_S0"): 5, ("numeric", "L0_S1"): 0})

    def test_cross_table(self):
        a = pl.Series([True, True, False, False])
        b = pl.Series([True, False, True, False])
        self.assertEqual(cross_table(a, b), {"both": 1, "a_only": 1, "b_only": 1, "neither": 1})


if __name__ == "__main__":
    unittest.main()
