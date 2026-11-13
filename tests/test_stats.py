import math
import unittest

import polars as pl

from ml.stats import rate_table, wilson_interval


class Wilson(unittest.TestCase):
    def test_known_values(self):
        lo, hi = wilson_interval(0, 10)
        self.assertAlmostEqual(lo, 0.0, places=6)
        self.assertAlmostEqual(hi, 3.8416 / 13.8416, places=3)  # 0.2775
        lo, hi = wilson_interval(5, 10)
        self.assertAlmostEqual(lo, 0.2366, places=3)
        self.assertAlmostEqual(hi, 0.7634, places=3)

    def test_bounds(self):
        for k, n in [(0, 1), (1, 1), (6879, 1183747)]:
            lo, hi = wilson_interval(k, n)
            self.assertTrue(0.0 <= lo <= k / n <= hi <= 1.0)

    def test_empty_group(self):
        lo, hi = wilson_interval(0, 0)
        self.assertTrue(math.isnan(lo) and math.isnan(hi))

    def test_invalid_k(self):
        with self.assertRaises(ValueError):
            wilson_interval(5, 3)


class RateTable(unittest.TestCase):
    def test_small_group_flag_and_columns(self):
        df = pl.DataFrame({"g": ["a", "b"], "n": [999, 1000], "k": [5, 5]})
        out = rate_table(df, ["g"])
        self.assertEqual(out.columns, ["g", "n", "k", "rate", "ci_low", "ci_high", "small_group"])
        self.assertEqual(out["small_group"].to_list(), [True, False])
        self.assertAlmostEqual(out["rate"][1], 0.005)
        self.assertTrue(out["ci_low"][1] < 0.005 < out["ci_high"][1])


if __name__ == "__main__":
    unittest.main()
