import tempfile
import unittest
from pathlib import Path

import numpy as np
import polars as pl

from ml.data.timing import (assign_time_bin, autocorrelation, burst_test, compute_station_time_range,
                            dominant_period, part_time_features, station_consistency, transition_table,
                            transitions_long)

GROUPS = {"L0_S0": ["L0_S0_D1", "L0_S0_D3"], "L0_S1": ["L0_S1_D5"], "L1_S5": ["L1_S5_D9", "L1_S5_D11", "L1_S5_D13"]}


def _times(max_cols: int) -> pl.DataFrame:
    df = pl.DataFrame({
        "Id": [1, 2, 3, 4],
        "L0_S0_D1": [1.0, None, None, 2.0],
        "L0_S0_D3": [1.0, None, 3.0, 2.5],
        "L0_S1_D5": [None, None, None, 5.0],
        "L1_S5_D9": [7.0, 8.0, None, None],
        "L1_S5_D11": [7.5, None, None, None],
        "L1_S5_D13": [6.0, 8.0, None, None],
    })
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "t.parquet"
        df.write_parquet(p)
        return compute_station_time_range(p, GROUPS, max_cols=max_cols)


class StationTimeRangeTest(unittest.TestCase):
    def test_min_max_and_nulls(self):
        t = _times(100)
        self.assertEqual(t.columns, ["L0_S0_tmin", "L0_S0_tmax", "L0_S1_tmin", "L0_S1_tmax", "L1_S5_tmin", "L1_S5_tmax"])
        self.assertEqual(t["L0_S0_tmin"].to_list(), [1.0, None, 3.0, 2.0])
        self.assertEqual(t["L0_S0_tmax"].to_list(), [1.0, None, 3.0, 2.5])
        self.assertEqual(t["L0_S1_tmin"].to_list(), [None, None, None, 5.0])
        self.assertEqual(t["L1_S5_tmin"].to_list(), [6.0, 8.0, None, None])
        self.assertEqual(t["L1_S5_tmax"].to_list(), [7.5, 8.0, None, None])

    def test_chunking_does_not_change_result(self):
        self.assertTrue(_times(1).equals(_times(100)))
        self.assertTrue(_times(2).equals(_times(100)))

    def test_consistency_counts(self):
        c = station_consistency(_times(100), list(GROUPS))
        row = {r["station_key"]: r for r in c.to_dicts()}
        self.assertEqual((row["L0_S0"]["n_with_time"], row["L0_S0"]["n_all_same"], row["L0_S0"]["n_differ"]), (3, 2, 1))
        self.assertAlmostEqual(row["L0_S0"]["max_diff"], 0.5)
        self.assertEqual((row["L1_S5"]["n_with_time"], row["L1_S5"]["n_differ"]), (2, 1))
        self.assertEqual(row["L0_S1"]["n_differ"], 0)


class PartTimeTest(unittest.TestCase):
    def test_first_last_duration(self):
        f = part_time_features(_times(100), list(GROUPS))
        self.assertEqual(f["t_first"].to_list(), [1.0, 8.0, 3.0, 2.0])
        self.assertEqual(f["t_last"].to_list(), [7.5, 8.0, 3.0, 5.0])
        self.assertEqual(f["duration"].to_list(), [6.5, 0.0, 0.0, 3.0])
        self.assertEqual(f["n_stations_with_time"].to_list(), [2, 1, 1, 2])

    def test_part_without_time_is_null(self):
        t = pl.DataFrame({"A_tmin": [None, 1.0], "A_tmax": [None, 1.0]}, schema={"A_tmin": pl.Float32, "A_tmax": pl.Float32})
        f = part_time_features(t, ["A"])
        self.assertEqual(f["t_first"].to_list(), [None, 1.0])
        self.assertEqual(f["n_stations_with_time"].to_list(), [0, 1])

    def test_time_bin(self):
        b = assign_time_bin(pl.Series([0.0, 9.99, 10.0, 25.3, None]), 10.0)
        self.assertEqual(b.to_list(), [0, 0, 1, 2, None])
        with self.assertRaises(ValueError):
            assign_time_bin(pl.Series([1.0]), 0)


class TransitionTest(unittest.TestCase):
    def test_order_by_time_then_station_number(self):
        nan = np.nan
        mat = np.array([
            [1.0, 2.0, 3.0],   # A-B-C
            [5.0, nan, 4.0],   # C-A (번호 역순)
            [2.0, 2.0, nan],   # 같은 시각: 번호순으로 A-B
            [nan, 7.0, nan],   # 스테이션 하나: 전이 없음
            [nan, nan, nan],   # 시각 없음
        ], dtype=np.float32)
        long = transitions_long(mat, chunk=2)
        self.assertEqual(long.height, 4)
        t = transition_table(long, ["A", "B", "C"])
        got = {(r["from_station"], r["to_station"]): r for r in t.to_dicts()}
        self.assertEqual(got[("A", "B")]["n"], 2)
        self.assertAlmostEqual(got[("A", "B")]["same_time_share"], 0.5)
        self.assertEqual(got[("B", "C")]["n"], 1)
        self.assertEqual(got[("C", "A")]["n"], 1)
        self.assertTrue(got[("C", "A")]["against_number_order"])
        self.assertFalse(got[("A", "B")]["against_number_order"])
        self.assertAlmostEqual(got[("C", "A")]["dt_median"], 1.0)


class BurstTest(unittest.TestCase):
    def test_random_labels_give_large_p(self):
        rng = np.random.default_rng(1)
        n = np.full(40, 5000)
        k = rng.multivariate_hypergeometric(n, 1200)
        r = burst_test(n, k, n_sim=500, seed=2)
        self.assertGreater(r["p_value"], 0.01)
        self.assertEqual(len(r["sim_lo"]), 40)

    def test_concentrated_defects_give_small_p(self):
        n = np.full(40, 5000)
        k = np.full(40, 20)
        k[7] = 420  # 한 구간에 불량을 몰아 넣음
        r = burst_test(n, k, n_sim=500, seed=2)
        self.assertLess(r["p_value"], 0.01)
        self.assertGreater(k[7], r["sim_hi"][7])

    def test_expected_follows_bin_size(self):
        r = burst_test(np.array([1000, 3000]), np.array([10, 30]), n_sim=200, seed=0)
        self.assertEqual(list(np.round(r["expected"], 6)), [10.0, 30.0])
        self.assertAlmostEqual(r["statistic"], 0.0)


class PeriodTest(unittest.TestCase):
    def test_finds_period_of_repeating_series(self):
        x = np.tile(np.array([10, 10, 10, 10, 10, 0, 0]), 60)
        acf = autocorrelation(x, 30)
        self.assertEqual(dominant_period(acf), 7)

    def test_no_period_in_constant_series(self):
        acf = autocorrelation(np.ones(100), 20)
        self.assertIsNone(dominant_period(acf))


if __name__ == "__main__":
    unittest.main()
