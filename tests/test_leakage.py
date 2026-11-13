import unittest

import numpy as np

from ml.data.leakage import group_kinds, last_station_index, ratio_outside, share_recorded_before


class LastStationTest(unittest.TestCase):
    def test_last_non_nan_column(self):
        nan = np.nan
        mat = np.array([
            [1.0, 2.0, 3.0],   # 끝까지
            [1.0, nan, nan],   # 첫 스테이션에서 끝남
            [nan, 5.0, nan],   # 가운데만
            [nan, nan, nan],   # 시각 없음
            [1.0, nan, 4.0],   # 가운데를 건너뛰고 끝까지
        ])
        self.assertEqual(last_station_index(mat).tolist(), [2, 0, 1, -1, 2])


class GroupKindsTest(unittest.TestCase):
    def test_train_test_both(self):
        group = np.array([0, 0, 1, 1, 2, 2, 2])
        is_train = np.array([True, True, False, False, True, False, False])
        self.assertEqual(group_kinds(group, is_train).tolist(),
                         ["train", "train", "test", "test", "both", "both", "both"])

    def test_empty(self):
        self.assertEqual(len(group_kinds(np.array([], dtype=np.int64), np.array([], dtype=bool))), 0)


class ShareTest(unittest.TestCase):
    def test_share_and_zero_total(self):
        s = share_recorded_before(np.array([5, 0, 0]), np.array([10, 4, 0]))
        self.assertEqual(s[:2].tolist(), [0.5, 0.0])
        self.assertTrue(np.isnan(s[2]))


class RatioTest(unittest.TestCase):
    def test_counts_bins_outside_range(self):
        r = ratio_outside(np.array([1000, 2000, 10, 1000]), np.array([1000, 2600, 100, 850]), 0.9, 1.1, 1000)
        self.assertEqual((r["bins_compared"], r["bins_outside"]), (3, 2))
        self.assertAlmostEqual(r["ratio_max"], 1.3)
        self.assertAlmostEqual(r["ratio_min"], 0.85)


if __name__ == "__main__":
    unittest.main()
