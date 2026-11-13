import unittest

import numpy as np

from ml.data.pairs import has_adjacent_id, has_mate_within, has_same_times


class AdjacentIdTest(unittest.TestCase):
    def test_adjacent_only_within_group(self):
        group = np.array([0, 0, 0, 1, 1, -1, -1])
        ids = np.array([10, 11, 20, 12, 30, 40, 41])
        # 10-11은 같은 묶음의 이웃. 12는 11과 이웃이지만 묶음이 다르다. 40-41은 묶음이 없다.
        self.assertEqual(has_adjacent_id(group, ids).tolist(), [True, True, False, False, False, False, False])

    def test_unsorted_input(self):
        group = np.array([2, 5, 2, 5])
        ids = np.array([101, 7, 100, 9])
        self.assertEqual(has_adjacent_id(group, ids).tolist(), [True, False, True, False])

    def test_empty(self):
        self.assertEqual(len(has_adjacent_id(np.array([], dtype=np.int64), np.array([], dtype=np.int64))), 0)


class SameTimesTest(unittest.TestCase):
    def test_same_times_within_group(self):
        nan = np.nan
        group = np.array([0, 0, 0, 1, 1, 2, 2, -1])
        times = np.array([
            [1.0, nan],   # 묶음 0: 아래 행과 같다
            [1.0, nan],
            [1.5, nan],   # 묶음 0: 시각이 다르다
            [3.0, 4.0],   # 묶음 1: 서로 다르다
            [3.0, 4.5],
            [nan, nan],   # 묶음 2: 시각이 하나도 없다
            [nan, nan],
            [1.0, nan],   # 묶음 없음
        ])
        self.assertEqual(has_same_times(group, times).tolist(), [True, True, False, False, False, False, False, False])

    def test_same_times_in_different_groups_do_not_match(self):
        group = np.array([0, 1])
        times = np.array([[1.0], [1.0]])
        self.assertEqual(has_same_times(group, times).tolist(), [False, False])


class MateWithinTest(unittest.TestCase):
    def test_train_only_mates(self):
        group = np.array([0, 0, 1, 1, 1, -1])
        is_train = np.array([True, False, True, True, False, True])
        self.assertEqual(has_mate_within(group, is_train).tolist(), [False, False, True, True, False, False])


if __name__ == "__main__":
    unittest.main()
