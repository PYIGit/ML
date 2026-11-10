import unittest

import numpy as np
import polars as pl

from ml.data.routes import (decode_route, line_combination_table, line_pair_conditional,
                            line_visited, route_keys, route_table, top_share)


class RouteTest(unittest.TestCase):
    def setUp(self):
        self.v = pl.DataFrame({"A": [True, True, False, True, False],
                               "B": [False, False, True, True, False],
                               "C": [True, True, False, False, False]})
        self.y = np.array([1, 0, 0, 1, 0])

    def test_keys_and_decode(self):
        k = route_keys(self.v)
        self.assertEqual(k.tolist(), [5, 5, 2, 3, 0])
        self.assertEqual(decode_route(5, ["A", "B", "C"]), ["A", "C"])
        self.assertEqual(decode_route(0, ["A", "B", "C"]), [])

    def test_too_many_stations(self):
        wide = pl.DataFrame({f"s{i}": [True] for i in range(65)})
        with self.assertRaises(ValueError):
            route_keys(wide)

    def test_route_table_totals(self):
        t = route_table(route_keys(self.v), self.y, ["A", "B", "C"], min_size=2)
        self.assertEqual(t["n"].sum(), 5)
        self.assertEqual(t["k"].sum(), 2)
        top = t.row(0, named=True)
        self.assertEqual((top["route_key"], top["n"], top["k"], top["stations"]), (5, 2, 1, "A,C"))
        self.assertEqual(top["n_stations"], 2)
        self.assertFalse(top["small_group"])
        self.assertTrue(t.filter(pl.col("route_key") == 3)["small_group"].item())
        self.assertAlmostEqual(top_share(t, 1), 0.4)


class LineTest(unittest.TestCase):
    def test_line_visit_and_combos(self):
        v = pl.DataFrame({"s0": [True, False, True, False], "s1": [False, False, True, False],
                          "s2": [False, True, True, False]})
        lines = line_visited(v, {"s0": 0, "s1": 0, "s2": 1}, n_lines=3)
        self.assertEqual(lines["L0"].to_list(), [True, False, True, False])
        self.assertEqual(lines["L2"].to_list(), [False] * 4)
        t = line_combination_table(lines, np.array([1, 0, 0, 0]), min_size=2)
        self.assertEqual(t["n"].sum(), 4)
        d = {r["combo"]: r["n"] for r in t.iter_rows(named=True)}
        self.assertEqual(d, {"L0": 1, "L1": 1, "L0+L1": 1, "none": 1})
        p = line_pair_conditional(lines)
        row = p.filter((pl.col("given") == "L0") & (pl.col("then") == "L1")).row(0, named=True)
        self.assertEqual((row["n_given"], row["n_both"], row["p"]), (2, 1, 0.5))


if __name__ == "__main__":
    unittest.main()
