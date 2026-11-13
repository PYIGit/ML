import unittest

from ml.data.columns import ColumnInfo, build_column_meta, parse_column_name


class ParseColumnName(unittest.TestCase):
    def test_numeric_feature(self):
        self.assertEqual(parse_column_name("L0_S0_F0"), ColumnInfo(0, 0, "F", 0))

    def test_date_feature_multi_digit(self):
        info = parse_column_name("L3_S51_D4261")
        self.assertEqual((info.line, info.station, info.kind, info.number), (3, 51, "D", 4261))
        self.assertEqual(info.station_key, "L3_S51")

    def test_invalid_names_raise(self):
        for bad in ["Id", "Response", "L0_S0_X1", "l0_s0_f1", "L0_S0_F", "L0_S_F1", "L0_S0_F1 ", ""]:
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    parse_column_name(bad)


class BuildColumnMeta(unittest.TestCase):
    def test_special_and_feature_columns(self):
        df = build_column_meta("train_numeric", ["Id", "L0_S1_F25", "Response"], [0, 30, 0], 100)
        self.assertEqual(df["role"].to_list(), ["id", "feature", "target"])
        self.assertEqual(df["split"].to_list(), ["train"] * 3)
        self.assertEqual(df["data_kind"].to_list(), ["numeric"] * 3)
        row = df.filter(df["column"] == "L0_S1_F25").row(0, named=True)
        self.assertEqual((row["line"], row["station"], row["kind"], row["number"]), (0, 1, "F", 25))
        self.assertAlmostEqual(row["null_rate"], 0.3)

    def test_unknown_null_count(self):
        df = build_column_meta("test_date", ["L0_S0_D1"], [None], 10)
        self.assertIsNone(df["null_count"][0])
        self.assertIsNone(df["null_rate"][0])

    def test_length_mismatch(self):
        with self.assertRaises(ValueError):
            build_column_meta("train_numeric", ["Id"], [], 1)

    def test_invalid_feature_name_raises(self):
        with self.assertRaises(ValueError):
            build_column_meta("train_numeric", ["weird"], [0], 1)


if __name__ == "__main__":
    unittest.main()
