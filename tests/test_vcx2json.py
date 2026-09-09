import unittest

from CryoNetRefine.data.output.vcx2json import add_percentile


class AddPercentileTest(unittest.TestCase):
    def test_accepts_missing_optional_outlier_metrics(self):
        scores = {
            "CC_mask": 0.5,
            "rotamer_outliers": 0.0,
            "cbeta_deviations": 0.0,
        }

        result = add_percentile(scores)

        self.assertNotIn("rama_outliers", result)
        self.assertNotIn("rama_outliers_perc", result)
        self.assertEqual(result["rotamer_outliers_perc"], 1.0)
        self.assertEqual(result["cbeta_deviations_perc"], 1.0)

    def test_calibrates_present_zero_rama_outliers(self):
        result = add_percentile({"rama_outliers": 0.0})

        self.assertEqual(result["rama_outliers_perc"], 1.0)


if __name__ == "__main__":
    unittest.main()
