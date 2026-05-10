from __future__ import annotations

import math
import unittest

from src.functional_eval.validation import compare_surfaces, sort_surface


class SurfaceValidationTest(unittest.TestCase):
    def test_sort_surface_orders_by_row_then_col(self) -> None:
        records = [(1, 0, 10.0), (0, 2, 2.0), (0, 1, 1.0)]

        self.assertEqual(
            [(0, 1, 1.0), (0, 2, 2.0), (1, 0, 10.0)],
            sort_surface(records),
        )

    def test_compare_accepts_unsorted_equivalent_records(self) -> None:
        comparison = compare_surfaces(
            [(1, 0, 2.0), (0, 0, 1.0)],
            [(0, 0, 1.0 + 5e-7), (1, 0, 2.0)],
        )

        self.assertTrue(comparison.allclose)
        self.assertEqual(0, comparison.mismatch_count)
        self.assertEqual(2, comparison.point_count)
        self.assertAlmostEqual(5e-7, comparison.max_abs_error)
        self.assertGreater(comparison.rmse, 0.0)

    def test_compare_reports_finite_mismatches(self) -> None:
        comparison = compare_surfaces(
            [(0, 0, 1.0), (0, 1, 2.0), (0, 2, 4.0)],
            [(0, 0, 1.1), (0, 1, 2.0), (0, 2, 5.0)],
            max_mismatches=1,
        )

        self.assertFalse(comparison.allclose)
        self.assertEqual(2, comparison.mismatch_count)
        self.assertEqual(1, len(comparison.first_mismatches))
        first = comparison.first_mismatches[0]
        self.assertEqual((0, 0), (first.row, first.col))
        self.assertAlmostEqual(1.0, comparison.max_abs_error)
        self.assertAlmostEqual(
            math.sqrt((0.1**2 + 0.0 + 1.0) / 3),
            comparison.rmse,
        )

    def test_compare_treats_paired_nan_as_matching(self) -> None:
        comparison = compare_surfaces(
            [(0, 0, math.nan)],
            [(0, 0, math.nan)],
        )

        self.assertTrue(comparison.allclose)
        self.assertEqual(0.0, comparison.max_abs_error)
        self.assertEqual(0.0, comparison.rmse)

    def test_compare_requires_infinity_sign_match(self) -> None:
        matching = compare_surfaces(
            [(0, 0, math.inf), (0, 1, -math.inf)],
            [(0, 0, math.inf), (0, 1, -math.inf)],
        )
        mismatching = compare_surfaces(
            [(0, 0, math.inf)],
            [(0, 0, -math.inf)],
        )

        self.assertTrue(matching.allclose)
        self.assertFalse(mismatching.allclose)
        self.assertEqual(1, mismatching.mismatch_count)
        self.assertTrue(math.isinf(mismatching.max_abs_error))
        self.assertTrue(math.isinf(mismatching.rmse))

    def test_compare_rejects_point_count_mismatch(self) -> None:
        with self.assertRaisesRegex(AssertionError, "point count"):
            compare_surfaces([(0, 0, 1.0)], [])

    def test_compare_rejects_coordinate_alignment_mismatch(self) -> None:
        with self.assertRaisesRegex(AssertionError, "coordinate mismatch"):
            compare_surfaces([(0, 0, 1.0)], [(0, 1, 1.0)])


if __name__ == "__main__":
    unittest.main()
