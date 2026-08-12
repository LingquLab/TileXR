from __future__ import annotations

import unittest

from tools.moonep.reduce_grad_benchmark import (
    ReduceGradDimensions,
    build_experts_to_copy,
    cross_rank_statistics,
    expected_expert_value,
    gradient_source_regions,
    plan_statistics,
)


class ReduceGradBenchmarkTests(unittest.TestCase):
    def test_gradient_tail_uses_full_gradient_as_registration_backing(self):
        class Gradient:
            def __init__(self):
                self.narrow_calls = []

            def narrow(self, dimension, start, length):
                source = object()
                self.narrow_calls.append((dimension, start, length, source))
                return source

        gradients = [Gradient(), Gradient(), Gradient()]
        sources, registrations = gradient_source_regions(gradients, 8, 4)
        self.assertEqual(registrations, gradients)
        self.assertEqual(
            sources, [gradient.narrow_calls[0][3] for gradient in gradients]
        )
        self.assertTrue(
            all(gradient.narrow_calls[0][:3] == (0, 8, 4) for gradient in gradients)
        )

    def test_rank_count_below_four_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "at least 4 ranks"):
            ReduceGradDimensions(3, 2, 2, 32, 32)
        with self.assertRaisesRegex(ValueError, "at least 4 ranks"):
            build_experts_to_copy(3, 2, 2, "full")

    def test_rank_by_slot_index_overflow_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "contributor indices"):
            ReduceGradDimensions(8, 8, (1 << 31) // 8 + 1, 32, 32)

    def test_baseline_full_plan_preserves_source_rank_slot_order(self):
        dimensions = ReduceGradDimensions(8, 8, 14, 3584, 3072)
        plan = build_experts_to_copy(
            dimensions.ranks,
            dimensions.slots,
            dimensions.experts_per_rank,
            "full",
        )
        self.assertEqual(plan[0], [-1] * 14)
        self.assertEqual(plan[1], list(range(8)) + [-1] * 6)
        self.assertEqual(plan[2], list(range(8)) + [-1] * 6)
        self.assertEqual(plan[3], list(range(8)) + [-1] * 6)
        self.assertEqual(plan[4], [-1] * 14)
        stats = plan_statistics(plan)
        self.assertEqual(stats["live_entries"], 24)
        self.assertEqual(stats["active_experts"], 8)
        self.assertEqual(stats["max_contributors_per_expert"], 3)
        self.assertNotEqual(expected_expert_value(plan, 0, 0), 0.0)

    def test_balanced_full_uses_every_source_slot_and_owner(self):
        plan = build_experts_to_copy(8, 14, 8, "balanced-full")
        stats = plan_statistics(plan)
        self.assertEqual(stats["live_entries"], 112)
        self.assertEqual(stats["density"], 1.0)
        self.assertTrue(all(0 <= expert < 64 for row in plan for expert in row))
        owners = {expert // 8 for row in plan for expert in row}
        self.assertEqual(owners, set(range(8)))

    def test_cross_rank_max_is_aggregated_before_percentiles(self):
        stats = cross_rank_statistics(((1.0, 7.0), (3.0, 4.0), (5.0, 2.0)))
        self.assertEqual(stats["cross_rank_max_us"], [5.0, 7.0])
        self.assertEqual(stats["p50_us"], 6.0)
        self.assertAlmostEqual(stats["p99_us"], 6.98)


if __name__ == "__main__":
    unittest.main()
