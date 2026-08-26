import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from portfolio_insurance.backtest import ExecutionModel
from portfolio_insurance.policy_selection import (
    SelectionCriteria,
    export_policy_selection,
    load_selection_criteria,
    run_policy_selection,
)
from portfolio_insurance.scenarios import constant_inflation_index


class TestPolicySelection(unittest.TestCase):
    def setUp(self):
        self.initial_nav = 1_000_000
        self.weights = {"fixed_income": 0.3, "gold": 0.4, "equity": 0.3}
        self.criteria = SelectionCriteria(
            return_weight=0.30,
            drawdown_weight=0.25,
            breach_weight=0.35,
            cost_weight=0.10,
            minimum_neighbors=2,
            minimum_feasible_neighbor_fraction=0,
            maximum_neighbor_utility_gap=1,
        )
        rows = []
        policy_outcomes = {
            1: (0.20, 0.30, 1.00, 3_000),
            2: (0.14, 0.14, 0.25, 1_400),
            3: (0.09, 0.06, 0.00, 500),
            4: (0.08, 0.10, 0.25, 1_000),
        }
        for scenario_number, scenario in enumerate(("baseline", "joint_decline")):
            for path_number in range(4):
                for multiplier, (
                    real_return,
                    drawdown,
                    breach_rate,
                    cost,
                ) in policy_outcomes.items():
                    breached = path_number < round(breach_rate * 4)
                    rows.append(
                        {
                            "path_id": f"{scenario}-{path_number}",
                            "run_id": f"{scenario}-{path_number}-{multiplier}",
                            "generator": "moving_block_bootstrap",
                            "scenario": scenario,
                            "multiplier": multiplier,
                            "floor_fraction": 0.8,
                            "review_frequency": "monthly",
                            "drift_band": 0.05,
                            "real_return": real_return - 0.02 * scenario_number,
                            "maximum_drawdown": -drawdown,
                            "ever_breached": breached,
                            "total_cost": cost,
                        }
                    )
        self.path_metrics = pd.DataFrame(rows)

        self.index = pd.date_range("2024-01-01", periods=91, freq="B", tz="UTC")
        pattern = np.resize(
            np.array([0.006, -0.004, 0.003, -0.007, 0.008, 0.001]),
            len(self.index) - 1,
        )
        returns = pd.DataFrame(
            {
                "fixed_income": np.full(len(pattern), 0.0003),
                "gold": pattern,
                "equity": pattern * 1.25 - 0.0002,
            },
            index=self.index[1:],
        )
        self.prices = pd.concat(
            [
                pd.DataFrame([[100, 100, 100]], columns=returns.columns, index=self.index[:1]),
                100 * (1 + returns).cumprod(),
            ]
        )
        self.inflation = constant_inflation_index(self.index, 0.20)

    def _run(self, criteria=None, path_metrics=None):
        return run_policy_selection(
            self.path_metrics if path_metrics is None else path_metrics,
            self.prices,
            self.inflation,
            self.weights,
            "medium_term",
            criteria or self.criteria,
            test_bars=15,
            folds=3,
            minimum_train_bars=30,
            execution=ExecutionModel(10, 5, 1),
            initial_nav=self.initial_nav,
        )

    def test_selects_stable_pareto_policy_and_marks_dominated_policy(self):
        result = self._run()
        self.assertEqual(result.status, "selected")
        self.assertEqual(result.selected_policy["multiplier"], 3)
        selected = result.candidates[result.candidates["is_selected"]].iloc[0]
        self.assertTrue(selected["is_pareto"])
        self.assertTrue(selected["is_stable_plateau"])
        dominated = result.candidates[result.candidates["multiplier"] == 4].iloc[0]
        self.assertFalse(dominated["is_pareto"])

    def test_walk_forward_uses_later_non_overlapping_test_returns(self):
        result = self._run()
        folds = result.walk_forward.folds
        self.assertEqual(len(folds), 3)
        self.assertTrue((pd.to_datetime(folds.train_end) < pd.to_datetime(folds.test_start)).all())
        counts = result.walk_forward.policy_metrics.groupby("fold").size()
        self.assertTrue((counts == 4).all())
        self.assertIn(
            result.validation_status,
            {"completed_no_mandate_gates", "incomplete_fold_selection"},
        )

    def test_hard_gates_refuse_to_select_an_infeasible_surface(self):
        strict = SelectionCriteria(
            return_weight=0.30,
            drawdown_weight=0.25,
            breach_weight=0.35,
            cost_weight=0.10,
            minimum_neighbors=2,
            minimum_feasible_neighbor_fraction=0,
            maximum_neighbor_utility_gap=1,
            minimum_real_return=0.50,
        )
        result = self._run(strict)
        self.assertEqual(result.status, "no_feasible_policy")
        self.assertIsNone(result.selected_policy)
        self.assertFalse(result.candidates["constraint_feasible"].any())

    def test_strict_return_gate_requires_outperformance_not_equality(self):
        strict = SelectionCriteria(
            return_weight=0.30,
            drawdown_weight=0.25,
            breach_weight=0.35,
            cost_weight=0.10,
            minimum_neighbors=2,
            minimum_feasible_neighbor_fraction=0,
            maximum_neighbor_utility_gap=1,
            minimum_real_return=0.0,
            strict_minimum_real_return=True,
        )
        equal_to_benchmark = self.path_metrics.copy()
        equal_to_benchmark["real_return"] = 0.0

        result = self._run(strict, equal_to_benchmark)

        self.assertEqual(result.status, "no_feasible_policy")
        self.assertFalse(result.candidates["constraint_feasible"].any())

    def test_policy_selection_rejects_a_non_dollar_benchmark(self):
        with self.assertRaisesRegex(ValueError, "USD/IRR"):
            run_policy_selection(
                self.path_metrics,
                self.prices,
                self.inflation,
                self.weights,
                "medium_term",
                self.criteria,
                test_bars=15,
                folds=3,
                minimum_train_bars=30,
                inflation_benchmark_name="cpi",
            )

    def test_result_is_reproducible_and_exports_auditable_artifacts(self):
        first = self._run()
        second = self._run()
        self.assertEqual(first.metadata["selection_id"], second.metadata["selection_id"])
        pd.testing.assert_frame_equal(first.candidates, second.candidates)
        pd.testing.assert_frame_equal(first.walk_forward.folds, second.walk_forward.folds)
        with tempfile.TemporaryDirectory() as directory:
            export_policy_selection(first, directory)
            expected = {
                "candidate_policies.csv",
                "context_metrics.csv",
                "walk_forward_folds.csv",
                "walk_forward_policy_metrics.csv",
                "walk_forward_candidates.csv",
                "selected_policy.json",
                "metadata.json",
            }
            self.assertEqual(expected, {path.name for path in Path(directory).iterdir()})

    def test_versioned_profiles_are_separate(self):
        config_path = Path(__file__).parents[1] / "configs" / "policy_selection.toml"
        medium = load_selection_criteria(config_path, "medium_term")
        long = load_selection_criteria(config_path, "long_term")
        self.assertGreater(medium.breach_weight, long.breach_weight)
        self.assertGreater(long.return_weight, medium.return_weight)
        self.assertEqual(medium.minimum_real_return, 0.0)
        self.assertFalse(medium.strict_minimum_real_return)
        self.assertEqual(long.minimum_real_return, 0.0)
        self.assertTrue(long.strict_minimum_real_return)


if __name__ == "__main__":
    unittest.main()
