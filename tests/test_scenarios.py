import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from portfolio_insurance.backtest import ExecutionModel, ReviewFrequency
from portfolio_insurance.scenarios import (
    HistoricalStressWindow,
    ScenarioPath,
    SimulationConfig,
    SimulationMethod,
    StressScenario,
    apply_stress,
    constant_inflation_index,
    export_scenario_analysis,
    generate_scenario_paths,
    historical_stress_paths,
    run_scenario_analysis,
)


class TestScenarios(unittest.TestCase):
    def setUp(self):
        self.index = pd.date_range("2025-01-01", periods=60, freq="B", tz="UTC")
        pattern = np.array([0.01, 0.012, -0.008, -0.015, 0.006, 0.004, -0.02, 0.018])
        gold_returns = np.resize(pattern, len(self.index) - 1)
        equity_returns = np.resize(pattern * 1.4 - 0.001, len(self.index) - 1)
        fixed_returns = np.full(len(self.index) - 1, 0.0003)
        returns = pd.DataFrame(
            {
                "fixed_income": fixed_returns,
                "gold": gold_returns,
                "equity": equity_returns,
            },
            index=self.index[1:],
        )
        self.prices = pd.concat(
            [
                pd.DataFrame([[100, 100, 100]], columns=returns.columns, index=self.index[:1]),
                100 * (1 + returns).cumprod(),
            ]
        )
        self.inflation = constant_inflation_index(self.index, 0.30)
        self.fx = pd.Series(
            100 * np.power(1.0008, np.arange(len(self.index))), self.index, name="fx"
        )
        self.weights = {"fixed_income": 0.3, "gold": 0.4, "equity": 0.3}

    def test_block_bootstrap_is_joint_and_reproducible(self):
        config = SimulationConfig(paths=3, horizon_bars=20, block_size=4, seed=17)
        tradable = pd.DataFrame(True, index=self.index, columns=self.prices.columns)
        tradable.loc[self.index[4:8], "equity"] = False
        first = generate_scenario_paths(self.prices, config, self.inflation, self.fx, tradable)
        second = generate_scenario_paths(self.prices, config, self.inflation, self.fx, tradable)
        self.assertEqual([path.seed for path in first], [path.seed for path in second])
        for left, right in zip(first, second, strict=True):
            pd.testing.assert_frame_equal(left.prices, right.prices)
            pd.testing.assert_series_equal(left.inflation_index, right.inflation_index)
            pd.testing.assert_series_equal(left.fx_reference, right.fx_reference)
            pd.testing.assert_frame_equal(left.tradable, right.tradable)

        source = (
            pd.concat([self.prices, self.inflation.rename("inflation"), self.fx], axis=1)
            .pct_change(fill_method=None)
            .dropna()
        )
        generated = (
            pd.concat(
                [
                    first[0].prices,
                    first[0].inflation_index.rename("inflation"),
                    first[0].fx_reference,
                ],
                axis=1,
            )
            .pct_change(fill_method=None)
            .dropna()
        )
        source_rows = {tuple(np.round(row, 12)) for row in source.to_numpy()}
        self.assertTrue(
            all(tuple(np.round(row, 12)) in source_rows for row in generated.to_numpy())
        )

    def test_regime_switching_paths_are_positive_and_seeded(self):
        config = SimulationConfig(
            method=SimulationMethod.REGIME_SWITCHING,
            paths=2,
            horizon_bars=15,
            seed=41,
            student_t_df=4,
        )
        first = generate_scenario_paths(self.prices, config, self.inflation, self.fx)
        second = generate_scenario_paths(self.prices, config, self.inflation, self.fx)
        self.assertTrue((first[0].prices > 0).all().all())
        pd.testing.assert_frame_equal(first[0].prices, second[0].prices)

    def test_stationary_bootstrap_is_reproducible(self):
        config = SimulationConfig(
            method=SimulationMethod.STATIONARY_BOOTSTRAP,
            paths=2,
            horizon_bars=15,
            block_size=5,
            seed=73,
        )
        first = generate_scenario_paths(self.prices, config, self.inflation, self.fx)
        second = generate_scenario_paths(self.prices, config, self.inflation, self.fx)
        pd.testing.assert_frame_equal(first[1].prices, second[1].prices)
        self.assertEqual(first[1].generator, SimulationMethod.STATIONARY_BOOTSTRAP.value)

    def test_market_fx_and_liquidity_stress_is_explicit(self):
        path = ScenarioPath(
            "test-path",
            "test",
            "baseline",
            self.prices.iloc[:8],
            self.inflation.iloc[:8],
            self.fx.iloc[:8],
            1,
        )
        stress = StressScenario(
            "combined",
            shock_bar=2,
            asset_shocks={"gold": -0.20, "equity": -0.25},
            fx_jump=0.10,
            fx_asset_betas={"gold": 1.0},
            liquidity_cost_multiplier=3,
            extra_slippage_bps=20,
            latency_bars_addition=1,
        )
        stressed = apply_stress(path, stress)
        pd.testing.assert_frame_equal(stressed.prices.iloc[:2], path.prices.iloc[:2])
        original_return = path.prices.gold.pct_change(fill_method=None).iloc[2]
        stressed_return = stressed.prices.gold.pct_change(fill_method=None).iloc[2]
        self.assertAlmostEqual(stressed_return, (1 + original_return) * 0.8 * 1.1 - 1)
        original_fx_return = path.fx_reference.pct_change(fill_method=None).iloc[2]
        stressed_fx_return = stressed.fx_reference.pct_change(fill_method=None).iloc[2]
        self.assertAlmostEqual(stressed_fx_return, (1 + original_fx_return) * 1.1 - 1)

    def test_historical_windows_keep_realized_dates(self):
        window = HistoricalStressWindow("realized_selloff", self.index[10], self.index[20])
        path = historical_stress_paths(self.prices, [window], self.inflation, self.fx)[0]
        self.assertEqual(path.generator, "historical")
        self.assertEqual(path.prices.index[0], self.index[10])
        self.assertEqual(path.prices.index[-1], self.index[20])

    def test_analysis_aggregates_true_breach_probability_and_exports(self):
        simulation = SimulationConfig(paths=8, horizon_bars=16, block_size=4, seed=99)
        stress = StressScenario(
            "joint_decline",
            asset_shocks={"gold": -0.35, "equity": -0.45},
            liquidity_cost_multiplier=2,
            extra_slippage_bps=10,
        )
        result = run_scenario_analysis(
            self.prices,
            self.weights,
            multipliers=[2, 3],
            floors=[0.90],
            frequencies=[ReviewFrequency.DAILY],
            drift_bands=[0.02],
            simulation=simulation,
            execution=ExecutionModel(10, 5, 1),
            inflation_index=self.inflation,
            fx_reference=self.fx,
            stresses=[stress],
        )
        self.assertEqual(len(result.surface), 4)
        self.assertEqual(len(result.path_metrics), 32)
        self.assertTrue(
            {
                "real_return_lower",
                "real_return_upper",
                "drawdown_loss_tail_quantile",
                "drawdown_loss_expected_shortfall",
                "real_drawdown_loss_tail_quantile",
                "real_drawdown_loss_expected_shortfall",
                "floor_breach_probability_lower",
                "floor_breach_probability_upper",
                "real_floor_breach_probability_lower",
                "real_floor_breach_probability_upper",
            }
            <= set(result.surface)
        )
        self.assertTrue(
            {
                "real_maximum_drawdown",
                "usd_relative_maximum_drawdown",
                "maximum_floor_shortfall",
                "real_floor_maximum_shortfall",
                "real_floor_maximum_breach_duration_bars",
            }
            <= set(result.path_metrics)
        )
        self.assertTrue((result.path_metrics["maximum_floor_shortfall"] >= 0).all())
        self.assertTrue((result.path_metrics["real_floor_maximum_shortfall"] >= 0).all())
        self.assertTrue(
            (
                result.surface["drawdown_loss_expected_shortfall"]
                >= result.surface["drawdown_loss_tail_quantile"]
            ).all()
        )
        group = result.path_metrics[
            (result.path_metrics.scenario == "joint_decline")
            & (result.path_metrics.multiplier == 3)
        ]
        surface_row = result.surface[
            (result.surface.scenario == "joint_decline") & (result.surface.multiplier == 3)
        ].iloc[0]
        self.assertAlmostEqual(surface_row.floor_breach_probability, group.ever_breached.mean())

        repeated = run_scenario_analysis(
            self.prices,
            self.weights,
            multipliers=[2, 3],
            floors=[0.90],
            frequencies=[ReviewFrequency.DAILY],
            drift_bands=[0.02],
            simulation=simulation,
            execution=ExecutionModel(10, 5, 1),
            inflation_index=self.inflation,
            fx_reference=self.fx,
            stresses=[stress],
        )
        self.assertEqual(result.metadata["analysis_id"], repeated.metadata["analysis_id"])
        pd.testing.assert_frame_equal(result.surface, repeated.surface)
        pd.testing.assert_frame_equal(result.path_metrics, repeated.path_metrics)

        with tempfile.TemporaryDirectory() as directory:
            export_scenario_analysis(result, directory)
            self.assertTrue((Path(directory) / "surface.csv").exists())
            self.assertTrue((Path(directory) / "path_metrics.csv").exists())
            self.assertTrue((Path(directory) / "metadata.json").exists())

    def test_parallel_analysis_matches_single_worker(self):
        simulation = SimulationConfig(paths=2, horizon_bars=8, block_size=3, seed=123)
        arguments = {
            "prices": self.prices,
            "weights": self.weights,
            "multipliers": [3],
            "floors": [0.8],
            "frequencies": [ReviewFrequency.WEEKLY],
            "drift_bands": [0.05],
            "simulation": simulation,
            "inflation_index": self.inflation,
            "fx_reference": self.fx,
        }
        sequential = run_scenario_analysis(**arguments, max_workers=1)
        parallel = run_scenario_analysis(**arguments, max_workers=2)
        pd.testing.assert_frame_equal(sequential.surface, parallel.surface)
        pd.testing.assert_frame_equal(sequential.path_metrics, parallel.path_metrics)
        self.assertEqual(sequential.metadata["analysis_id"], parallel.metadata["analysis_id"])


if __name__ == "__main__":
    unittest.main()
