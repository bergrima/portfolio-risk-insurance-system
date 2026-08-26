import tempfile
import unittest
from pathlib import Path

import pandas as pd

from portfolio_insurance.backtest import ReviewFrequency
from portfolio_insurance.experiments import export_experiment, run_parameter_grid


class TestExperiments(unittest.TestCase):
    def test_grid_is_reproducible_and_exports_every_ledger(self):
        index = pd.date_range("2026-01-01", periods=8, freq="B", tz="UTC")
        prices = pd.DataFrame(
            {
                "fixed_income": [100 + i * 0.1 for i in range(8)],
                "gold": [100, 102, 98, 99, 103, 101, 104, 106],
                "equity": [100, 99, 97, 101, 100, 103, 102, 105],
            },
            index=index,
        )
        weights = {"fixed_income": 0.3, "gold": 0.4, "equity": 0.3}
        result = run_parameter_grid(
            prices,
            weights,
            multipliers=[2, 3],
            floors=[0.8],
            frequencies=[ReviewFrequency.DAILY, ReviewFrequency.MONTHLY],
            drift_bands=[0.02],
            reference_prices=pd.Series(range(100, 108), index=index),
        )
        self.assertEqual(len(result.table), 8)
        self.assertEqual(len(result.runs), 8)
        self.assertEqual(len(result.attributions), 8)
        self.assertTrue(result.baselines)
        repeated = run_parameter_grid(
            prices,
            weights,
            multipliers=[2, 3],
            floors=[0.8],
            frequencies=[ReviewFrequency.DAILY, ReviewFrequency.MONTHLY],
            drift_bands=[0.02],
            reference_prices=pd.Series(range(100, 108), index=index),
        )
        self.assertEqual(set(result.runs), set(repeated.runs))
        separated = run_parameter_grid(
            prices,
            weights,
            multipliers=[3],
            floors=[0.8],
            frequencies=[ReviewFrequency.DAILY],
            drift_bands=[0.10],
            protection_drift_band=0.025,
        )
        self.assertEqual(
            separated.table.loc[0, "protection_drift_band"], 0.025
        )
        with tempfile.TemporaryDirectory() as directory:
            export_experiment(result, directory)
            self.assertTrue((Path(directory) / "summary.csv").exists())
            self.assertTrue(
                all((Path(directory) / run_id / "ledger.csv").exists() for run_id in result.runs)
            )
            self.assertTrue(
                all(
                    (Path(directory) / run_id / "attribution.csv").exists()
                    for run_id in result.runs
                )
            )
