import tempfile
import unittest
from pathlib import Path

import pandas as pd

from portfolio_insurance.backtest import (
    BacktestPolicy,
    ReviewFrequency,
    Strategy,
    run_backtest,
)
from portfolio_insurance.regime import (
    CalibrationSettings,
    RegimePolicy,
    calibrate_regimes,
    confirmed_zigzag,
    export_regime_calibration,
    load_regime_calibration,
)


class TestRegime(unittest.TestCase):
    def test_confirmed_zigzag_never_exposes_a_pivot_before_confirmation(self):
        index = pd.date_range("2025-01-01", periods=9, freq="B", tz="UTC")
        levels = pd.Series([100, 110, 108, 103, 98, 94, 100, 106, 101], index=index)

        pivots = confirmed_zigzag(levels, 0.05)

        self.assertFalse(pivots.empty)
        self.assertTrue((pivots["confirmed_at"] > pivots["pivot_at"]).all())

    def test_calibration_exports_auditable_thresholds_and_round_trips(self):
        index = pd.date_range("2024-01-01", periods=25, freq="B", tz="UTC")
        pattern = [
            100,
            112,
            105,
            90,
            97,
            116,
            108,
            92,
            99,
            120,
            111,
            94,
            102,
            124,
            115,
            96,
            104,
            128,
            118,
            98,
            106,
            130,
            120,
            100,
            108,
        ]
        levels = pd.Series(pattern, index=index)
        settings = CalibrationSettings(
            reversal_threshold=0.05,
            continuation_step=0.01,
            grid_step=0.01,
            minimum_samples=2,
            target_probability=0.50,
        )
        bundle = calibrate_regimes(
            {"gold": levels, "equity": levels * 1.1},
            valid_after=index[-1] + pd.Timedelta(days=1),
            settings=settings,
        )

        self.assertEqual(set(bundle.policies), {"gold", "equity"})
        self.assertTrue((bundle.pivots["confirmed_at"] > bundle.pivots["pivot_at"]).all())
        self.assertTrue((bundle.legs["known_at"] >= bundle.legs["end_at"]).all())
        with tempfile.TemporaryDirectory() as directory:
            export_regime_calibration(bundle, directory)
            loaded = load_regime_calibration(directory)
            self.assertEqual(
                bundle.metadata["calibration_id"], loaded.metadata["calibration_id"]
            )
            self.assertEqual(bundle.policies, loaded.policies)
            self.assertTrue((Path(directory) / "continuation_profiles.csv").is_file())

    def test_regime_gate_starts_at_strategic_weights_then_activates_and_exits(self):
        index = pd.date_range("2026-01-01", periods=8, freq="B", tz="UTC")
        prices = pd.DataFrame(
            {
                "fixed_income": [100] * 8,
                "gold": [100, 100, 94, 89, 85, 87, 94, 96],
                "equity": [100] * 8,
            },
            index=index,
        )
        weights = {"fixed_income": 0.3, "gold": 0.4, "equity": 0.3}
        policies = {
            "gold": RegimePolicy(0.10, 0.10),
            "equity": RegimePolicy(0.20, 0.10),
        }

        result = run_backtest(
            prices,
            weights,
            BacktestPolicy(
                strategy=Strategy.VPPI,
                review_frequency=ReviewFrequency.DAILY,
                drift_band=0.05,
                multiplier=3,
                floor_fraction=0.8,
                regime_policies=policies,
                regime_calibration_id="test-calibration",
            ),
        )

        first = result.ledger.iloc[0]
        self.assertAlmostEqual(first["weight_gold"], weights["gold"])
        self.assertFalse(first["order_scheduled"])
        entries = result.ledger[
            result.ledger["regime_transition_gold"] == "enter_protection"
        ]
        exits = result.ledger[result.ledger["regime_transition_gold"] == "exit_protection"]
        self.assertEqual(len(entries), 1)
        self.assertEqual(len(exits), 1)
        self.assertTrue(entries.iloc[0]["order_scheduled"])
        self.assertTrue(exits.iloc[0]["order_scheduled"])
        self.assertLess(result.ledger.loc[index[4], "weight_gold"], weights["gold"])
        self.assertAlmostEqual(result.ledger.loc[index[7], "weight_gold"], weights["gold"])
        self.assertEqual(result.summary["protection_activation_count_gold"], 1)
        result.verify_ledger()

    def test_protection_band_can_trigger_when_ordinary_band_would_not(self):
        index = pd.date_range("2026-01-01", periods=6, freq="B", tz="UTC")
        prices = pd.DataFrame(
            {
                "fixed_income": [100] * 6,
                "gold": [100, 100, 94, 94, 94, 94],
                "equity": [100] * 6,
            },
            index=index,
        )
        weights = {"fixed_income": 0.3, "gold": 0.4, "equity": 0.3}
        regimes = {
            "gold": RegimePolicy(0.05, 0.10),
            "equity": RegimePolicy(0.20, 0.10),
        }

        separated = run_backtest(
            prices,
            weights,
            BacktestPolicy(
                strategy=Strategy.VPPI,
                review_frequency=ReviewFrequency.DAILY,
                drift_band=0.50,
                protection_drift_band=0.05,
                multiplier=3,
                floor_fraction=0.8,
                regime_policies=regimes,
                regime_calibration_id="separate-band-test",
            ),
        )
        legacy = run_backtest(
            prices,
            weights,
            BacktestPolicy(
                strategy=Strategy.VPPI,
                review_frequency=ReviewFrequency.DAILY,
                drift_band=0.50,
                multiplier=3,
                floor_fraction=0.8,
                regime_policies=regimes,
                regime_calibration_id="legacy-band-test",
            ),
        )

        entry = separated.ledger[
            separated.ledger["regime_transition_gold"] == "enter_protection"
        ].iloc[0]
        self.assertTrue(entry["order_scheduled"])
        self.assertFalse(legacy.ledger["order_scheduled"].any())
        self.assertEqual(
            separated.metadata["policy"]["protection_drift_band"], 0.05
        )
        self.assertNotEqual(
            separated.metadata["run_id"], legacy.metadata["run_id"]
        )


if __name__ == "__main__":
    unittest.main()
