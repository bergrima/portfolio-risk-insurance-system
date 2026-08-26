import unittest

import pandas as pd

from portfolio_insurance.backtest import (
    BacktestPolicy,
    ExecutionModel,
    ReviewFrequency,
    Strategy,
    attribute_effects,
    relative_prices,
    run_backtest,
    run_baselines,
)


class TestBacktest(unittest.TestCase):
    def setUp(self):
        self.index = pd.date_range("2026-01-01", periods=6, freq="B", tz="UTC")
        self.prices = pd.DataFrame(
            {
                "fixed_income": [100] * 6,
                "gold": [100, 120, 120, 90, 90, 99],
                "equity": [100, 100, 80, 80, 88, 88],
            },
            index=self.index,
        )
        self.weights = {"fixed_income": 0.3, "gold": 0.4, "equity": 0.3}

    def test_no_same_bar_execution(self):
        result = run_backtest(
            self.prices, self.weights, BacktestPolicy(review_frequency=ReviewFrequency.DAILY)
        )
        done = result.ledger[result.ledger.order_executed]
        self.assertTrue((done.index.to_series().array > done.decision_at.array).all())

    def test_biweekly_vppi_keeps_strategic_weights_for_first_ten_bars(self):
        index = pd.date_range("2026-01-01", periods=13, freq="B", tz="UTC")
        prices = pd.DataFrame(
            {
                "fixed_income": [100] * len(index),
                "gold": [100] * len(index),
                "equity": [100] * len(index),
            },
            index=index,
        )
        result = run_backtest(
            prices,
            self.weights,
            BacktestPolicy(
                strategy=Strategy.VPPI,
                review_frequency=ReviewFrequency.BIWEEKLY,
                multiplier=3,
                floor_fraction=0.8,
            ),
        )

        self.assertFalse(result.ledger.iloc[:10]["reviewed"].any())
        self.assertFalse(result.ledger.iloc[:10]["order_scheduled"].any())
        self.assertTrue(result.ledger.iloc[10]["order_scheduled"])
        self.assertAlmostEqual(result.ledger.iloc[0]["weight_gold"], self.weights["gold"])

    def test_costs_and_reconciliation(self):
        free = run_backtest(
            self.prices, self.weights, BacktestPolicy(review_frequency=ReviewFrequency.DAILY)
        )
        costly = run_backtest(
            self.prices,
            self.weights,
            BacktestPolicy(
                review_frequency=ReviewFrequency.DAILY, execution=ExecutionModel(20, 10)
            ),
        )
        self.assertLess(costly.summary["ending_nav"], free.summary["ending_nav"])
        self.assertAlmostEqual(costly.summary["total_cost"], costly.ledger.cost.sum())

    def test_vppi_and_floor_reporting(self):
        result = run_backtest(
            self.prices,
            self.weights,
            BacktestPolicy(
                strategy=Strategy.VPPI,
                review_frequency=ReviewFrequency.DAILY,
                floor_fraction=0.95,
                multiplier=2,
            ),
        )
        self.assertTrue(result.ledger.order_scheduled.any())
        self.assertIn("floor_breach_probability", result.summary)
        self.assertTrue(
            result.ledger.loc[result.ledger.reviewed, "decision_target_gold"].notna().all()
        )

    def test_independent_sleeves_have_distinct_exposure_and_auditable_orders(self):
        result = run_backtest(
            self.prices,
            self.weights,
            BacktestPolicy(
                strategy=Strategy.VPPI,
                review_frequency=ReviewFrequency.DAILY,
                floor_fraction=0.9,
                multiplier=3,
                execution=ExecutionModel(latency_bars=2),
            ),
        )
        self.assertFalse(
            result.ledger["sleeve_nav_gold"].equals(result.ledger["sleeve_nav_equity"])
        )
        self.assertTrue(result.ledger["blocked_by_pending_order_id"].notna().any())
        result.verify_ledger()
        repeated = run_backtest(
            self.prices,
            self.weights,
            BacktestPolicy(strategy=Strategy.VPPI, review_frequency=ReviewFrequency.DAILY),
        )
        again = run_backtest(
            self.prices,
            self.weights,
            BacktestPolicy(strategy=Strategy.VPPI, review_frequency=ReviewFrequency.DAILY),
        )
        self.assertEqual(repeated.metadata["run_id"], again.metadata["run_id"])

    def test_baselines_attribution_and_reference(self):
        base = run_baselines(self.prices, self.weights)
        insured = run_backtest(self.prices, self.weights, BacktestPolicy(strategy=Strategy.VPPI))
        effects = attribute_effects(base["buy_and_hold"], base["calendar_rebalanced"], insured)
        pd.testing.assert_series_equal(effects.reconstructed, insured.returns, check_names=False)
        usd = pd.Series([1, 1, 2, 2, 2, 2], index=self.index)
        self.assertEqual(relative_prices(self.prices, usd).iloc[2].gold, 60)

    def test_order_waits_until_closed_asset_reopens(self):
        tradable = pd.DataFrame(True, index=self.index, columns=self.prices.columns)
        tradable.loc[self.index[2], "equity"] = False
        result = run_backtest(
            self.prices,
            self.weights,
            BacktestPolicy(
                strategy=Strategy.VPPI,
                review_frequency=ReviewFrequency.DAILY,
                floor_fraction=0.8,
                multiplier=2,
            ),
            tradable=tradable,
        )
        self.assertTrue(result.ledger.loc[self.index[2], "order_execution_deferred"])
        self.assertFalse(result.ledger.loc[self.index[2], "order_executed"])
        self.assertTrue(result.ledger.loc[self.index[3], "order_executed"])
        self.assertEqual(result.ledger.loc[self.index[2], "deferred_assets"], "equity")
        self.assertEqual(result.summary["deferred_execution_bars"], 1)
        result.verify_ledger()
        always_open = run_backtest(
            self.prices,
            self.weights,
            BacktestPolicy(strategy=Strategy.VPPI),
        )
        self.assertNotEqual(result.metadata["run_id"], always_open.metadata["run_id"])

    def test_fixed_income_accrues_for_every_calendar_day(self):
        index = pd.DatetimeIndex(["2026-01-01", "2026-01-04"], tz="UTC")
        prices = pd.DataFrame(
            {"fixed_income": [100, 100], "gold": [100, 100], "equity": [100, 100]},
            index=index,
        )
        weights = {"fixed_income": 0.8, "gold": 0.1, "equity": 0.1}
        result = run_backtest(
            prices,
            weights,
            BacktestPolicy(
                strategy=Strategy.BUY_AND_HOLD,
                reserve_daily_return=0.0012,
                reserve_calendar_day_accrual=True,
            ),
            initial_nav=1_000_000,
        )

        expected = 1_000_000 * (0.8 * 1.0012**3 + 0.2)
        self.assertAlmostEqual(result.summary["ending_nav"], expected)
