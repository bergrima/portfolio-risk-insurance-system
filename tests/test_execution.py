import sqlite3
import tempfile
import unittest
from contextlib import closing
from pathlib import Path

from portfolio_insurance.backtest import ExecutionModel
from portfolio_insurance.execution import (
    ExecutionControls,
    OrderStatus,
    PaperPortfolio,
    load_execution_controls,
)


class TestControlledExecution(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.path = Path(self.temporary.name) / "paper.db"
        self.weights = {"fixed_income": 0.3, "gold": 0.4, "equity": 0.3}
        self.prices = {"fixed_income": 100.0, "gold": 100.0, "equity": 100.0}
        self.policy = {
            "multiplier": 3.0,
            "floor_fraction": 0.8,
            "review_frequency": "daily",
            "drift_band": 0.05,
        }
        self.selection = {
            "portfolio": "medium_term",
            "selection_id": "selection-001",
            "source_analysis_id": "analysis-001",
            "selection_status": "selected",
            "validation_status": "completed_no_mandate_gates",
        }

    def tearDown(self):
        self.temporary.cleanup()

    def _account(self, controls=None):
        return PaperPortfolio.initialize(
            self.path,
            portfolio="medium_term",
            strategic_weights=self.weights,
            selected_policy=self.policy,
            selection_metadata=self.selection,
            prices=self.prices,
            valuation_at="2026-08-20T12:00:00Z",
            approved_by="investment-committee",
            approved_at="2026-08-20T13:00:00Z",
            controls=controls or ExecutionControls(max_daily_turnover=1.0),
            execution=ExecutionModel(10, 5, 1),
        )

    def test_initialization_records_approval_and_verified_dashboard(self):
        account = self._account()
        state = account.state()
        self.assertTrue(state["audit"]["valid"])
        self.assertEqual(state["audit"]["events"], 2)
        self.assertEqual(state["valuation"]["reconciliation_status"], "matched")
        dashboard = self.path.with_name("dashboard.html")
        self.assertTrue(dashboard.is_file())
        dashboard_text = dashboard.read_text(encoding="utf-8")
        self.assertIn("Distance to floor", dashboard_text)
        self.assertIn("Human approval required", dashboard_text)

    def test_order_requires_approval_and_executes_only_on_later_valuation(self):
        account = self._account()
        proposal = account.run_day(
            prices=self.prices,
            valuation_at="2026-08-21T12:00:00Z",
            observed_at="2026-08-21T13:00:00Z",
        )
        self.assertEqual(proposal.order_status, OrderStatus.PENDING_APPROVAL.value)
        self.assertIsNotNone(proposal.order_id)
        state = account.state()
        self.assertEqual(state["orders"][0]["approved_by"], None)

        repeated = account.run_day(
            prices=self.prices,
            valuation_at="2026-08-21T12:00:00Z",
            observed_at="2026-08-21T13:30:00Z",
        )
        self.assertTrue(repeated.idempotent)
        account.approve_order(
            proposal.order_id,
            approved_by="portfolio-manager",
            approved_at="2026-08-21T14:00:00Z",
        )
        approved = account.state()["orders"][0]
        self.assertEqual(approved["status"], OrderStatus.APPROVED.value)
        self.assertIsNone(approved["executed_at"])

        execution = account.run_day(
            prices={"fixed_income": 100.1, "gold": 101.0, "equity": 99.0},
            valuation_at="2026-08-22T12:00:00Z",
            observed_at="2026-08-22T13:00:00Z",
        )
        self.assertEqual(execution.order_id, proposal.order_id)
        executed = account.state()["orders"][0]
        self.assertEqual(executed["status"], OrderStatus.EXECUTED.value)
        self.assertGreater(executed["total_cost"], 0)
        with closing(sqlite3.connect(self.path)) as connection:
            fills = connection.execute(
                "SELECT COUNT(*) FROM fills WHERE order_id = ?", (proposal.order_id,)
            ).fetchone()[0]
        self.assertEqual(fills, 3)
        self.assertTrue(account.verify_audit_chain())

    def test_turnover_limit_blocks_order_before_approval(self):
        account = self._account(ExecutionControls(max_daily_turnover=0.05))
        result = account.run_day(
            prices=self.prices,
            valuation_at="2026-08-21T12:00:00Z",
            observed_at="2026-08-21T13:00:00Z",
        )
        self.assertEqual(result.order_status, OrderStatus.CONTROL_BLOCKED.value)
        self.assertIn("turnover_limit", {alert["code"] for alert in result.alerts})
        with self.assertRaisesRegex(ValueError, "not pending approval"):
            account.approve_order(
                result.order_id,
                approved_by="portfolio-manager",
                approved_at="2026-08-21T14:00:00Z",
            )

    def test_approved_order_waits_for_a_valuation_after_approval(self):
        account = self._account()
        proposal = account.run_day(
            prices=self.prices,
            valuation_at="2026-08-21T12:00:00Z",
            observed_at="2026-08-21T13:00:00Z",
        )
        account.approve_order(
            proposal.order_id,
            approved_by="portfolio-manager",
            approved_at="2026-08-22T13:00:00Z",
        )
        account.run_day(
            prices=self.prices,
            valuation_at="2026-08-22T12:00:00Z",
            observed_at="2026-08-22T14:00:00Z",
        )
        self.assertEqual(account.state()["orders"][0]["status"], OrderStatus.APPROVED.value)
        account.run_day(
            prices=self.prices,
            valuation_at="2026-08-23T12:00:00Z",
            observed_at="2026-08-23T13:00:00Z",
        )
        self.assertEqual(account.state()["orders"][0]["status"], OrderStatus.EXECUTED.value)

    def test_pre_trade_reconciliation_break_blocks_an_approved_order(self):
        account = self._account()
        proposal = account.run_day(
            prices=self.prices,
            valuation_at="2026-08-21T12:00:00Z",
            observed_at="2026-08-21T13:00:00Z",
        )
        account.approve_order(
            proposal.order_id,
            approved_by="portfolio-manager",
            approved_at="2026-08-21T14:00:00Z",
        )
        result = account.run_day(
            prices=self.prices,
            valuation_at="2026-08-22T12:00:00Z",
            observed_at="2026-08-22T13:00:00Z",
            observed_units={"fixed_income": 0.0, "gold": 0.0, "equity": 0.0},
        )
        self.assertIn("execution_blocked", {alert["code"] for alert in result.alerts})
        self.assertEqual(account.state()["orders"][0]["status"], OrderStatus.APPROVED.value)
        with closing(sqlite3.connect(self.path)) as connection:
            self.assertEqual(connection.execute("SELECT COUNT(*) FROM fills").fetchone()[0], 0)

    def test_approved_order_waits_for_closed_market(self):
        account = self._account()
        proposal = account.run_day(
            prices=self.prices,
            valuation_at="2026-08-21T12:00:00Z",
            observed_at="2026-08-21T13:00:00Z",
        )
        account.approve_order(
            proposal.order_id,
            approved_by="portfolio-manager",
            approved_at="2026-08-21T14:00:00Z",
        )
        deferred = account.run_day(
            prices=self.prices,
            tradable={"fixed_income": True, "gold": True, "equity": False},
            valuation_at="2026-08-22T12:00:00Z",
            observed_at="2026-08-22T13:00:00Z",
        )
        self.assertIn(
            "execution_deferred_market_closed", {alert["code"] for alert in deferred.alerts}
        )
        self.assertEqual(account.state()["orders"][0]["status"], OrderStatus.APPROVED.value)
        account.run_day(
            prices=self.prices,
            valuation_at="2026-08-23T12:00:00Z",
            observed_at="2026-08-23T13:00:00Z",
        )
        self.assertEqual(account.state()["orders"][0]["status"], OrderStatus.EXECUTED.value)

    def test_order_cannot_be_approved_after_expiry(self):
        account = self._account()
        proposal = account.run_day(
            prices=self.prices,
            valuation_at="2026-08-21T12:00:00Z",
            observed_at="2026-08-21T13:00:00Z",
        )
        with self.assertRaisesRegex(ValueError, "expired"):
            account.approve_order(
                proposal.order_id,
                approved_by="portfolio-manager",
                approved_at="2026-08-22T14:00:00Z",
            )
        self.assertEqual(account.state()["orders"][0]["status"], OrderStatus.EXPIRED.value)
        self.assertIn("Expired", self.path.with_name("dashboard.html").read_text(encoding="utf-8"))

    def test_kill_switch_cancels_open_order_and_prevents_approval(self):
        account = self._account()
        proposal = account.run_day(
            prices=self.prices,
            valuation_at="2026-08-21T12:00:00Z",
            observed_at="2026-08-21T13:00:00Z",
        )
        account.set_kill_switch(
            True,
            actor="risk-officer",
            reason="data incident",
            changed_at="2026-08-21T13:30:00Z",
        )
        state = account.state()
        self.assertTrue(state["kill_switch"])
        self.assertEqual(state["orders"][0]["status"], OrderStatus.CANCELED.value)
        with self.assertRaisesRegex(ValueError, "kill switch"):
            account.approve_order(
                proposal.order_id,
                approved_by="portfolio-manager",
                approved_at="2026-08-21T14:00:00Z",
            )

    def test_reconciliation_floor_and_stale_data_alerts_block_trading(self):
        account = self._account()
        result = account.run_day(
            prices={"fixed_income": 100.0, "gold": 10.0, "equity": 10.0},
            valuation_at="2026-08-21T12:00:00Z",
            observed_at="2026-08-23T13:00:00Z",
            observed_units={"fixed_income": 0.0, "gold": 0.0, "equity": 0.0},
        )
        codes = {alert["code"] for alert in result.alerts}
        self.assertTrue({"stale_data", "reconciliation_break", "floor_breach"} <= codes)
        self.assertEqual(result.reconciliation_status, "break")
        self.assertEqual(result.order_status, OrderStatus.CONTROL_BLOCKED.value)

    def test_audit_rows_cannot_be_mutated(self):
        account = self._account()
        with (
            closing(sqlite3.connect(self.path)) as connection,
            self.assertRaisesRegex(sqlite3.IntegrityError, "immutable"),
        ):
            connection.execute("UPDATE audit_events SET actor = 'tamper' WHERE sequence = 1")
        self.assertTrue(account.verify_audit_chain())

    def test_control_profiles_are_separate(self):
        config_path = Path(__file__).parents[1] / "configs" / "execution_controls.toml"
        medium = load_execution_controls(config_path, "medium_term")
        long = load_execution_controls(config_path, "long_term")
        self.assertGreater(medium.max_daily_turnover, long.max_daily_turnover)


if __name__ == "__main__":
    unittest.main()
