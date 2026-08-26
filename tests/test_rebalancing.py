import unittest

from portfolio_insurance.rebalancing import RebalancePolicy, needs_rebalance


class TestRebalancing(unittest.TestCase):
    def test_triggers_at_band_boundary(self) -> None:
        target = {"gold": 0.40, "equity": 0.30, "fixed_income": 0.30}
        current = {"gold": 0.45, "equity": 0.27, "fixed_income": 0.28}
        self.assertTrue(needs_rebalance(current, target, RebalancePolicy(0.05)))

    def test_does_not_trigger_inside_band(self) -> None:
        target = {"gold": 0.40, "equity": 0.30, "fixed_income": 0.30}
        current = {"gold": 0.43, "equity": 0.28, "fixed_income": 0.29}
        self.assertFalse(needs_rebalance(current, target, RebalancePolicy(0.05)))
