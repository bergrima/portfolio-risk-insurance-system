import unittest

from portfolio_insurance.vppi import VppiPolicy, allocate_vppi


class TestVppi(unittest.TestCase):
    def test_uses_multiplier_on_cushion(self) -> None:
        allocation = allocate_vppi(nav=100.0, floor_value=80.0, policy=VppiPolicy(multiplier=3.0))
        self.assertAlmostEqual(allocation.cushion, 20.0)
        self.assertAlmostEqual(allocation.risky_value, 60.0)
        self.assertAlmostEqual(allocation.reserve_value, 40.0)

    def test_respects_risky_cap(self) -> None:
        allocation = allocate_vppi(
            nav=100.0,
            floor_value=50.0,
            policy=VppiPolicy(multiplier=4.0, max_risky_weight=0.70),
        )
        self.assertAlmostEqual(allocation.risky_value, 70.0)
        self.assertAlmostEqual(allocation.reserve_value, 30.0)

    def test_moves_fully_to_reserve_at_or_below_floor(self) -> None:
        allocation = allocate_vppi(nav=75.0, floor_value=80.0, policy=VppiPolicy(multiplier=3.0))
        self.assertEqual(allocation.cushion, 0.0)
        self.assertEqual(allocation.risky_value, 0.0)
        self.assertEqual(allocation.reserve_value, 75.0)
