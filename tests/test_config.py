import unittest
from pathlib import Path

from portfolio_insurance.config import load_project_config

CONFIG_PATH = Path(__file__).parents[1] / "configs" / "portfolios.toml"


class TestConfig(unittest.TestCase):
    def test_contains_expected_strategic_weights(self) -> None:
        config = load_project_config(CONFIG_PATH)
        self.assertEqual(set(config.portfolios), {"medium_term", "long_term"})
        self.assertEqual(
            dict(config.portfolios["medium_term"].weights),
            {"fixed_income": 0.30, "gold": 0.40, "equity": 0.30},
        )
        self.assertEqual(
            dict(config.portfolios["long_term"].weights),
            {"fixed_income": 0.10, "gold": 0.50, "equity": 0.40},
        )
        self.assertEqual(config.portfolios["long_term"].horizon_months, 12)
        self.assertIsNone(config.assets["fixed_income"].daily_return)
        self.assertFalse(config.assets["fixed_income"].calendar_day_accrual)

    def test_usd_is_a_non_investable_reference_asset(self) -> None:
        config = load_project_config(CONFIG_PATH)
        self.assertIn("usd", config.assets)
        self.assertFalse(config.assets["usd"].investable)
        self.assertTrue(
            all("usd" not in portfolio.weights for portfolio in config.portfolios.values())
        )
