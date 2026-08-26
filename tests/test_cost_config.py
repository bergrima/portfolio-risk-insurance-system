import unittest
from pathlib import Path

from portfolio_insurance.backtest import ExecutionModel, calculate_trade_costs
from portfolio_insurance.cost_config import load_asset_transaction_costs

PATH = Path(__file__).parents[1] / "configs" / "transaction_costs.toml"


class TestTransactionCosts(unittest.TestCase):
    def setUp(self) -> None:
        self.execution = ExecutionModel(asset_costs=load_asset_transaction_costs(PATH))

    def test_loads_the_approved_asset_and_side_specific_rates(self) -> None:
        self.assertAlmostEqual(self.execution.transaction_bps("fixed_income", 1), 1.875)
        self.assertAlmostEqual(self.execution.transaction_bps("fixed_income", -1), 1.875)
        self.assertAlmostEqual(self.execution.transaction_bps("equity", 1), 11.6)
        self.assertAlmostEqual(self.execution.transaction_bps("equity", -1), 11.875)
        self.assertAlmostEqual(self.execution.transaction_bps("gold", 1), 12.5)
        self.assertAlmostEqual(self.execution.transaction_bps("gold", -1), 12.5)

    def test_calculates_commission_by_asset_and_trade_side(self) -> None:
        buy_commission, _, _ = calculate_trade_costs(
            {"fixed_income": 100_000, "equity": 100_000, "gold": 100_000},
            self.execution,
        )
        sell_commission, _, _ = calculate_trade_costs(
            {"fixed_income": -100_000, "equity": -100_000, "gold": -100_000},
            self.execution,
        )

        self.assertAlmostEqual(buy_commission, 18.75 + 116 + 125)
        self.assertAlmostEqual(sell_commission, 18.75 + 118.75 + 125)


if __name__ == "__main__":
    unittest.main()
