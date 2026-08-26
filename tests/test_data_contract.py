import unittest
from pathlib import Path

from portfolio_insurance.data_contract import SeriesKind, load_data_contract

PATH = Path(__file__).parents[1] / "configs" / "data_contract.toml"


class TestDataContract(unittest.TestCase):
    def test_loads_every_asset_and_macro_series(self) -> None:
        contract = load_data_contract(PATH)
        self.assertEqual(
            {s.asset for s in contract.series.values() if s.investable},
            {"fixed_income", "gold", "equity"},
        )
        self.assertEqual(contract.series["usd_irr"].asset, "usd")
        self.assertEqual(contract.series["cpi"].kind, SeriesKind.INFLATION_INDEX)
        self.assertEqual(contract.series["cpi"].publication_lag_days, 15)
