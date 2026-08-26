import json
import tempfile
import unittest
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import pandas as pd

from portfolio_insurance.data_contract import load_data_contract
from portfolio_insurance.data_quality import validate_market_data
from portfolio_insurance.data_sources import (
    DEFAULT_TSE_SYMBOLS,
    PytseAdapter,
    WallexAdapter,
    fetch_market_data,
    load_snapshot,
    point_in_time_prices,
    point_in_time_tradability,
    save_snapshot,
    valuation_index,
)

ROOT = Path(__file__).parents[1]
CONTRACT = load_data_contract(ROOT / "configs" / "data_contract.toml")


class FakePytse:
    def __init__(self, valuations: pd.DatetimeIndex):
        self.dates = valuations.tz_convert("Asia/Tehran").tz_localize(None).normalize()
        self.call = None

    def download(self, **kwargs):
        self.call = kwargs
        return {
            symbol: pd.DataFrame(
                {"date": self.dates, "adjClose": [100 + index for index in range(len(self.dates))]}
            )
            for symbol in kwargs["symbols"]
        }


class TestDataSources(unittest.TestCase):
    def setUp(self):
        self.valuations = valuation_index("2026-01-03", "2026-01-07", CONTRACT)

    def wallex_payload(self, url):
        query = parse_qs(urlparse(url).query)
        self.assertEqual(query["symbol"], ["USDTTMN"])
        openings = [
            int((timestamp - pd.Timedelta(hours=1)).timestamp()) for timestamp in self.valuations
        ]
        return {"s": "ok", "t": openings, "c": [100_000 + i for i in range(len(openings))]}

    def test_fetches_point_in_time_tse_and_wallex_data(self):
        fake = FakePytse(self.valuations)
        snapshot = fetch_market_data(
            CONTRACT,
            "2026-01-03",
            "2026-01-07",
            pytse=PytseAdapter(fake),
            wallex=WallexAdapter(self.wallex_payload),
        )
        self.assertTrue(fake.call["adjust"])
        self.assertEqual(fake.call["symbols"], list(DEFAULT_TSE_SYMBOLS.values()))
        self.assertEqual(set(snapshot.data.series), set(DEFAULT_TSE_SYMBOLS) | {"usd_irr"})
        usd = snapshot.data[snapshot.data.series == "usd_irr"]
        self.assertEqual(usd.value.iloc[0], 1_000_000)
        self.assertTrue((snapshot.data.available_at <= snapshot.data.valuation_at).all())
        report = validate_market_data(
            snapshot.data,
            CONTRACT,
            "2026-01-03",
            "2026-01-07",
            series_keys=set(snapshot.data.series),
        )
        self.assertTrue(report.passed)
        prices = point_in_time_prices(snapshot.data, CONTRACT, ["fixed_income", "gold", "equity"])
        self.assertEqual(prices.shape, (5, 3))
        tradable = point_in_time_tradability(
            snapshot.data, CONTRACT, ["fixed_income", "gold", "equity"], prices.index
        )
        self.assertTrue(tradable.all().all())

        carried = snapshot.data.copy()
        row = carried[carried.series == "equity_tr"].index[2]
        carried.loc[row, "observed_at"] = carried.loc[row, "valuation_at"] - pd.Timedelta(days=1)
        tradable = point_in_time_tradability(
            carried, CONTRACT, ["fixed_income", "gold", "equity"], prices.index
        )
        self.assertFalse(tradable.loc[prices.index[2], "equity"])

    def test_snapshot_checksum_and_manifest_round_trip(self):
        snapshot = fetch_market_data(
            CONTRACT,
            "2026-01-03",
            "2026-01-07",
            pytse=PytseAdapter(FakePytse(self.valuations)),
            wallex=WallexAdapter(self.wallex_payload),
        )
        with tempfile.TemporaryDirectory() as directory:
            save_snapshot(snapshot, directory)
            loaded = load_snapshot(directory)
            self.assertEqual(loaded.manifest.row_count, len(snapshot.data))
            manifest = json.loads((Path(directory) / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(len(manifest["sha256"]), 64)

    def test_wallex_rejects_failed_payload(self):
        adapter = WallexAdapter(lambda _: {"s": "error"})
        with self.assertRaisesRegex(ValueError, "unsuccessful"):
            adapter.fetch(self.valuations)

    def test_wallex_skips_ranges_before_series_inception(self):
        calls = 0

        def payload(url):
            nonlocal calls
            calls += 1
            if calls == 1:
                return {"s": "no_data"}
            return self.wallex_payload(url)

        valuations = valuation_index("2025-12-01", "2026-01-07", CONTRACT)
        result = WallexAdapter(payload, max_workers=1).fetch(valuations)
        self.assertGreater(calls, 1)
        self.assertFalse(result.empty)
