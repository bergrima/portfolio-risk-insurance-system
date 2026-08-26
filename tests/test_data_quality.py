import unittest
from pathlib import Path

import pandas as pd
from pandas.tseries.offsets import CustomBusinessDay

from portfolio_insurance.data_contract import load_data_contract
from portfolio_insurance.data_quality import validate_market_data

CONTRACT = load_data_contract(Path(__file__).parents[1] / "configs" / "data_contract.toml")


def complete_frame() -> pd.DataFrame:
    dates = pd.date_range(
        "2026-01-03",
        periods=5,
        freq=CustomBusinessDay(weekmask=CONTRACT.policy.weekmask),
        tz="UTC",
    )
    rows = []
    for key, definition in CONTRACT.series.items():
        for index, date in enumerate(dates):
            observed = date - pd.Timedelta(days=definition.publication_lag_days)
            rows.append(
                {
                    "series": key,
                    "valuation_at": date,
                    "observed_at": observed,
                    "available_at": date,
                    "value": 100 + index,
                }
            )
    return pd.DataFrame(rows)


class TestDataQuality(unittest.TestCase):
    def test_complete_point_in_time_data_passes(self) -> None:
        report = validate_market_data(complete_frame(), CONTRACT)
        self.assertTrue(report.passed)
        self.assertTrue(all(value == 1 for value in report.coverage.values()))

    def test_detects_look_ahead_staleness_and_bad_values(self) -> None:
        frame = complete_frame()
        frame.loc[0, "available_at"] = frame.loc[0, "valuation_at"] + pd.Timedelta(days=1)
        frame.loc[1, "observed_at"] = frame.loc[1, "valuation_at"] - pd.Timedelta(days=30)
        frame.loc[2, "value"] = -1
        report = validate_market_data(frame, CONTRACT)
        self.assertFalse(report.passed)
        self.assertTrue(
            {"look_ahead", "stale", "invalid_value"} <= {issue.code for issue in report.issues}
        )
        with self.assertRaisesRegex(ValueError, "non-positive"):
            report.raise_for_errors({"stale", "look_ahead"})

    def test_detects_missing_coverage(self) -> None:
        frame = complete_frame()
        frame = frame.loc[
            ~(
                (frame["series"] == "gold_tr")
                & (frame["valuation_at"] == frame["valuation_at"].max())
            )
        ]
        report = validate_market_data(frame, CONTRACT, "2026-01-03", "2026-01-07")
        self.assertIn("coverage", {issue.code for issue in report.issues})

    def test_coverage_can_begin_at_recorded_series_inception(self) -> None:
        frame = complete_frame()
        first_gold = frame.loc[frame["series"] == "gold_tr", "valuation_at"].min()
        frame = frame.loc[
            ~(
                (frame["series"] == "gold_tr")
                & (frame["valuation_at"] < first_gold + pd.Timedelta(days=2))
            )
        ]
        report = validate_market_data(
            frame,
            CONTRACT,
            "2026-01-03",
            "2026-01-07",
            series_start_dates={"gold_tr": first_gold + pd.Timedelta(days=2)},
        )
        self.assertNotIn("coverage", {issue.code for issue in report.issues})

    def test_selected_contract_series_ignore_a_retired_snapshot_series(self) -> None:
        frame = complete_frame()
        retired = frame.loc[frame["series"] == "gold_tr"].copy()
        retired["series"] = "retired_asset_tr"
        frame = pd.concat([frame, retired], ignore_index=True)

        full_report = validate_market_data(frame, CONTRACT)
        selected_report = validate_market_data(
            frame,
            CONTRACT,
            series_keys={"fixed_income_tr", "gold_tr", "equity_tr", "usd_irr"},
        )

        self.assertIn("unknown_series", {issue.code for issue in full_report.issues})
        self.assertTrue(selected_report.passed)
