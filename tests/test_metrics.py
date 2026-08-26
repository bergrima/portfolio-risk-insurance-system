import unittest

import pandas as pd

from portfolio_insurance.metrics import (
    drawdown_series,
    drawdown_series_from_levels,
    expected_shortfall,
    maximum_drawdown,
    maximum_drawdown_from_levels,
    maximum_underwater_duration,
    relative_returns,
)


class TestMetrics(unittest.TestCase):
    def test_drawdown_tracks_decline_from_running_peak(self) -> None:
        returns = pd.Series([0.10, -0.20, 0.25])
        drawdowns = drawdown_series(returns)
        self.assertAlmostEqual(drawdowns.iloc[0], 0.0)
        self.assertAlmostEqual(drawdowns.iloc[1], -0.20)
        self.assertAlmostEqual(drawdowns.iloc[2], 0.0)
        self.assertAlmostEqual(maximum_drawdown(returns), -0.20)

    def test_expected_shortfall_averages_tail_observations(self) -> None:
        returns = pd.Series([-0.10, -0.05, 0.00, 0.03, 0.04])
        self.assertAlmostEqual(expected_shortfall(returns, confidence=0.80), -0.10)

    def test_level_drawdown_preserves_every_underwater_bar(self) -> None:
        levels = pd.Series([100.0, 120.0, 90.0, 90.0, 120.0, 110.0])
        drawdowns = drawdown_series_from_levels(levels)

        self.assertAlmostEqual(drawdowns.iloc[2], -0.25)
        self.assertAlmostEqual(maximum_drawdown_from_levels(levels), -0.25)
        self.assertEqual(maximum_underwater_duration(drawdowns), 2)

    def test_relative_returns_use_geometric_conversion(self) -> None:
        local_returns = pd.Series([0.20, 0.05])
        usd_irr_returns = pd.Series([0.10, 0.05])

        result = relative_returns(local_returns, usd_irr_returns)

        self.assertAlmostEqual(result.iloc[0], (1.20 / 1.10) - 1.0)
        self.assertAlmostEqual(result.iloc[1], 0.0)
