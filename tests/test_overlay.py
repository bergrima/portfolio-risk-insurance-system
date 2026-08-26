import unittest

from portfolio_insurance.overlay import apply_risk_overlays


class TestRiskOverlay(unittest.TestCase):
    def test_released_risk_weight_moves_to_fixed_income(self) -> None:
        strategic = {"fixed_income": 0.30, "gold": 0.40, "equity": 0.30}

        result = apply_risk_overlays(strategic, {"gold": 0.50, "equity": 0.0})

        self.assertAlmostEqual(result["gold"], 0.20)
        self.assertAlmostEqual(result["equity"], 0.0)
        self.assertAlmostEqual(result["fixed_income"], 0.80)
        self.assertAlmostEqual(sum(result.values()), 1.0)

    def test_assets_without_overlay_keep_strategic_weight(self) -> None:
        strategic = {
            "fixed_income": 0.10,
            "gold": 0.50,
            "equity": 0.40,
        }

        result = apply_risk_overlays(strategic, {"gold": 0.50})

        self.assertAlmostEqual(result["equity"], 0.40)
        self.assertAlmostEqual(result["fixed_income"], 0.35)
