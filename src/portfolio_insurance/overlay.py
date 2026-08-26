from __future__ import annotations

from collections.abc import Mapping
from math import isclose


def apply_risk_overlays(
    strategic_weights: Mapping[str, float],
    exposure_factors: Mapping[str, float],
    reserve_asset: str = "fixed_income",
) -> dict[str, float]:
    """Apply sleeve-level risk exposure factors and move released weight to reserve.

    An exposure factor is the fraction of a strategic risky sleeve that remains invested.
    For example, factor 0.5 on a 40% gold sleeve produces 20% gold and releases 20%
    to fixed income. Assets without an explicit factor retain their strategic weight.
    """
    if reserve_asset not in strategic_weights:
        raise ValueError("reserve asset must exist in strategic weights")
    unknown = set(exposure_factors) - set(strategic_weights)
    if unknown:
        raise ValueError(f"unknown assets in exposure factors: {sorted(unknown)}")
    if reserve_asset in exposure_factors:
        raise ValueError("reserve asset cannot receive a risk exposure factor")
    if any(factor < 0 or factor > 1 for factor in exposure_factors.values()):
        raise ValueError("exposure factors must be between zero and one")

    result = dict(strategic_weights)
    released_weight = 0.0
    for asset, factor in exposure_factors.items():
        insured_weight = strategic_weights[asset] * factor
        released_weight += strategic_weights[asset] - insured_weight
        result[asset] = insured_weight
    result[reserve_asset] += released_weight

    if not isclose(sum(result.values()), 1.0, abs_tol=1e-12):
        raise ValueError("overlay result must sum to one")
    return result
