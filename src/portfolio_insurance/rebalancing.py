from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from math import isclose


@dataclass(frozen=True, slots=True)
class RebalancePolicy:
    absolute_drift_band: float = 0.05

    def __post_init__(self) -> None:
        if not 0 <= self.absolute_drift_band <= 1:
            raise ValueError("absolute_drift_band must be between zero and one")


def needs_rebalance(
    current_weights: Mapping[str, float],
    target_weights: Mapping[str, float],
    policy: RebalancePolicy,
) -> bool:
    """Check drift only; the scheduling layer decides when this check is allowed."""
    if set(current_weights) != set(target_weights):
        raise ValueError("current and target weights must reference the same assets")
    for asset in target_weights:
        drift = abs(current_weights[asset] - target_weights[asset])
        if drift > policy.absolute_drift_band or isclose(
            drift, policy.absolute_drift_band, abs_tol=1e-12
        ):
            return True
    return False
