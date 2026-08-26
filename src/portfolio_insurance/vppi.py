from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class VppiPolicy:
    multiplier: float
    max_risky_weight: float = 1.0

    def __post_init__(self) -> None:
        if self.multiplier < 0:
            raise ValueError("multiplier must be non-negative")
        if not 0 <= self.max_risky_weight <= 1:
            raise ValueError("max_risky_weight must be between zero and one")


@dataclass(frozen=True, slots=True)
class VppiAllocation:
    nav: float
    floor_value: float
    cushion: float
    risky_value: float
    reserve_value: float

    @property
    def risky_weight(self) -> float:
        return self.risky_value / self.nav


def allocate_vppi(nav: float, floor_value: float, policy: VppiPolicy) -> VppiAllocation:

    if nav <= 0:
        raise ValueError("nav must be positive")
    if floor_value < 0:
        raise ValueError("floor_value must be non-negative")

    cushion = max(nav - floor_value, 0.0)
    risky_cap = nav * policy.max_risky_weight
    risky_value = min(policy.multiplier * cushion, risky_cap)
    reserve_value = nav - risky_value
    return VppiAllocation(
        nav=nav,
        floor_value=floor_value,
        cushion=cushion,
        risky_value=risky_value,
        reserve_value=reserve_value,
    )
