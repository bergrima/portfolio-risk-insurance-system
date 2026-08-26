from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from math import isclose
from types import MappingProxyType


@dataclass(frozen=True, slots=True)
class AssetDefinition:
    key: str
    label: str
    risky: bool
    investable: bool = True
    daily_return: float | None = None
    calendar_day_accrual: bool = False

    def __post_init__(self) -> None:
        if not self.key.strip():
            raise ValueError("asset key cannot be empty")
        if self.daily_return is not None and self.daily_return <= -1:
            raise ValueError("daily asset return must be greater than -100%")
        if self.calendar_day_accrual and self.daily_return is None:
            raise ValueError("calendar-day accrual requires a configured daily return")


@dataclass(frozen=True, slots=True)
class PortfolioDefinition:
    key: str
    label: str
    horizon_months: int
    weights: Mapping[str, float]

    def __post_init__(self) -> None:
        if self.horizon_months <= 0:
            raise ValueError("horizon_months must be positive")
        if not self.weights:
            raise ValueError("portfolio must contain at least one asset")
        if any(weight < 0 or weight > 1 for weight in self.weights.values()):
            raise ValueError("portfolio weights must be between zero and one")
        if not isclose(sum(self.weights.values()), 1.0, abs_tol=1e-10):
            raise ValueError("portfolio weights must sum to one")
        object.__setattr__(self, "weights", MappingProxyType(dict(self.weights)))
