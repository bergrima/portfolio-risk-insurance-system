from __future__ import annotations

import tomllib
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from types import MappingProxyType


class SeriesKind(StrEnum):
    TOTAL_RETURN = "total_return"
    NAV = "nav"
    EXCHANGE_RATE = "exchange_rate"
    INFLATION_INDEX = "inflation_index"


class MissingDataPolicy(StrEnum):
    FAIL = "fail"
    PREVIOUS = "previous"


@dataclass(frozen=True, slots=True)
class SeriesContract:
    key: str
    asset: str | None
    instrument: str
    kind: SeriesKind
    currency: str
    source: str
    max_stale_days: int
    publication_lag_days: int = 0
    investable: bool = True

    def __post_init__(self) -> None:
        if not self.key.strip() or not self.instrument.strip() or not self.source.strip():
            raise ValueError("series key, instrument, and source cannot be empty")
        if len(self.currency) != 3 or self.currency.upper() != self.currency:
            raise ValueError("currency must be an uppercase ISO-style three-letter code")
        if self.max_stale_days < 0 or self.publication_lag_days < 0:
            raise ValueError("staleness and publication lag cannot be negative")


@dataclass(frozen=True, slots=True)
class DataPolicy:
    timezone: str
    calendar: str
    weekmask: str
    valuation_time: str
    missing_data: MissingDataPolicy
    minimum_coverage: float

    def __post_init__(self) -> None:
        if not 0 < self.minimum_coverage <= 1:
            raise ValueError("minimum_coverage must be in (0, 1]")


@dataclass(frozen=True, slots=True)
class DataContract:
    policy: DataPolicy
    series: Mapping[str, SeriesContract]


def load_data_contract(path: str | Path) -> DataContract:
    with Path(path).open("rb") as handle:
        raw = tomllib.load(handle)
    policy = DataPolicy(
        timezone=raw["policy"]["timezone"],
        calendar=raw["policy"]["calendar"],
        weekmask=raw["policy"].get("weekmask", "Mon Tue Wed Thu Fri"),
        valuation_time=raw["policy"]["valuation_time"],
        missing_data=MissingDataPolicy(raw["policy"]["missing_data"]),
        minimum_coverage=raw["policy"]["minimum_coverage"],
    )
    series = {
        key: SeriesContract(
            key=key,
            asset=value.get("asset"),
            instrument=value["instrument"],
            kind=SeriesKind(value["kind"]),
            currency=value["currency"],
            source=value["source"],
            max_stale_days=value["max_stale_days"],
            publication_lag_days=value.get("publication_lag_days", 0),
            investable=value.get("investable", True),
        )
        for key, value in raw["series"].items()
    }
    assets = [item.asset for item in series.values() if item.investable]
    duplicates = {asset for asset in assets if assets.count(asset) > 1}
    if duplicates:
        raise ValueError(f"multiple investable series for assets: {sorted(duplicates)}")
    return DataContract(policy=policy, series=MappingProxyType(series))
