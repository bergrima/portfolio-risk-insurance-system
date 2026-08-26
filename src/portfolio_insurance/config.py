from __future__ import annotations

import tomllib
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

from .domain import AssetDefinition, PortfolioDefinition


@dataclass(frozen=True, slots=True)
class ProjectConfig:
    assets: Mapping[str, AssetDefinition]
    portfolios: Mapping[str, PortfolioDefinition]


def load_project_config(path: str | Path) -> ProjectConfig:
    config_path = Path(path)
    with config_path.open("rb") as handle:
        raw = tomllib.load(handle)

    assets = {
        key: AssetDefinition(
            key=key,
            label=value["label"],
            risky=value["risky"],
            investable=value.get("investable", True),
            daily_return=value.get("daily_return"),
            calendar_day_accrual=value.get("calendar_day_accrual", False),
        )
        for key, value in raw["assets"].items()
    }
    portfolios = {
        key: PortfolioDefinition(
            key=key,
            label=value["label"],
            horizon_months=value["horizon_months"],
            weights=value["weights"],
        )
        for key, value in raw["portfolios"].items()
    }

    unknown_assets = {
        asset
        for portfolio in portfolios.values()
        for asset in portfolio.weights
        if asset not in assets
    }
    if unknown_assets:
        raise ValueError(f"unknown assets referenced by portfolios: {sorted(unknown_assets)}")
    non_investable_assets = {
        asset
        for portfolio in portfolios.values()
        for asset in portfolio.weights
        if not assets[asset].investable
    }
    if non_investable_assets:
        raise ValueError(
            f"non-investable assets referenced by portfolios: {sorted(non_investable_assets)}"
        )

    return ProjectConfig(assets=MappingProxyType(assets), portfolios=MappingProxyType(portfolios))
