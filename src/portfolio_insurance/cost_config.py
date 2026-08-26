from __future__ import annotations

import tomllib
from pathlib import Path

from .backtest import AssetTransactionCost


def load_asset_transaction_costs(path: str | Path) -> tuple[AssetTransactionCost, ...]:
    """Load human-readable percentage fees and convert them to basis points."""
    with Path(path).open("rb") as handle:
        raw = tomllib.load(handle)
    assets = raw.get("assets", {})
    if not assets:
        raise ValueError("transaction-cost config must define at least one asset")
    return tuple(
        AssetTransactionCost(
            asset=asset,
            buy_bps=float(values["buy_percent"]) * 100,
            sell_bps=float(values["sell_percent"]) * 100,
        )
        for asset, values in sorted(assets.items())
    )
