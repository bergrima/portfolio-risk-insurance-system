# Stage 3 — Scenario Analysis and Monte Carlo

## Purpose

Stage 3 measures how each VPPI policy behaves across many joint market paths. Every path
is passed to the Stage 2 event-driven engine, so review timing, delayed execution, drift
bands, independent insured sleeves, costs, gaps, and floor breaches retain the same
meaning as the historical backtest.

Simulation is a sensitivity analysis, not a forecast or a guarantee. Conclusions are
conditional on the history, model, stress assumptions, and sample length supplied.

## Path generators

The generator samples asset, inflation, and FX return rows jointly. This retains their
observed same-bar dependence instead of simulating each series independently.

For bootstrap paths, the engine samples the asset tradability mask with the same block as
the returns. This preserves market-closure behavior and prevents simulated execution at a
carried, non-executable price. Model-based regime paths assume open markets unless a
separate closure stress is supplied.

- `moving_block_bootstrap` samples contiguous, fixed-length historical blocks. It is the
  default because it retains short-run volatility clustering and is easy to audit.
- `stationary_bootstrap` uses random block lengths with the configured mean block size
  and circular continuation through the source history.
- `regime_switching` classifies observations into lower- and higher-volatility states,
  estimates a smoothed two-state transition matrix, and draws multivariate Student-t
  innovations. The fat tails and persistent states provide a model-based complement to
  resampling.

Each path receives an independent child seed derived from the recorded root seed. The
analysis ID includes hashes of asset, inflation, and FX inputs plus all policy, model,
stress, execution, and historical-window settings.

## Stress scenarios

Named historical windows use realized prices and timestamps. Deterministic shocks can be
layered on both historical and simulated paths without changing any earlier bar.

The standard presets are deliberately transparent:

| Preset | Assumption |
|---|---|
| `exchange_rate_jump` | USD/IRR rises 25%; gold uses beta 1.0 and equity 0.25 |
| `simultaneous_asset_decline` | Gold -15% and equity -25% |
| `reduced_liquidity` | Costs and base slippage triple, 25 bps extra slippage, one extra latency bar |

FX beta shocks are applied multiplicatively to local asset returns. The FX reference is
also shocked so dollar-relative portfolio outcomes remain measurable. Liquidity settings
apply across the full scenario horizon because Stage 2's execution model is path-wide.
Custom API scenarios can combine direct asset shocks, FX jumps and betas, and liquidity
changes in one event.

## Policy surface and intervals

The primary output is grouped by:

`(generator, scenario, multiplier, floor, review frequency, drift band)`

For every group, `surface.csv` reports nominal, inflation-adjusted, and USD-relative
returns and drawdowns, drawdown duration, turnover, total cost, and nominal and
purchasing-power floor shortfalls. Each continuous outcome has a mean, median, and central
empirical interval at the configured confidence level.

Performance drawdown is measured causally from the portfolio's running high-water mark on
every bar. It is not replaced by ZigZag or other threshold-confirmed pivots. When a calibrated
protection gate is supplied, confirmed historical ZigZag legs determine fixed entry/exit
thresholds and the scenario path applies those thresholds causally to its own running
peak/trough state. Each path retains its
maximum drawdown loss and longest underwater duration. Across paths, the surface reports
the configured tail quantile and expected shortfall of the drawdown-loss distribution;
the default tail confidence is 95%.

The nominal floor is the operational floor used by VPPI. A second evaluation floor grows
with the selected inflation benchmark. When `usd_irr` is selected, this measures whether
the portfolio preserved its floor in USD/IRR-relative purchasing-power units. For both
floors the path output records maximum shortfall, conditional shortfall, longest breach
duration, and terminal shortfall. The indexed floor is an evaluation target, not a legal
guarantee or an investable USD reserve.

`floor_breach_probability` is the fraction of paths that ever cross the nominal portfolio
floor; `real_floor_breach_probability` applies the selected inflation-indexed floor. They
are not historical fractions of bars below a floor. Breach and real-loss probabilities
include Wilson score confidence intervals. `path_metrics.csv` retains one row per path and
policy, including each deterministic Stage 2 run ID, for audit and diagnosis.

## Command-line workflow

USD/IRR is the default and required mandate benchmark. The default horizon is 21 valuation
bars per configured portfolio month: 126 bars for medium-term and 252 bars for long-term.
CPI or an annual inflation assumption can still be used for separate diagnostic research.

`--inflation-proxy usd_irr` uses the saved USDT/IRT series (converted and stored as
USD/IRR) as the mandate benchmark. In that mode, `real_return` means
USD-relative return rather than official CPI-adjusted return, and the choice is recorded
in metadata.

```powershell
portfolio-insurance run-scenarios --snapshot data/snapshots/full_history `
  --portfolio medium_term --paths 1000 --workers 4 --seed 20260824 `
  --method moving_block_bootstrap --block-size 10 `
  --confidence 0.90 --tail-confidence 0.95 `
  --multipliers 2,3,4 --floors 0.75,0.8,0.9 `
  --frequencies daily,weekly,biweekly,monthly --drift-bands 0.02,0.05 `
  --slippage-bps 5 --latency-bars 1 `
  --inflation-proxy usd_irr --allow-market-closures --standard-stresses `
  --historical-window drawdown_1:2024-04-01:2024-06-30
```

The output directory contains:

- `surface.csv`: the policy surface and uncertainty intervals;
- `path_metrics.csv`: path-level outcomes and reproducibility identifiers; and
- `metadata.json`: input hashes, root seed, generator, stress definitions, and complete
  run settings.

Historical windows must be covered by the snapshot. A credible research run should use a
history long enough to contain multiple market conditions; the short live-smoke snapshot
is suitable only for software verification.
