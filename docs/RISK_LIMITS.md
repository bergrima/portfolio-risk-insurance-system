# Risk Limits and Approval

## Project return mandate

A mandate boundary is supplied by the portfolio owner, not estimated by the simulation
engine. USD/IRR is the primary benchmark. The medium-term policy must not underperform it
over 126 valuation bars; the long-term policy must strictly outperform it over 252 bars.
With the configured 10% lower-tail probability, these are 90%-confidence empirical goals.
The engine must return “no feasible policy” when no policy satisfies the relevant goal.

Floor level, drawdown, breach probability, and cost are parameters and outcomes to test.
They are not predeclared acceptance limits. They rank feasible policies on the Pareto and
stability surface after the return mandate has been satisfied.

## Limits currently supported

| Limit | Meaning |
|---|---|
| Minimum real return | Lowest acceptable adverse-context tail return versus the configured benchmark |
| Maximum drawdown | Largest acceptable peak-to-trough loss |
| Maximum floor-breach probability | Largest acceptable share of paths that ever cross the synthetic floor |
| Maximum cost rate | Largest acceptable modeled execution cost as a share of initial NAV |

These boundaries are passed to `select-policy`. A policy must also be Pareto-efficient and
lie on a stable neighboring-parameter plateau.

## Context matters

The current Stage 4 implementation applies hard gates to each policy's worst context,
including deterministic crisis stresses. Consequently, requiring non-negative real return
under every stress can make the entire surface infeasible. That may be the intended
mandate, but it should not be confused with a baseline return objective.

A complete approval should state whether a limit applies to ordinary simulated paths,
named historical closures, deterministic crisis stresses, or all contexts without
exception.

## Current status

The saved 2026-08-24 medium-term candidate predates this mandate and is invalid for policy
approval because its worst-context tail USD-relative return is approximately −27.1%.
Stage 3 and Stage 4 must be rerun with the updated three-asset portfolios, return gates,
asset/side fee schedule, and point-in-time historical fixed-income total returns.
