# Stage 4 — Policy Selection

## Purpose

Stage 4 converts the Stage 3 policy surface into a decision that is explicit about
trade-offs and refuses to manufacture certainty. It selects a policy only when all three
conditions hold:

1. the policy is not dominated on robust real return, drawdown, floor-breach probability,
   and cost;
2. any supplied mandate limits are satisfied; and
3. immediately adjacent parameter choices also behave well enough to establish a
   stability plateau.

If the grid is too sparse, the limits are infeasible, or no Pareto policy lies on a
plateau, the result records that status and leaves `selected_policy` empty. It does not
fall back to an in-sample maximum.

## Evidence and Pareto objectives

The selection input is the path-level `path_metrics.csv` from Stage 3. Results are first
grouped by policy, generator, and scenario. Within each context, Stage 4 calculates:

- lower-tail real return at the configured tail probability;
- lower-tail maximum drawdown expressed as a positive loss;
- the fraction of paths that ever breach the floor; and
- median cost as a fraction of initial NAV.

The policy-level objective is deliberately adverse-context aware: real return is the
worst context's tail return, while drawdown, breach probability, and cost are the worst
context values. A policy is Pareto-efficient when no other policy is at least as good on
all four objectives and strictly better on one.

The objective weights in `configs/policy_selection.toml` rank policies only after the
Pareto and mandate tests. The medium-term profile requires its 10th-percentile six-month
USD-relative return to be at least zero. The long-term profile requires its
10th-percentile 12-month USD-relative return to be strictly positive. The medium-term
profile places more weight on floor protection; the long-term profile places more weight
on dollar-relative growth. Drawdown, breach probability, and cost remain Pareto objectives
without invented hard limits.

The return mandates apply to the worst evaluated generator/scenario context. If no policy
satisfies them, Stage 4 returns `no_feasible_policy` instead of selecting the least-bad
candidate. See `docs/RISK_LIMITS.md`.

## Stability plateau

Parameter values are ordered on the evaluated grid. An immediate neighbor changes
exactly one of multiplier, floor, review frequency, or drift band by one available grid
step. Stage 4 normalizes the four objectives, computes a weighted utility, and records:

- neighbor count;
- mean, worst, and dispersion of neighbor utility;
- the share of neighbors that satisfy mandate limits; and
- the utility drop to the worst neighbor.

A plateau must meet the configured minimum neighbor count, feasible-neighbor fraction,
and maximum utility gap. The selected policy is the highest plateau score among feasible
Pareto policies. `candidate_policies.csv` retains every objective, dominance, constraint,
neighbor, and score field so the decision can be challenged or reproduced.

## Walk-forward validation

Walk-forward folds use the realized point-in-time snapshot, not simulated future data.
For each expanding fold, the engine:

1. divides all available training history into complete portfolio-horizon windows;
2. selects a policy using only those training windows;
3. runs every policy over the immediately later, unseen test window; and
4. marks which test result belonged to the policy selected during training.

The shared price at the train/test boundary only establishes the initial level; the first
scored return occurs strictly after the training cutoff. OOS results never alter the final
Stage 3 selection. They validate the repeated selection process and remain available for
neighbor sensitivity diagnostics.

The sequence of training-selected policies must satisfy the configured dollar-relative
return mandate on every later fold to receive `passed`. Missing fold selections, too few
folds, and failed limits remain explicit statuses.

## Command-line workflow

Run a sufficiently broad Stage 3 grid first. A one-policy grid cannot establish a
stability plateau.

```powershell
portfolio-insurance select-policy `
  --analysis runs/stage3/medium_term `
  --snapshot data/snapshots/full_history `
  --portfolio medium_term --folds 3 `
  --inflation-proxy usd_irr --allow-market-closures
```

Run the long-term portfolio separately against its own Stage 3 export. Its default test
horizon is 252 valuation returns; medium-term defaults to 126. `--test-bars` and
`--minimum-train-bars` make the fold design explicit when the available history requires
a different research design.

The export contains:

- `candidate_policies.csv`: full Pareto, mandate, utility, and plateau decision table;
- `context_metrics.csv`: generator/scenario evidence before worst-context aggregation;
- `selected_policy.json`: selected parameters or an explicit non-selection status;
- `walk_forward_folds.csv`: training cutoffs and later outcomes for each selected policy;
- `walk_forward_policy_metrics.csv`: every policy's path-level OOS result;
- `walk_forward_candidates.csv`: OOS sensitivity diagnostics across policies; and
- `metadata.json`: criteria, hashes, execution assumptions, fold design, and deterministic
  selection ID.

Simulation, historical backtesting, and walk-forward validation remain research evidence.
None creates a legal or absolute floor guarantee.
