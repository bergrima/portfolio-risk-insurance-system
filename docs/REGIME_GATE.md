# Drawdown / confirmed-ZigZag protection gate

## Financial purpose

The strategic allocation is the portfolio's starting and normal-state allocation. VPPI is
not applied at inception. It becomes active for an insured asset only after that asset has
fallen far enough from its causal running peak to enter a protected state. The asset returns
to its full strategic weight only after a sufficiently large recovery from the protected-state
trough.

This is a risk-state filter, not a price forecast. Its intended benefit is to avoid the old
behavior in which VPPI reduced gold and equity immediately, before any adverse evidence was
observed.

## Calibration contract

Calibration and evaluation are strictly separated. The current reproducible calibration uses
all available observations before the common portfolio inception:

| Asset | Training start | Training end | Observations | Completed down legs | Completed up legs |
|---|---:|---:|---:|---:|---:|
| Equity | 2015-11-01 | 2022-08-28 | 1,781 | 52 | 52 |
| Gold | 2017-06-10 | 2022-08-28 | 1,362 | 49 | 50 |

The portfolio evaluation starts on 2022-08-29. No observation on or after that boundary is
used to choose the gate thresholds.

The calibration procedure is:

1. identify 5% ZigZag reversals, recording both the historical extreme date and the later
   confirmation date;
2. construct only completed, confirmed peak-to-trough and trough-to-peak legs;
3. for each candidate trigger, estimate the probability that the leg extends at least a
   further 2%, conditional on already reaching the trigger;
4. require at least eight completed legs, an empirical continuation probability of at least
   60%, and a 90% Wilson lower bound of at least 50%; and
5. select the earliest supported trigger, separately for down and up legs.

The pre-evaluation result is:

| Asset | Protection entry | Down legs continuing another 2% | Protection exit | Up legs continuing another 2% |
|---|---:|---:|---:|---:|
| Equity | 5% drawdown | 44 / 52 (84.6%) | 5% recovery | 39 / 52 (75.0%) |
| Gold | 5% drawdown | 39 / 49 (79.6%) | 5% recovery | 45 / 50 (90.0%) |

These frequencies describe the training sample. They are not probabilities guaranteed to
persist in later markets.

## Causal execution rule

Each insured asset has an independent state:

- **Normal:** hold the configured strategic allocation and apply only the ordinary calendar /
  drift-band rebalance rule.
- **Enter protection:** when the observed close is at least the calibrated percentage below
  the running peak, activate that asset's VPPI sleeve.
- **Protected:** update the trough and the VPPI target; released exposure goes to the actual
  historical fixed-income series.
- **Exit protection:** when the close has recovered by the calibrated percentage from the
  protected-state trough, restore the full strategic allocation.

A state change creates an immediate review even between calendar review dates. The order is
still queued for a later valuation bar and waits for closed assets to become tradable. The
ledger records the signal, peak, trough, drawdown, recovery, state, transition, decision, and
later execution separately.

The ordinary drift band and the protection drift band can be configured separately. The
ordinary band applies only while every insured sleeve is normal. A state-change review, or a
scheduled review while any sleeve is protected, uses the protection band. If no protection
band is supplied, the ordinary band is used for both states for backward compatibility.

## Historical comparison

The following local-IRR results use the full common history from 2022-08-29 through
2026-08-24, historical adjusted total-return data for fixed income, configured transaction
fees, monthly review, 5% drift band, multiplier 3, floor 80%, and one-bar latency.

| Portfolio / policy | Total return | Annualized volatility | Maximum drawdown | Turnover | Cost |
|---|---:|---:|---:|---:|---:|
| Medium buy and hold | 815.81% | 22.24% | -19.93% | 0.00x | 0 |
| Medium calendar rebalance | 679.01% | 16.03% | -13.79% | 1.39x | 4,197 |
| Medium always-on VPPI | 647.81% | 15.75% | -14.24% | 2.20x | 4,740 |
| Medium gated VPPI | 655.79% | 15.88% | -13.93% | 2.13x | 4,121 |
| Long buy and hold | 975.33% | 25.12% | -22.15% | 0.00x | 0 |
| Long calendar rebalance | 854.85% | 20.13% | -19.82% | 1.14x | 5,066 |
| Long always-on VPPI | 829.59% | 19.73% | -19.09% | 2.64x | 6,503 |
| Long gated VPPI | 866.17% | 19.83% | -19.08% | 2.83x | 7,148 |

The gate improves the always-on VPPI return by 7.97 percentage points in the medium portfolio
and 36.59 percentage points in the long portfolio. It also preserves the user's specified
initial weights. It does not dominate buy-and-hold: the reduction in drawdown is purchased by
giving up part of the strong gold/equity upside. In the medium portfolio it also remains below
the calendar-rebalanced baseline.

## Interpretation and remaining validation

Performance drawdown remains the ordinary causal high-water-mark drawdown. ZigZag is used only
to calibrate and audit the regime gate; it does not replace the portfolio drawdown metric.

The gate is integrated into historical backtests, scenario paths, and walk-forward policy
selection. Paper initialization deliberately refuses a gated selection until the paper
database and operator workflow persist and audit the peak/trough state. Required work before
paper use includes walk-forward recalibration tests, threshold-neighborhood stability, stress
and gap analysis, and paper observation of false exits and re-entries.
