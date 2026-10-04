# L=64 production results

## Bond-dimension convergence

The production sweep used `D = 4, 8, 12, 16, 24, 32, 40` at `Delta=0.5`.

The key numerical comparison is

\[
|E_{D=32}/L-E_{D=40}/L|=4.14207\times10^{-6}.
\]

This is below the chosen `1e-5` convergence tolerance.

For 1000 Born trajectories per D, the D=32 versus D=40 mean Rényi-entropy differences are:

| alpha | mean at D=32 | mean at D=40 | absolute difference |
|---:|---:|---:|---:|
| 2.0 | 0.243127 | 0.232758 | 0.010369 |
| 2.5 | 0.225523 | 0.214997 | 0.010525 |
| 3.0 | 0.213986 | 0.203474 | 0.010512 |
| 4.0 | 0.199776 | 0.189428 | 0.010349 |

At the available Monte Carlo statistics these shifts are about 1.1 combined standard errors, so they are not resolved at the two-standard-error level used by the convergence report.

## Interpretation

D=32 is a reasonable working bond dimension for this data set because:

1. the variational energy has nearly saturated by D=32,
2. the AB/BA bond-energy difference is already small and continues to decrease,
3. the measured Rényi averages at D=32 are statistically compatible with D=40 at the chosen resolution.

The statement is intentionally numerical: it does not claim mathematically exact convergence. For a stricter study, increase the number of optimization steps and Born trajectories and add D=48.
## Effective MPS correlation length

The transfer-matrix correlation length changes from approximately `34.42` sites at `D=32` to `36.02` sites at `D=40`. This quantity is **not converged**, and no claim of correlation-length saturation is made.

That behavior is consistent with a finite-bond-dimension approximation to the critical XXZ chain at `Delta=0.5`: increasing `D` allows the MPS to represent correlations over longer distances, so the effective `xi(D)` can continue to grow even when local observables such as the energy are already very stable.

Accordingly, the repository uses the energy and the sampled Rényi observables to assess whether `D=32` is an adequate working bond dimension. The correlation-length plot is retained only as a transfer-matrix diagnostic.

