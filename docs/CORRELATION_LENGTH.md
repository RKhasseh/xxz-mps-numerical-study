# Effective MPS correlation length

The two-site MPS transfer matrix has eigenvalues ordered by magnitude as

\[
|\lambda_1| \ge |\lambda_2| \ge \cdots.
\]

The code reports the effective finite-bond-dimension correlation length

\[
\xi(D)=-\frac{2}{\log|\lambda_2/\lambda_1|},
\]

where the factor of two converts the two-site transfer-cell length into lattice sites.

For the `L=64`, `Delta=0.5` production data:

| D | xi(D) [sites] |
|---:|---:|
| 4 | 3.91 |
| 8 | 14.74 |
| 12 | 25.67 |
| 16 | 32.68 |
| 24 | 32.37 |
| 32 | 34.42 |
| 40 | 36.02 |

At `Delta=0.5` the thermodynamic XXZ chain is critical. Therefore the physical correlation length is not expected to approach a finite thermodynamic value. A finite-D MPS has a finite *effective* correlation length because finite bond dimension limits the amount of long-range entanglement it can represent.

Consequently, an increase of `xi(D)` with `D` is expected. The small non-monotonic change between `D=16` and `D=24` is not used as a physical conclusion; these are independently optimized finite-D variational states.

## What this figure is used for

The correlation-length figure is a **transfer-matrix diagnostic** showing how the long-range correlation capacity of the variational state changes with bond dimension.

It is **not** used to claim that `D=32` is converged. The `D=32` working choice is instead assessed using:

1. stability of the variational energy per site against `D=40`, and
2. stability of the sampled conditional Rényi entropies against `D=40` at the available Monte Carlo precision.
