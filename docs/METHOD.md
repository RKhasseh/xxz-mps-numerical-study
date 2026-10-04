# Method

## 1. Variational state

The state is a two-site periodic MPS,

\[
|\psi(A,B)\rangle=\sum_{s_1\ldots s_L}\mathrm{Tr}\left(A^{s_1}B^{s_2}\cdots A^{s_{L-1}}B^{s_L}\right)|s_1\ldots s_L\rangle.
\]

The virtual tensors obey the U(1) selection rule associated with conservation of total `S^z`. The bond dimension `D` controls the size of the variational MPS space.

## 2. Ground-state search

For every bond dimension, the code minimizes

\[
E(A,B)=\frac{\langle\psi(A,B)|H|\psi(A,B)\rangle}{\langle\psi(A,B)|\psi(A,B)\rangle}
\]

using automatic differentiation, Adam, and then L-BFGS. Smaller-D states are embedded into the larger-D charge spaces to provide nested warm starts.

## 3. Transfer matrix diagnostics

The code reports

- energy per site,
- the difference between `AB` and `BA` bond energies,
- gradient norm after optimization,
- effective finite-D MPS correlation length,
- transfer-sector norm discarded during Born sampling.

The effective correlation length is extracted from the two leading eigenvalue magnitudes of the two-site transfer matrix,

\[
\xi(D)=-\frac{2}{\log |\lambda_2/\lambda_1|}.
\]

At `Delta=0.5` the XXZ chain is critical in the thermodynamic limit. Therefore `xi(D)` is treated as a **finite-entanglement diagnostic** that can grow with `D`, not as a quantity that is expected to converge to a finite value. It is not part of the `D=32` pass/fail convergence test.

## 4. Measurement protocol

The ring is split as

\[
A_1-B_1-A_2-B_2,
\]

with lengths `(L/8,L/4,3L/8,L/4)`. `B1` and `B2` are measured in the `S^z` basis by sequential Born sampling.

## 5. Conditional entanglement

For each measurement record, the projected state on `A1 union A2` is represented through the remaining tensor network. The squared Schmidt values `p_i` across `A1|A2` are obtained from symmetry-resolved Gram matrices. The code then computes

\[
S_\alpha=\frac{1}{1-\alpha}\log\sum_i p_i^\alpha.
\]

## 6. Magnetization imbalance

With spin up encoded as `+1` and spin down as `-1`, define

\[
M_{B_r}=\sum_{j\in B_r}\sigma_j,
\qquad
m_{B_r}=M_{B_r}/|B_r|,
\]

and

\[
\Delta m=m_{B_1}-m_{B_2}.
\]

This is a direct observable of each sampled lattice measurement record. The `S_alpha` versus `Delta m` plots therefore contain only numerical MPS/XXZ data.
