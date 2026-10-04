#!/usr/bin/env python3
"""
Numerical finite-D study of the periodic spin-1/2 XXZ chain with a two-site,
U(1)-symmetric MPS.

This file is intentionally CFT-free.  It does two numerical jobs:

1. Bond-dimension convergence of the variational MPS ground state.
2. Optionally, Born sampling on B1,B2 followed by the conditional Schmidt
   spectrum / Renyi entropies of A1|A2.

The default D sequence is nested and preserves the original D=32 charge
multiplicities exactly.  D=40 is included as a validation point so that D=32
can be tested against a larger variational space rather than declared converged
only because it is the largest D.

Examples
--------
Fast ground-state convergence study at L=64:
    python src/xxz_mps_study.py --L 64 --records 0

Stronger convergence study including the target numerical Renyi observables:
    python src/xxz_mps_study.py --L 64 --Ds 24 32 40 --records 1000

Outputs are written below OUTPUT_ROOT (or --output):
    bond_dimension_summary.csv
    convergence_report.json
    plots/convergence/energy_per_site_vs_D.png
    plots/convergence/energy_difference_vs_D.png
    plots/diagnostics/effective_correlation_length_vs_D.png
    D_<D>/optimized_mps.npz
    D_<D>/metadata.json
and, if --records > 0:
    renyi_summary.csv
    plots/convergence/renyi_mean_vs_D.png
    plots/convergence/renyi_difference_vs_reference.png
    D_<D>/trajectory_results.csv
    D_<D>/renyi_summary.csv
    delta_m_summary.csv
    plots/delta_m/entanglement_vs_delta_m_D<D>_alpha<A>.png

No exact diagonalization and no CFT formulas are used.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
import argparse
import json
import math
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch


# =============================================================================
# DEFAULT SETTINGS
# =============================================================================
DELTA = 0.50
DEFAULT_L = 64
DEFAULT_DS = (4, 8, 12, 16, 24, 32, 40)
DEFAULT_CANDIDATE_D = 32
ALPHAS = (2.0, 2.5, 3.0, 4.0)
DEFAULT_RECORDS = 0          # use 1000 for production observable convergence
RNG_SEED = 20260925
OPT_SEED = 17
ADAM_STEPS = 40
ADAM_LR = 1.0e-3
LBFGS_STEPS = 60
SCHMIDT_TOL = 1.0e-14
SAMPLING_SECTOR_TOL = 1.0e-8
WARM_START_NOISE = 1.0e-5
OUTPUT_ROOT = Path(__file__).resolve().parents[1] / "output"

# These are user-chosen numerical convergence criteria, not mathematical truths.
ENERGY_TOL_PER_SITE = 1.0e-5
ENTROPY_ABS_TOL = 5.0e-3
ENTROPY_SIGMA_FACTOR = 2.0

torch.set_default_dtype(torch.float64)
torch.set_num_threads(1)


# =============================================================================
# NESTED U(1) VIRTUAL-CHARGE LAYOUTS
# =============================================================================
# D=32 exactly matches the uploaded calculation.  Smaller layouts are strict
# subspaces; D=40 extends it.  Multiplicities are symmetric under q -> -q.
EVEN_MULTIPLICITIES = {
    4:  {0: 2,  2: 1},
    8:  {0: 4,  2: 2},
    12: {0: 4,  2: 3, 4: 1},
    16: {0: 6,  2: 4, 4: 1},
    24: {0: 8,  2: 5, 4: 2, 6: 1},
    32: {0: 10, 2: 7, 4: 3, 6: 1},
    40: {0: 12, 2: 9, 4: 4, 6: 1},
    48: {0: 14, 2: 11, 4: 5, 6: 1},
}
ODD_MULTIPLICITIES = {
    4:  {1: 2},
    8:  {1: 3, 3: 1},
    12: {1: 5, 3: 1},
    16: {1: 6, 3: 2},
    24: {1: 8, 3: 3, 5: 1},
    32: {1: 11, 3: 4, 5: 1},
    40: {1: 14, 3: 5, 5: 1},
    48: {1: 17, 3: 6, 5: 1},
}


def symmetric_charge_list(mult: dict[int, int], D: int) -> np.ndarray:
    vals: list[int] = []
    for q in sorted(mult):
        n = int(mult[q])
        if q == 0:
            vals.extend([0] * n)
        else:
            vals.extend([-q] * n)
            vals.extend([+q] * n)
    vals = sorted(vals)
    if len(vals) != D:
        raise ValueError(f"Charge multiplicities produce {len(vals)} states, expected D={D}.")
    return np.asarray(vals, dtype=np.int64)


def charge_layout(D: int) -> tuple[np.ndarray, np.ndarray]:
    if D not in EVEN_MULTIPLICITIES or D not in ODD_MULTIPLICITIES:
        allowed = sorted(set(EVEN_MULTIPLICITIES) & set(ODD_MULTIPLICITIES))
        raise ValueError(f"No charge layout for D={D}. Available: {allowed}")
    return (
        symmetric_charge_list(EVEN_MULTIPLICITIES[D], D),
        symmetric_charge_list(ODD_MULTIPLICITIES[D], D),
    )


# =============================================================================
# DATA STRUCTURES
# =============================================================================
@dataclass
class U1RectSpec:
    q_left: np.ndarray
    q_right: np.ndarray
    flat_indices: torch.Tensor
    n_parameters: int
    D: int


@dataclass
class MPSContext:
    D: int
    q_even: np.ndarray
    q_odd: np.ndarray
    groups_even_np: dict[int, np.ndarray]
    groups_odd_np: dict[int, np.ndarray]
    groups_even_t: dict[int, torch.Tensor]
    groups_odd_t: dict[int, torch.Tensor]
    common_diffs: tuple[int, ...]


@dataclass
class TwoSiteResult:
    energy: float
    A: np.ndarray
    B: np.ndarray
    theta_A: np.ndarray
    theta_B: np.ndarray
    q_even: np.ndarray
    q_odd: np.ndarray
    n_parameters_A: int
    n_parameters_B: int
    e_AB: float
    e_BA: float
    grad_norm: float


@dataclass
class TransferCache:
    A: np.ndarray
    B: np.ndarray
    scale_A: float
    scale_B: float
    E_full: tuple[np.ndarray, np.ndarray]
    E_blocks: tuple[dict[int, np.ndarray], dict[int, np.ndarray]]
    Es_blocks: tuple[tuple[dict[int, np.ndarray], dict[int, np.ndarray]],
                     tuple[dict[int, np.ndarray], dict[int, np.ndarray]]]
    suffix_blocks: list[dict[int, np.ndarray]]
    retained_diffs: tuple[int, ...]
    sector_norm_weights: dict[int, float]
    discarded_norm_fraction: float
    norm_retained: float
    xi_sites: float


# =============================================================================
# U(1) TENSORS AND BLOCK INDEXING
# =============================================================================
def pair_difference_groups(charges: np.ndarray) -> dict[int, np.ndarray]:
    D = len(charges)
    groups: dict[int, list[int]] = defaultdict(list)
    for i in range(D):
        for I in range(D):
            groups[int(charges[i] - charges[I])].append(i * D + I)
    return {q: np.asarray(v, dtype=np.int64) for q, v in groups.items()}


def make_context(D: int) -> MPSContext:
    q_even, q_odd = charge_layout(D)
    ge = pair_difference_groups(q_even)
    go = pair_difference_groups(q_odd)
    common = tuple(sorted(set(ge) & set(go)))
    return MPSContext(
        D=D,
        q_even=q_even,
        q_odd=q_odd,
        groups_even_np=ge,
        groups_odd_np=go,
        groups_even_t={q: torch.tensor(v, dtype=torch.long) for q, v in ge.items()},
        groups_odd_t={q: torch.tensor(v, dtype=torch.long) for q, v in go.items()},
        common_diffs=common,
    )


def make_rect_spec(q_left: np.ndarray, q_right: np.ndarray) -> U1RectSpec:
    D = len(q_left)
    if len(q_right) != D:
        raise ValueError("Left and right virtual spaces must have the same D.")
    idx = []
    for spin, qphys in enumerate((-1, +1)):
        for l, ql in enumerate(q_left):
            for r, qr in enumerate(q_right):
                if qr == ql + qphys:
                    idx.append(spin * D * D + l * D + r)
    return U1RectSpec(
        q_left=np.asarray(q_left, dtype=np.int64),
        q_right=np.asarray(q_right, dtype=np.int64),
        flat_indices=torch.tensor(idx, dtype=torch.long),
        n_parameters=len(idx),
        D=D,
    )


def build_rect_tensor(raw_theta: torch.Tensor, spec: U1RectSpec) -> torch.Tensor:
    theta = raw_theta / torch.linalg.norm(raw_theta)
    flat = torch.zeros(2 * spec.D * spec.D, dtype=raw_theta.dtype, device=raw_theta.device)
    flat = flat.scatter(0, spec.flat_indices.to(raw_theta.device), theta)
    return flat.reshape(2, spec.D, spec.D)


def dense_to_raw(T: np.ndarray, spec: U1RectSpec) -> np.ndarray:
    flat = np.asarray(T, dtype=float).reshape(-1)
    return flat[spec.flat_indices.cpu().numpy()].copy()


def local_xxz_bond(delta: float) -> torch.Tensor:
    h = torch.zeros((2, 2, 2, 2))
    sz = torch.tensor([-0.5, +0.5])
    for s1 in range(2):
        for s2 in range(2):
            h[s1, s2, s1, s2] = delta * sz[s1] * sz[s2]
    h[1, 0, 0, 1] = 0.5
    h[0, 1, 1, 0] = 0.5
    return h


def transfer_torch(T: torch.Tensor) -> torch.Tensor:
    D = T.shape[1]
    return torch.einsum("slr,sLR->lLrR", T, T).reshape(D * D, D * D)


def two_tensor_bond_transfer(T1: torch.Tensor, T2: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
    D = T1.shape[1]
    C = torch.einsum("slm,tmr->stlr", T1, T2)
    return torch.einsum("abst,stlr,abLR->lLrR", h, C, C).reshape(D * D, D * D)


# =============================================================================
# PERIODIC FINITE-RING ENERGY
# =============================================================================
def periodic_two_site_energy_blocked(
    raw_A: torch.Tensor,
    raw_B: torch.Tensor,
    delta: float,
    spec_A: U1RectSpec,
    spec_B: U1RectSpec,
    ctx: MPSContext,
    L: int,
    return_parts: bool = False,
):
    if L % 2 != 0:
        raise ValueError("Two-site periodic ansatz requires even L.")

    A = build_rect_tensor(raw_A, spec_A)
    B = build_rect_tensor(raw_B, spec_B)
    EA = transfer_torch(A)
    EB = transfer_torch(B)
    h = local_xxz_bond(delta)
    EhAB = two_tensor_bond_transfer(A, B, h)
    EhBA = two_tensor_bond_transfer(B, A, h)

    cA = torch.linalg.norm(EA)
    cB = torch.linalg.norm(EB)
    EA = EA / cA
    EB = EB / cB
    EhAB = EhAB / (cA * cB)
    EhBA = EhBA / (cA * cB)

    ncell = L // 2
    norm = torch.zeros((), dtype=EA.dtype, device=EA.device)
    numAB = torch.zeros_like(norm)
    numBA = torch.zeros_like(norm)

    for q in ctx.common_diffs:
        ie = ctx.groups_even_t[q].to(EA.device)
        io = ctx.groups_odd_t[q].to(EA.device)
        ea = EA[ie[:, None], io[None, :]]
        eb = EB[io[:, None], ie[None, :]]
        te = ea @ eb
        to = eb @ ea
        pe = torch.linalg.matrix_power(te, ncell - 1)
        po = torch.linalg.matrix_power(to, ncell - 1)
        norm = norm + torch.trace(te @ pe)
        hab = EhAB[ie[:, None], ie[None, :]]
        hba = EhBA[io[:, None], io[None, :]]
        numAB = numAB + torch.trace(hab @ pe)
        numBA = numBA + torch.trace(hba @ po)

    eAB = numAB / norm
    eBA = numBA / norm
    total = (L / 2.0) * (eAB + eBA)
    if return_parts:
        return total, eAB, eBA
    return total


# =============================================================================
# NESTED-D WARM START
# =============================================================================
def occurrence_map(old_charges: np.ndarray, new_charges: np.ndarray) -> dict[int, int]:
    """Map kth occurrence of each old charge to kth occurrence in the new space."""
    new_by_q: dict[int, list[int]] = defaultdict(list)
    for j, q in enumerate(new_charges):
        new_by_q[int(q)].append(j)
    used: dict[int, int] = defaultdict(int)
    mapping: dict[int, int] = {}
    for i, q0 in enumerate(old_charges):
        q = int(q0)
        k = used[q]
        if q not in new_by_q or k >= len(new_by_q[q]):
            raise ValueError("New charge layout is not a superset of the old layout.")
        mapping[i] = new_by_q[q][k]
        used[q] += 1
    return mapping


def embed_tensor(
    old_T: np.ndarray,
    old_q_left: np.ndarray,
    old_q_right: np.ndarray,
    new_q_left: np.ndarray,
    new_q_right: np.ndarray,
) -> np.ndarray:
    Dnew = len(new_q_left)
    out = np.zeros((2, Dnew, Dnew), dtype=float)
    ml = occurrence_map(old_q_left, new_q_left)
    mr = occurrence_map(old_q_right, new_q_right)
    for s in (0, 1):
        for i_old, i_new in ml.items():
            for j_old, j_new in mr.items():
                out[s, i_new, j_new] = old_T[s, i_old, j_old]
    return out


def initial_raw_from_previous(
    prev: TwoSiteResult | None,
    spec_A: U1RectSpec,
    spec_B: U1RectSpec,
    ctx: MPSContext,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    if prev is None:
        return (
            0.1 * rng.standard_normal(spec_A.n_parameters),
            0.1 * rng.standard_normal(spec_B.n_parameters),
        )

    Anew = embed_tensor(prev.A, prev.q_even, prev.q_odd, ctx.q_even, ctx.q_odd)
    Bnew = embed_tensor(prev.B, prev.q_odd, prev.q_even, ctx.q_odd, ctx.q_even)
    rawA = dense_to_raw(Anew, spec_A)
    rawB = dense_to_raw(Bnew, spec_B)

    # Tiny noise activates newly introduced variational directions.
    rawA += WARM_START_NOISE * rng.standard_normal(rawA.shape)
    rawB += WARM_START_NOISE * rng.standard_normal(rawB.shape)
    return rawA, rawB


# =============================================================================
# OPTIMIZATION
# =============================================================================
def optimize_two_site(
    D: int,
    L: int,
    delta: float,
    prev: TwoSiteResult | None,
    adam_steps: int,
    lbfgs_steps: int,
    seed: int,
) -> TwoSiteResult:
    ctx = make_context(D)
    specA = make_rect_spec(ctx.q_even, ctx.q_odd)
    specB = make_rect_spec(ctx.q_odd, ctx.q_even)

    rng = np.random.default_rng(seed + 1009 * D + L)
    initA, initB = initial_raw_from_previous(prev, specA, specB, ctx, rng)
    rawA = torch.nn.Parameter(torch.tensor(initA))
    rawB = torch.nn.Parameter(torch.tensor(initB))

    with torch.no_grad():
        e0 = float(periodic_two_site_energy_blocked(rawA, rawB, delta, specA, specB, ctx, L))
    print(f"D={D:2d}: initial E/L = {e0 / L:.12f}  "
          f"(nA={specA.n_parameters}, nB={specB.n_parameters})")

    opt = torch.optim.Adam([rawA, rawB], lr=ADAM_LR)
    for _ in range(adam_steps):
        opt.zero_grad()
        e = periodic_two_site_energy_blocked(rawA, rawB, delta, specA, specB, ctx, L)
        e.backward()
        opt.step()

    lbfgs = torch.optim.LBFGS(
        [rawA, rawB],
        max_iter=lbfgs_steps,
        tolerance_grad=1e-11,
        tolerance_change=1e-14,
        line_search_fn="strong_wolfe",
    )

    def closure():
        lbfgs.zero_grad()
        ee = periodic_two_site_energy_blocked(rawA, rawB, delta, specA, specB, ctx, L)
        ee.backward()
        return ee

    lbfgs.step(closure)

    lbfgs.zero_grad()
    e, eAB, eBA = periodic_two_site_energy_blocked(
        rawA, rawB, delta, specA, specB, ctx, L, return_parts=True
    )
    e.backward()
    grad = float(torch.sqrt(rawA.grad.square().sum() + rawB.grad.square().sum()).detach())

    with torch.no_grad():
        A = build_rect_tensor(rawA, specA).cpu().numpy().copy()
        B = build_rect_tensor(rawB, specB).cpu().numpy().copy()
        thA = (rawA / torch.linalg.norm(rawA)).cpu().numpy().copy()
        thB = (rawB / torch.linalg.norm(rawB)).cpu().numpy().copy()

    result = TwoSiteResult(
        energy=float(e.detach()),
        A=A,
        B=B,
        theta_A=thA,
        theta_B=thB,
        q_even=ctx.q_even.copy(),
        q_odd=ctx.q_odd.copy(),
        n_parameters_A=specA.n_parameters,
        n_parameters_B=specB.n_parameters,
        e_AB=float(eAB.detach()),
        e_BA=float(eBA.detach()),
        grad_norm=grad,
    )
    print(f"      optimized E/L = {result.energy / L:.12f}, "
          f"|eAB-eBA| = {abs(result.e_AB-result.e_BA):.3e}, grad = {grad:.3e}")
    return result


# =============================================================================
# NUMPY TRANSFER OBJECTS / CORRELATION LENGTH / SAMPLING CACHE
# =============================================================================
def transfer_numpy(T: np.ndarray) -> np.ndarray:
    D = T.shape[1]
    return np.einsum("slr,sLR->lLrR", T, T).reshape(D * D, D * D)


def local_transfer_numpy(Ts: np.ndarray) -> np.ndarray:
    D = Ts.shape[0]
    return np.einsum("lr,LR->lLrR", Ts, Ts).reshape(D * D, D * D)


def block_spectrum_cell(EA: np.ndarray, EB: np.ndarray, ctx: MPSContext) -> np.ndarray:
    mags: list[float] = []
    for q in ctx.common_diffs:
        ie = ctx.groups_even_np[q]
        io = ctx.groups_odd_np[q]
        ea = EA[np.ix_(ie, io)]
        eb = EB[np.ix_(io, ie)]
        vals = np.linalg.eigvals(ea @ eb)
        mags.extend(np.abs(vals).tolist())
    if not mags:
        return np.array([], dtype=float)
    return np.sort(np.asarray(mags, dtype=float))[::-1]


def make_transfer_cache(A: np.ndarray, B: np.ndarray, ctx: MPSContext, L: int) -> TransferCache:
    EAraw = transfer_numpy(A)
    EBraw = transfer_numpy(B)
    mags = block_spectrum_cell(EAraw, EBraw, ctx)
    if len(mags) == 0 or not np.isfinite(mags[0]) or mags[0] <= 0:
        raise RuntimeError("Invalid transfer-matrix spectrum.")
    rho = float(mags[0])
    xi_sites = (
        float(-2.0 / np.log(mags[1] / mags[0]))
        if len(mags) > 1 and 0 < mags[1] < mags[0]
        else float("inf")
    )

    nA = float(np.linalg.norm(EAraw))
    nB = float(np.linalg.norm(EBraw))
    scaleA = math.sqrt(rho * nA / nB)
    scaleB = math.sqrt(rho * nB / nA)
    EA = EAraw / scaleA
    EB = EBraw / scaleB
    Efull = (EA, EB)

    Eblocks: list[dict[int, np.ndarray]] = []
    Esblocks: list[tuple[dict[int, np.ndarray], dict[int, np.ndarray]]] = []
    for T, E, scale, gl, gr in (
        (A, EA, scaleA, ctx.groups_even_np, ctx.groups_odd_np),
        (B, EB, scaleB, ctx.groups_odd_np, ctx.groups_even_np),
    ):
        Eblocks.append({q: E[np.ix_(gl[q], gr[q])] for q in ctx.common_diffs})
        spin_dicts = []
        for s in (0, 1):
            Es = local_transfer_numpy(T[s]) / scale
            spin_dicts.append({q: Es[np.ix_(gl[q], gr[q])] for q in ctx.common_diffs})
        Esblocks.append(tuple(spin_dicts))

    suffix_all: list[dict[int, np.ndarray] | None] = [None] * (L + 1)
    suffix_all[L] = {q: np.eye(len(ctx.groups_even_np[q])) for q in ctx.common_diffs}
    for j in range(L - 1, -1, -1):
        p = j & 1
        assert suffix_all[j + 1] is not None
        suffix_all[j] = {q: Eblocks[p][q] @ suffix_all[j + 1][q] for q in ctx.common_diffs}

    assert suffix_all[0] is not None
    sector_weights = {q: float(np.trace(suffix_all[0][q]).real) for q in ctx.common_diffs}
    total = float(sum(sector_weights.values()))
    if not np.isfinite(total) or total <= 0:
        raise RuntimeError("Invalid periodic MPS norm in transfer-sector decomposition.")

    retained = tuple(
        q for q in ctx.common_diffs
        if abs(sector_weights[q]) / abs(total) >= SAMPLING_SECTOR_TOL
    )
    if 0 not in retained:
        retained = tuple(sorted(set(retained) | {0}))
    kept_total = float(sum(sector_weights[q] for q in retained))
    discarded = float(abs(total - kept_total) / abs(total))

    suffix: list[dict[int, np.ndarray] | None] = [None] * (L + 1)
    suffix[L] = {q: np.eye(len(ctx.groups_even_np[q])) for q in retained}
    for j in range(L - 1, -1, -1):
        p = j & 1
        assert suffix[j + 1] is not None
        suffix[j] = {q: Eblocks[p][q] @ suffix[j + 1][q] for q in retained}

    return TransferCache(
        A=A,
        B=B,
        scale_A=scaleA,
        scale_B=scaleB,
        E_full=Efull,
        E_blocks=(Eblocks[0], Eblocks[1]),
        Es_blocks=(Esblocks[0], Esblocks[1]),
        suffix_blocks=[x for x in suffix if x is not None],
        retained_diffs=retained,
        sector_norm_weights=sector_weights,
        discarded_norm_fraction=discarded,
        norm_retained=kept_total,
        xi_sites=xi_sites,
    )


def range_transfer_full(cache: TransferCache, D: int, start: int, stop: int) -> np.ndarray:
    out = np.eye(D * D)
    for j in range(start, stop):
        out = out @ cache.E_full[j & 1]
    return out


def range_transfer_blocks(
    cache: TransferCache,
    ctx: MPSContext,
    start: int,
    stop: int,
) -> dict[int, np.ndarray]:
    g0 = ctx.groups_even_np if start % 2 == 0 else ctx.groups_odd_np
    out = {q: np.eye(len(g0[q])) for q in cache.retained_diffs}
    for j in range(start, stop):
        p = j & 1
        out = {q: out[q] @ cache.E_blocks[p][q] for q in cache.retained_diffs}
    return out


def block_trace_product(A: np.ndarray, B: np.ndarray) -> float:
    return float(np.einsum("ij,ji->", A, B, optimize=True).real)


def renormalize_block_prefix(left: dict[int, np.ndarray]) -> tuple[dict[int, np.ndarray], float]:
    scale = max(float(np.linalg.norm(M)) for M in left.values())
    if not np.isfinite(scale) or scale <= 0:
        raise RuntimeError("Block prefix became non-finite or zero.")
    return {q: M / scale for q, M in left.items()}, math.log(scale)


# =============================================================================
# FIXED-FRACTION MEASUREMENT GEOMETRY (NUMERICAL ONLY)
# =============================================================================
def fixed_fraction_lengths(L: int) -> tuple[int, int, int, int]:
    if L % 8 != 0:
        raise ValueError("Sampling geometry (L/8,L/4,3L/8,L/4) requires L divisible by 8.")
    return (L // 8, L // 4, 3 * L // 8, L // 4)


def sample_geometry_record(
    cache: TransferCache,
    ctx: MPSContext,
    L: int,
    lengths: tuple[int, int, int, int],
    rng: np.random.Generator,
):
    n_a1, n_b1, n_a2, _ = lengths
    x1 = n_a1
    x2 = x1 + n_b1
    x3 = x2 + n_a2

    left = range_transfer_blocks(cache, ctx, 0, x1)
    log_scale = 0.0
    bits1: list[int] = []
    for j in range(x1, x2):
        p = j & 1
        candidates = []
        weights = []
        for s in (0, 1):
            c = {q: left[q] @ cache.Es_blocks[p][s][q] for q in cache.retained_diffs}
            w = sum(
                block_trace_product(c[q], cache.suffix_blocks[j + 1][q])
                for q in cache.retained_diffs
            )
            candidates.append(c)
            weights.append(max(float(w), 0.0))
        total = weights[0] + weights[1]
        if total <= 0 or not np.isfinite(total):
            raise RuntimeError("Invalid Born weights while sampling B1.")
        s = int(rng.random() < weights[1] / total)
        bits1.append(s)
        left, add = renormalize_block_prefix(candidates[s])
        log_scale += add

    middle = range_transfer_blocks(cache, ctx, x2, x3)
    left = {q: left[q] @ middle[q] for q in cache.retained_diffs}
    left, add = renormalize_block_prefix(left)
    log_scale += add

    bits2: list[int] = []
    for j in range(x3, L):
        p = j & 1
        candidates = []
        weights = []
        for s in (0, 1):
            c = {q: left[q] @ cache.Es_blocks[p][s][q] for q in cache.retained_diffs}
            w = sum(
                block_trace_product(c[q], cache.suffix_blocks[j + 1][q])
                for q in cache.retained_diffs
            )
            candidates.append(c)
            weights.append(max(float(w), 0.0))
        total = weights[0] + weights[1]
        if total <= 0 or not np.isfinite(total):
            raise RuntimeError("Invalid Born weights while sampling B2.")
        s = int(rng.random() < weights[1] / total)
        bits2.append(s)
        left, add = renormalize_block_prefix(candidates[s])
        log_scale += add

    tr = sum(float(np.trace(left[q]).real) for q in cache.retained_diffs)
    if tr <= 0:
        p_record = 0.0
    else:
        logp = log_scale + math.log(tr) - math.log(cache.norm_retained)
        p_record = float(math.exp(logp)) if logp > -745 else 0.0
    return bits1, bits2, p_record


# =============================================================================
# CONDITIONAL SCHMIDT SPECTRUM
# =============================================================================
def physical_charge_pair_groups(q_left: np.ndarray, q_right: np.ndarray):
    D = len(q_left)
    groups: dict[int, list[int]] = defaultdict(list)
    pairs: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for l in range(D):
        for r in range(D):
            Q = int(q_right[r] - q_left[l])
            groups[Q].append(l * D + r)
            pairs[Q].append((l, r))
    return (
        {Q: np.asarray(v, dtype=np.int64) for Q, v in groups.items()},
        {Q: np.asarray(v, dtype=np.int64) for Q, v in pairs.items()},
    )


def gram_from_transfer_product(Eprod: np.ndarray, D: int) -> np.ndarray:
    T = Eprod.reshape(D, D, D, D)
    G = T.transpose(0, 2, 1, 3).reshape(D * D, D * D)
    return 0.5 * (G + G.T)


def gram_sqrt_blocks(Eprod: np.ndarray, q_left: np.ndarray, q_right: np.ndarray):
    D = len(q_left)
    G = gram_from_transfer_product(Eprod, D)
    ids, pairs = physical_charge_pair_groups(q_left, q_right)
    out = {}
    for Q, ix in ids.items():
        block = G[np.ix_(ix, ix)]
        vals, vecs = np.linalg.eigh(block)
        scale = max(float(np.max(np.abs(vals))), np.finfo(float).tiny)
        vals[vals < SCHMIDT_TOL * scale] = 0.0
        vals = np.maximum(vals, 0.0)
        if np.max(vals) <= 0:
            continue
        sq = (vecs * np.sqrt(vals)) @ vecs.T
        out[Q] = (sq, pairs[Q])
    return out


def fixed_block_matrix(cache: TransferCache, D: int, start: int, bits: list[int]) -> np.ndarray:
    P = np.eye(D)
    for r, s in enumerate(bits):
        T = cache.A if (start + r) % 2 == 0 else cache.B
        P = P @ T[s]
    return P


def measurement_charge(bits: list[int]) -> int:
    return int(sum(+1 if s else -1 for s in bits))


def conditional_schmidt_probabilities_blocked(
    cache: TransferCache,
    D: int,
    b1_start: int,
    bits1: list[int],
    b2_start: int,
    bits2: list[int],
    sqrtG1_blocks: dict,
    sqrtG2_blocks: dict,
) -> np.ndarray:
    X = fixed_block_matrix(cache, D, b1_start, bits1)
    Y = fixed_block_matrix(cache, D, b2_start, bits2)
    QX = measurement_charge(bits1)
    QY = measurement_charge(bits2)

    weights: list[float] = []
    for Q1, (sqrt1, pairs1) in sqrtG1_blocks.items():
        Q2 = -Q1 - QX - QY
        if Q2 not in sqrtG2_blocks:
            continue
        sqrt2, pairs2 = sqrtG2_blocks[Q2]
        i = pairs1[:, 0]
        j = pairs1[:, 1]
        k = pairs2[:, 0]
        l = pairs2[:, 1]
        K = X[j[:, None], k[None, :]] * Y[l[None, :], i[:, None]]
        if not np.any(K):
            continue
        C = sqrt1 @ K @ sqrt2
        gram = C @ C.T if C.shape[0] <= C.shape[1] else C.T @ C
        lam = np.linalg.eigvalsh(0.5 * (gram + gram.T))
        lam = np.maximum(lam, 0.0)
        weights.extend(lam.tolist())

    if not weights:
        return np.array([], dtype=float)
    w = np.asarray(weights, dtype=float)
    total = float(w.sum())
    if total <= 0 or not np.isfinite(total):
        return np.array([], dtype=float)
    p = w / total
    p = p[p > SCHMIDT_TOL]
    p /= p.sum()
    return p


def renyi_entropy(p: np.ndarray, alpha: float) -> float:
    if len(p) == 0:
        return float("nan")
    return float(np.log(np.sum(p ** alpha)) / (1.0 - alpha))


def record_id(bits: list[int]) -> int:
    out = 0
    for r, b in enumerate(bits):
        out |= int(b) << r
    return out


def magnetization_observables(bits1: list[int], bits2: list[int]) -> dict[str, float | int]:
    """Pure lattice observables from the two measured blocks.

    A bit 1 is spin up (+1) and a bit 0 is spin down (-1).  For each measured
    block Br, define the extensive magnetization M_Br = N_up - N_down and the
    magnetization density m_Br = M_Br / |Br|.  The imbalance used in the plots is

        delta_m = m_B1 - m_B2.

    This is computed only from the sampled spin outcomes; no continuum or CFT
    quantity enters the definition.
    """
    M1 = measurement_charge(bits1)
    M2 = measurement_charge(bits2)
    m1 = M1 / len(bits1) if bits1 else float("nan")
    m2 = M2 / len(bits2) if bits2 else float("nan")
    return {
        "M_B1": M1,
        "M_B2": M2,
        "m_B1": float(m1),
        "m_B2": float(m2),
        "delta_m": float(m1 - m2),
        "m_measured_total": float((M1 + M2) / (len(bits1) + len(bits2))),
    }


def analyze_numerical_entanglement(
    D: int,
    L: int,
    mps: TwoSiteResult,
    cache: TransferCache,
    ctx: MPSContext,
    n_records: int,
    rng_seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    lengths = fixed_fraction_lengths(L)
    n_a1, n_b1, n_a2, _ = lengths
    x1 = n_a1
    x2 = x1 + n_b1
    x3 = x2 + n_a2

    q0 = ctx.q_even
    q1 = ctx.q_even if x1 % 2 == 0 else ctx.q_odd
    q2 = ctx.q_even if x2 % 2 == 0 else ctx.q_odd
    q3 = ctx.q_even if x3 % 2 == 0 else ctx.q_odd

    G1prod = range_transfer_full(cache, D, 0, x1)
    G2prod = range_transfer_full(cache, D, x2, x3)
    sqrtG1 = gram_sqrt_blocks(G1prod, q0, q1)
    sqrtG2 = gram_sqrt_blocks(G2prod, q2, q3)

    rng = np.random.default_rng(rng_seed + 10007 * D + L)
    rec_cache: dict[tuple[int, int], dict] = {}
    rows = []
    for traj in range(n_records):
        b1, b2, p_record = sample_geometry_record(cache, ctx, L, lengths, rng)
        key = (record_id(b1), record_id(b2))
        if key not in rec_cache:
            p = conditional_schmidt_probabilities_blocked(
                cache, D, x1, b1, x3, b2, sqrtG1, sqrtG2
            )
            vals = {
                "record_id_B1": key[0],
                "record_id_B2": key[1],
                "record": "".join(str(int(x)) for x in (b1 + b2)),
                "schmidt_rank": int(len(p)),
                **magnetization_observables(b1, b2),
            }
            for alpha in ALPHAS:
                vals[f"S_mps_a{str(alpha).replace('.', 'p')}"] = renyi_entropy(p, alpha)
            rec_cache[key] = vals

        vals = dict(rec_cache[key])
        vals["p_record"] = p_record
        rows.append({"D": D, "L": L, "trajectory": traj, **vals})
        if (traj + 1) % 100 == 0:
            print(f"      D={D}: sampled {traj+1}/{n_records} trajectories")

    df = pd.DataFrame(rows)
    summary_rows = []
    for alpha in ALPHAS:
        col = f"S_mps_a{str(alpha).replace('.', 'p')}"
        x = df[col].to_numpy(float)
        x = x[np.isfinite(x)]
        n = len(x)
        mean = float(np.mean(x)) if n else float("nan")
        std = float(np.std(x, ddof=1)) if n > 1 else float("nan")
        sem = float(std / math.sqrt(n)) if n > 1 else float("nan")
        summary_rows.append({
            "D": D,
            "L": L,
            "alpha": alpha,
            "mean": mean,
            "std": std,
            "sem": sem,
            "median": float(np.median(x)) if n else float("nan"),
            "q25": float(np.quantile(x, 0.25)) if n else float("nan"),
            "q75": float(np.quantile(x, 0.75)) if n else float("nan"),
            "n_valid": n,
            "n_unique_records": len(rec_cache),
        })
    return df, pd.DataFrame(summary_rows)


# =============================================================================
# PLOTS, TRANSFER-MATRIX DIAGNOSTICS, AND CONVERGENCE REPORT
# =============================================================================
def plot_energy(summary: pd.DataFrame, outdir: Path):
    convergence = outdir / "plots" / "convergence"
    diagnostics = outdir / "plots" / "diagnostics"
    convergence.mkdir(parents=True, exist_ok=True)
    diagnostics.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    ax.plot(summary["D"], summary["E_per_site"], marker="o")
    ax.set_xlabel("Bond dimension D")
    ax.set_ylabel("Variational energy per site E/L")
    ax.set_title("XXZ MPS energy convergence")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(convergence / "energy_per_site_vs_D.png", dpi=180)
    plt.close(fig)

    ref = float(summary.iloc[-1]["E_per_site"])
    err = np.abs(summary["E_per_site"].to_numpy(float) - ref)
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    positive = err > 0
    ax.plot(summary.loc[positive, "D"], err[positive], marker="o")
    ax.set_yscale("log")
    ax.set_xlabel("Bond dimension D")
    ax.set_ylabel("|E(D)/L - E(Dmax)/L|")
    ax.set_title(f"Energy difference from D={int(summary.iloc[-1]['D'])}")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(convergence / "energy_difference_vs_D.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    ax.plot(summary["D"], summary["xi_sites"], marker="o")
    ax.set_xlabel("Bond dimension D")
    ax.set_ylabel("Effective MPS correlation length ξ(D) [sites]")
    ax.set_title("Effective finite-D correlation length (diagnostic)")
    ax.text(
        0.03, 0.97,
        "Critical XXZ: ξ(D) may keep increasing with D\n(not a finite-value convergence criterion)",
        transform=ax.transAxes, va="top", fontsize=9,
    )
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(diagnostics / "effective_correlation_length_vs_D.png", dpi=180)
    plt.close(fig)


def plot_renyi(renyi: pd.DataFrame, outdir: Path):
    if renyi.empty:
        return
    plots = outdir / "plots" / "convergence"
    plots.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6.8, 4.8))
    for alpha in sorted(renyi["alpha"].unique()):
        sub = renyi[renyi["alpha"] == alpha].sort_values("D")
        ax.errorbar(sub["D"], sub["mean"], yerr=sub["sem"], marker="o", capsize=3, label=f"α={alpha:g}")
    ax.set_xlabel("Bond dimension D")
    ax.set_ylabel("Mean conditional Rényi entropy")
    ax.set_title("Numerical Rényi convergence with bond dimension")
    ax.legend()
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(plots / "renyi_mean_vs_D.png", dpi=180)
    plt.close(fig)

    Dref = int(renyi["D"].max())
    ref = renyi[renyi["D"] == Dref].set_index("alpha")["mean"]
    fig, ax = plt.subplots(figsize=(6.8, 4.8))
    for alpha in sorted(renyi["alpha"].unique()):
        sub = renyi[renyi["alpha"] == alpha].sort_values("D").copy()
        sub["diff"] = np.abs(sub["mean"] - float(ref.loc[alpha]))
        sub = sub[sub["diff"] > 0]
        ax.plot(sub["D"], sub["diff"], marker="o", label=f"α={alpha:g}")
    ax.set_yscale("log")
    ax.set_xlabel("Bond dimension D")
    ax.set_ylabel(f"|mean Sα(D) - mean Sα(D={Dref})|")
    ax.set_title("Rényi difference from largest-D result")
    ax.legend()
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(plots / "renyi_difference_vs_reference.png", dpi=180)
    plt.close(fig)


def summarize_delta_m(trajectory_frames: list[pd.DataFrame]) -> pd.DataFrame:
    """Bin the trajectory entropies by the exact discrete value of delta_m."""
    if not trajectory_frames:
        return pd.DataFrame()
    df = pd.concat(trajectory_frames, ignore_index=True)
    rows = []
    for (D, L, dm), group in df.groupby(["D", "L", "delta_m"], sort=True):
        for alpha in ALPHAS:
            col = f"S_mps_a{str(alpha).replace('.', 'p')}"
            x = group[col].to_numpy(float)
            x = x[np.isfinite(x)]
            n = len(x)
            if not n:
                continue
            std = float(np.std(x, ddof=1)) if n > 1 else 0.0
            rows.append({
                "D": int(D),
                "L": int(L),
                "alpha": float(alpha),
                "delta_m": float(dm),
                "mean_entropy": float(np.mean(x)),
                "std_entropy": std,
                "sem_entropy": float(std / math.sqrt(n)) if n > 1 else 0.0,
                "count": int(n),
            })
    return pd.DataFrame(rows)


def plot_delta_m(delta_summary: pd.DataFrame, trajectory_frames: list[pd.DataFrame], outdir: Path, candidate_D: int):
    """Separate S_alpha vs delta_m plots for the selected bond dimension."""
    if delta_summary.empty or not trajectory_frames:
        return
    plots = outdir / "plots" / "delta_m"
    plots.mkdir(parents=True, exist_ok=True)
    all_traj = pd.concat(trajectory_frames, ignore_index=True)
    if candidate_D not in set(all_traj["D"].astype(int)):
        candidate_D = int(all_traj["D"].max())

    traj = all_traj[all_traj["D"] == candidate_D]
    binned = delta_summary[delta_summary["D"] == candidate_D]
    for alpha in ALPHAS:
        tag = str(alpha).replace('.', 'p')
        col = f"S_mps_a{tag}"
        sub = binned[binned["alpha"] == alpha].sort_values("delta_m")
        fig, ax = plt.subplots(figsize=(6.6, 4.6))
        ax.scatter(traj["delta_m"], traj[col], s=10, alpha=0.18, label="Born trajectories")
        ax.errorbar(
            sub["delta_m"], sub["mean_entropy"], yerr=sub["sem_entropy"],
            marker="o", linewidth=1.5, capsize=3, label="Mean ± SEM"
        )
        ax.set_xlabel(r"Magnetization imbalance $\Delta m = m_{B_1}-m_{B_2}$")
        ax.set_ylabel(rf"Conditional Rényi entropy $S_{{{alpha:g}}}$")
        ax.set_title(f"XXZ MPS entanglement vs magnetization imbalance (L={int(traj['L'].iloc[0])}, D={candidate_D})")
        ax.grid(True, alpha=0.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(plots / f"entanglement_vs_delta_m_D{candidate_D}_alpha{tag}.png", dpi=180)
        plt.close(fig)


def convergence_report(
    ground: pd.DataFrame,
    renyi: pd.DataFrame,
    candidate_D: int,
    energy_tol: float,
    entropy_tol: float,
    sigma_factor: float,
) -> dict:
    Ds = sorted(int(x) for x in ground["D"].unique())
    if candidate_D not in Ds:
        return {
            "candidate_D": candidate_D,
            "status": "not_evaluated",
            "reason": f"D={candidate_D} was not run.",
        }
    larger = [D for D in Ds if D > candidate_D]
    if not larger:
        return {
            "candidate_D": candidate_D,
            "status": "insufficient_reference",
            "reason": "Run at least one D larger than the candidate to test convergence.",
        }

    reference_D = min(larger)
    c = ground[ground["D"] == candidate_D].iloc[0]
    r = ground[ground["D"] == reference_D].iloc[0]
    dE = abs(float(c["E_per_site"]) - float(r["E_per_site"]))
    energy_pass = bool(dE <= energy_tol)

    observable_checks = []
    entropy_pass = True
    if not renyi.empty and candidate_D in set(renyi["D"]) and reference_D in set(renyi["D"]):
        for alpha in sorted(renyi["alpha"].unique()):
            ca = renyi[(renyi["D"] == candidate_D) & (renyi["alpha"] == alpha)]
            ra = renyi[(renyi["D"] == reference_D) & (renyi["alpha"] == alpha)]
            if ca.empty or ra.empty:
                continue
            ca = ca.iloc[0]
            ra = ra.iloc[0]
            diff = abs(float(ca["mean"]) - float(ra["mean"]))
            sem_c = float(ca["sem"]) if np.isfinite(ca["sem"]) else 0.0
            sem_r = float(ra["sem"]) if np.isfinite(ra["sem"]) else 0.0
            stat_tol = sigma_factor * math.sqrt(sem_c * sem_c + sem_r * sem_r)
            allowed = max(entropy_tol, stat_tol)
            passed = bool(diff <= allowed)
            entropy_pass = entropy_pass and passed
            observable_checks.append({
                "alpha": float(alpha),
                "mean_candidate": float(ca["mean"]),
                "mean_reference": float(ra["mean"]),
                "abs_difference": diff,
                "allowed_difference": allowed,
                "pass": passed,
            })

    overall = energy_pass and entropy_pass
    return {
        "candidate_D": candidate_D,
        "reference_D": reference_D,
        "energy_tolerance_per_site": energy_tol,
        "energy_abs_difference_per_site": dE,
        "energy_pass": energy_pass,
        "entropy_absolute_tolerance": entropy_tol,
        "entropy_sigma_factor": sigma_factor,
        "entropy_checks": observable_checks,
        "entropy_pass": entropy_pass if observable_checks else None,
        "overall_pass": overall if observable_checks else energy_pass,
        "interpretation": (
            "Candidate is numerically supported at the stated tolerances against the next larger D."
            if (overall if observable_checks else energy_pass)
            else "Candidate does not pass the stated convergence tolerances; increase D or improve optimization/statistics."
        ),
    }


# =============================================================================
# I/O
# =============================================================================
def save_mps(path: Path, D: int, L: int, delta: float, mps: TwoSiteResult):
    np.savez_compressed(
        path,
        A=mps.A,
        B=mps.B,
        theta_A=mps.theta_A,
        theta_B=mps.theta_B,
        q_even=mps.q_even,
        q_odd=mps.q_odd,
        D=np.array(D),
        L=np.array(L),
        Delta=np.array(delta),
        E_MPS=np.array(mps.energy),
        e_AB=np.array(mps.e_AB),
        e_BA=np.array(mps.e_BA),
        gradient_norm=np.array(mps.grad_norm),
        n_parameters_A=np.array(mps.n_parameters_A),
        n_parameters_B=np.array(mps.n_parameters_B),
    )


def main():
    parser = argparse.ArgumentParser(description="Numerical XXZ MPS bond-dimension convergence (no CFT).")
    parser.add_argument("--L", type=int, default=DEFAULT_L)
    parser.add_argument("--Ds", type=int, nargs="+", default=list(DEFAULT_DS))
    parser.add_argument("--records", type=int, default=DEFAULT_RECORDS,
                        help="Born trajectories per D; 0 skips post-measurement entropy analysis")
    parser.add_argument("--candidate-D", type=int, default=DEFAULT_CANDIDATE_D)
    parser.add_argument("--energy-tol", type=float, default=ENERGY_TOL_PER_SITE)
    parser.add_argument("--entropy-tol", type=float, default=ENTROPY_ABS_TOL)
    parser.add_argument("--entropy-sigma", type=float, default=ENTROPY_SIGMA_FACTOR)
    parser.add_argument("--adam-steps", type=int, default=ADAM_STEPS)
    parser.add_argument("--lbfgs-steps", type=int, default=LBFGS_STEPS)
    parser.add_argument("--seed", type=int, default=OPT_SEED)
    parser.add_argument("--output", type=Path, default=OUTPUT_ROOT)
    args = parser.parse_args()

    L = int(args.L)
    if L % 2:
        raise ValueError("L must be even for the two-site periodic MPS.")
    Ds = sorted(set(int(x) for x in args.Ds))
    for D in Ds:
        charge_layout(D)  # validate early
    if args.records > 0 and L % 8:
        raise ValueError("--records > 0 requires L divisible by 8 for the fixed-fraction geometry.")

    outroot = Path(args.output)
    outroot.mkdir(parents=True, exist_ok=True)
    print("=" * 88)
    print("XXZ TWO-SITE U(1) MPS — NUMERICAL BOND-DIMENSION CONVERGENCE")
    print("=" * 88)
    print(f"L={L}, Delta={DELTA}, Ds={Ds}, records/D={args.records}")
    print("No CFT formulas are used.")

    prev: TwoSiteResult | None = None
    ground_rows = []
    renyi_frames = []
    trajectory_frames = []

    for D in Ds:
        t0 = time.time()
        ctx = make_context(D)
        mps = optimize_two_site(
            D=D,
            L=L,
            delta=DELTA,
            prev=prev,
            adam_steps=args.adam_steps,
            lbfgs_steps=args.lbfgs_steps,
            seed=args.seed,
        )
        ddir = outroot / f"D_{D}"
        ddir.mkdir(parents=True, exist_ok=True)
        save_mps(ddir / "optimized_mps.npz", D, L, DELTA, mps)

        print(f"      building transfer cache for D={D} ...")
        cache = make_transfer_cache(mps.A, mps.B, ctx, L)
        runtime = time.time() - t0
        ground_rows.append({
            "D": D,
            "L": L,
            "Delta": DELTA,
            "E_MPS": mps.energy,
            "E_per_site": mps.energy / L,
            "e_AB": mps.e_AB,
            "e_BA": mps.e_BA,
            "dimerization_energy_abs": abs(mps.e_AB - mps.e_BA),
            "gradient_norm": mps.grad_norm,
            "xi_sites": cache.xi_sites,
            "L_over_xi": L / cache.xi_sites if np.isfinite(cache.xi_sites) else 0.0,
            "n_parameters_A": mps.n_parameters_A,
            "n_parameters_B": mps.n_parameters_B,
            "discarded_sampling_norm_fraction": cache.discarded_norm_fraction,
            "runtime_seconds": runtime,
        })

        renyi_summary = pd.DataFrame()
        if args.records > 0:
            print(f"      numerical post-measurement entropy analysis for D={D} ...")
            traj, renyi_summary = analyze_numerical_entanglement(
                D, L, mps, cache, ctx, args.records, RNG_SEED
            )
            traj.to_csv(ddir / "trajectory_results.csv", index=False)
            renyi_summary.to_csv(ddir / "renyi_summary.csv", index=False)
            renyi_frames.append(renyi_summary)
            trajectory_frames.append(traj)

        metadata = {
            "model": "spin-1/2 periodic XXZ",
            "Delta": DELTA,
            "L": L,
            "D": D,
            "ansatz": "two-site periodic U(1)-symmetric MPS",
            "records": int(args.records),
            "alphas": list(ALPHAS) if args.records > 0 else [],
            "even_bond_charges": ctx.q_even.tolist(),
            "odd_bond_charges": ctx.q_odd.tolist(),
            "E_MPS": mps.energy,
            "E_MPS_per_site": mps.energy / L,
            "e_AB": mps.e_AB,
            "e_BA": mps.e_BA,
            "gradient_norm": mps.grad_norm,
            "correlation_length_sites": cache.xi_sites,
            "discarded_sampling_norm_fraction": cache.discarded_norm_fraction,
            "runtime_seconds": runtime,
        }
        with open(ddir / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        prev = mps

    ground = pd.DataFrame(ground_rows).sort_values("D")
    ground.to_csv(outroot / "bond_dimension_summary.csv", index=False)
    renyi = pd.concat(renyi_frames, ignore_index=True) if renyi_frames else pd.DataFrame()
    if not renyi.empty:
        renyi.to_csv(outroot / "renyi_summary.csv", index=False)

    delta_summary = summarize_delta_m(trajectory_frames)
    if not delta_summary.empty:
        delta_summary.to_csv(outroot / "delta_m_summary.csv", index=False)

    plot_energy(ground, outroot)
    plot_renyi(renyi, outroot)
    plot_delta_m(delta_summary, trajectory_frames, outroot, int(args.candidate_D))

    report = convergence_report(
        ground=ground,
        renyi=renyi,
        candidate_D=int(args.candidate_D),
        energy_tol=float(args.energy_tol),
        entropy_tol=float(args.entropy_tol),
        sigma_factor=float(args.entropy_sigma),
    )
    with open(outroot / "convergence_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("\nSaved:")
    print(f"  {outroot / 'bond_dimension_summary.csv'}")
    if not renyi.empty:
        print(f"  {outroot / 'renyi_summary.csv'}")
    if not delta_summary.empty:
        print(f"  {outroot / 'delta_m_summary.csv'}")
    print(f"  {outroot / 'convergence_report.json'}")
    print(f"  {outroot / 'plots'}")
    print("  Note: xi(D) is a transfer-matrix diagnostic and is not part of the convergence pass/fail test.")
    print("\nConvergence assessment:")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
